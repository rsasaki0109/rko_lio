/*
 * MIT License
 *
 * Copyright (c) 2026 Sasaki
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Scan-to-scan reflectivity/intensity texture correlation along a single
// world-frame axis. Used to correct the along-tunnel translation direction
// that geometric ICP under-observes in self-similar environments (see
// LIO::register_scan's intensity-constraint block). Pure functions only, no
// LIO-internal types, so this is unit-testable in isolation like
// radar_ego_velocity.hpp / degeneracy_aware_solve.hpp.

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace rko_lio::core {

struct IntensityProfileConfig {
  double bin_size_m = 0.25;
  double half_length_m = 30.0;
  double max_shift_m = 1.5;
  double min_correlation = 0.6;
  std::size_t min_filled_bins = 40;
};

/** 1D reflectivity profile: mean intensity per bin along a fixed world axis/origin,
 *  normalized zero-mean/unit-variance over the filled bins. `valid` is false if the
 *  scan did not fill enough bins to be trustworthy (empty/underfilled rejection). */
struct IntensityProfile {
  std::vector<double> values;
  std::vector<bool> filled;
  std::size_t filled_count = 0;
  bool valid = false;
};

struct ProfileShiftResult {
  bool valid = false;
  double shift_m = 0.0;
  double correlation = -1.0;
  std::size_t overlap_bins = 0;
};

/**
 * Bin `points` (world frame) by their projection onto `axis` relative to `origin`,
 * averaging `intensities` per bin, then normalize the filled bins to zero-mean/unit-
 * variance. `axis` need not be unit length. `points.size()` must equal `intensities.size()`.
 */
inline IntensityProfile build_intensity_profile(const std::vector<Eigen::Vector3d>& points,
                                                 const std::vector<float>& intensities,
                                                 const Eigen::Vector3d& axis,
                                                 const Eigen::Vector3d& origin,
                                                 const IntensityProfileConfig& config) {
  IntensityProfile profile;
  if (points.size() != intensities.size() || points.empty() || config.bin_size_m <= 0.0 ||
      config.half_length_m <= 0.0 || axis.squaredNorm() < 1.0e-12) {
    return profile;
  }
  const Eigen::Vector3d unit_axis = axis.normalized();
  const auto num_bins =
      static_cast<std::size_t>(std::round(2.0 * config.half_length_m / config.bin_size_m)) + 1;
  const auto center_bin = static_cast<long>(num_bins / 2);

  std::vector<double> sums(num_bins, 0.0);
  std::vector<std::size_t> counts(num_bins, 0);
  for (std::size_t i = 0; i < points.size(); ++i) {
    const double s = unit_axis.dot(points[i] - origin);
    if (std::abs(s) > config.half_length_m || !std::isfinite(intensities[i])) {
      continue;
    }
    const long bin_index = center_bin + static_cast<long>(std::floor(s / config.bin_size_m + 0.5));
    if (bin_index < 0 || bin_index >= static_cast<long>(num_bins)) {
      continue;
    }
    sums[static_cast<std::size_t>(bin_index)] += intensities[i];
    ++counts[static_cast<std::size_t>(bin_index)];
  }

  std::vector<double> means(num_bins, 0.0);
  std::vector<bool> filled(num_bins, false);
  std::size_t filled_count = 0;
  for (std::size_t b = 0; b < num_bins; ++b) {
    if (counts[b] > 0) {
      means[b] = sums[b] / static_cast<double>(counts[b]);
      filled[b] = true;
      ++filled_count;
    }
  }
  profile.filled_count = filled_count;
  if (filled_count < config.min_filled_bins) {
    return profile; // valid stays false: not enough texture observed this scan
  }

  double mean = 0.0;
  for (std::size_t b = 0; b < num_bins; ++b) {
    if (filled[b]) {
      mean += means[b];
    }
  }
  mean /= static_cast<double>(filled_count);
  double variance = 0.0;
  for (std::size_t b = 0; b < num_bins; ++b) {
    if (filled[b]) {
      const double d = means[b] - mean;
      variance += d * d;
    }
  }
  variance /= static_cast<double>(filled_count);
  const double stddev = std::sqrt(std::max(variance, 1.0e-12));

  std::vector<double> normalized(num_bins, 0.0);
  for (std::size_t b = 0; b < num_bins; ++b) {
    if (filled[b]) {
      normalized[b] = (means[b] - mean) / stddev;
    }
  }

  profile.values = std::move(normalized);
  profile.filled = std::move(filled);
  profile.valid = true;
  return profile;
}

/**
 * Estimate the along-axis shift that best aligns `profile_b` to `profile_a`, defined so
 * that profile_a[i] ~= profile_b[i - shift_bins]. `profile_a` is the earlier (reference)
 * scan's world-anchored profile; `profile_b` is the later scan's profile built at the
 * same origin/axis. A positive shift_m means the texture profile_b observes appears
 * displaced further along +axis than where profile_a placed it -- i.e. if profile_b was
 * built using an imperfect pose prediction, shift_m is the translation-along-axis
 * correction needed to bring that prediction in line with the world-fixed texture.
 * Searches integer-bin shifts up to +/-max_shift_m, then refines with a parabolic peak
 * interpolation around the best integer shift. Invalid if either profile is invalid,
 * the peak correlation is below min_correlation, or the overlap is smaller than
 * min_filled_bins.
 */
inline ProfileShiftResult estimate_profile_shift(const IntensityProfile& profile_a,
                                                  const IntensityProfile& profile_b,
                                                  const IntensityProfileConfig& config) {
  ProfileShiftResult result;
  if (!profile_a.valid || !profile_b.valid || profile_a.values.size() != profile_b.values.size() ||
      config.bin_size_m <= 0.0) {
    return result;
  }
  const auto num_bins = static_cast<long>(profile_a.values.size());
  const auto max_shift_bins =
      std::max<long>(1, static_cast<long>(std::round(config.max_shift_m / config.bin_size_m)));

  auto correlation_at = [&](const long shift_bins) -> std::pair<double, std::size_t> {
    double dot = 0.0;
    std::size_t overlap = 0;
    for (long i = 0; i < num_bins; ++i) {
      const long j = i - shift_bins;
      if (j < 0 || j >= num_bins) {
        continue;
      }
      if (!profile_a.filled[static_cast<std::size_t>(i)] || !profile_b.filled[static_cast<std::size_t>(j)]) {
        continue;
      }
      dot += profile_a.values[static_cast<std::size_t>(i)] * profile_b.values[static_cast<std::size_t>(j)];
      ++overlap;
    }
    if (overlap == 0) {
      return {-2.0, 0};
    }
    return {dot / static_cast<double>(overlap), overlap};
  };

  double best_correlation = -2.0;
  long best_shift = 0;
  std::size_t best_overlap = 0;
  for (long shift_bins = -max_shift_bins; shift_bins <= max_shift_bins; ++shift_bins) {
    const auto& [correlation, overlap] = correlation_at(shift_bins);
    if (correlation > best_correlation) {
      best_correlation = correlation;
      best_shift = shift_bins;
      best_overlap = overlap;
    }
  }

  result.overlap_bins = best_overlap;
  if (best_overlap < config.min_filled_bins || best_correlation < config.min_correlation) {
    result.correlation = std::clamp(best_correlation, -1.0, 1.0);
    return result;
  }

  // Sub-bin refinement via parabolic interpolation of the correlation peak.
  double sub_bin_offset = 0.0;
  if (best_shift > -max_shift_bins && best_shift < max_shift_bins) {
    const double c_minus = correlation_at(best_shift - 1).first;
    const double c_center = best_correlation;
    const double c_plus = correlation_at(best_shift + 1).first;
    const double denominator = c_minus - 2.0 * c_center + c_plus;
    if (std::abs(denominator) > 1.0e-9) {
      sub_bin_offset = std::clamp(0.5 * (c_minus - c_plus) / denominator, -0.5, 0.5);
    }
  }

  result.valid = true;
  result.shift_m = (static_cast<double>(best_shift) + sub_bin_offset) * config.bin_size_m;
  result.correlation = std::clamp(best_correlation, -1.0, 1.0);
  return result;
}

/**
 * Pick a unit measurement axis from a translation step (e.g. this scan's ICP
 * displacement, or a fallback initial-guess displacement), without any
 * Hessian/persistence gating. Returns nullopt if `step` is shorter than
 * `min_step_m` -- too small to trust a direction from (a near-stationary rig
 * has no along-track signal). Used by LIO::register_scan's intensity
 * velocity-disagreement gate to choose/refresh its correlation axis every
 * scan, mirroring how the radar disagreement gate uses the radar direction.
 */
inline std::optional<Eigen::Vector3d> unit_axis_from_step(const Eigen::Vector3d& step, const double min_step_m) {
  if (!(min_step_m > 0.0) || step.norm() < min_step_m) {
    return std::nullopt;
  }
  return step.normalized();
}

/**
 * Convert a profile correlation shift into an intensity-implied along-axis
 * velocity, for comparison against an independently-estimated (e.g. ICP)
 * velocity along the same axis.
 *
 * Per estimate_profile_shift's contract, `shift_m` is the translation-along-
 * `axis` correction that brings the pose used to build profile_b (here: the
 * ICP initial guess, *before* this scan's ICP correction) into agreement
 * with the world-fixed texture recorded by profile_a (the previous scan's
 * stored, world-anchored profile). So the texture-implied true position of
 * the sensor along `axis`, at this scan's time, is:
 *   true_position_along_axis = axis . initial_guess_translation + shift_m
 * profile_a was anchored at the previous scan's own optimized pose, i.e. its
 * position along axis was:
 *   previous_position_along_axis = axis . previous_origin
 * The implied displacement along axis since the previous scan is the
 * difference of those two positions; dividing by the elapsed time gives the
 * implied velocity. `dt` must be the elapsed time (s) between the two scans;
 * a non-positive `dt` returns 0 (no signal) rather than dividing by zero.
 */
inline double intensity_implied_velocity_along_axis(const Eigen::Vector3d& axis,
                                                      const Eigen::Vector3d& initial_guess_translation,
                                                      const Eigen::Vector3d& previous_origin,
                                                      const double shift_m,
                                                      const double dt) {
  if (!(dt > 0.0)) {
    return 0.0;
  }
  const double true_position_along_axis = axis.dot(initial_guess_translation) + shift_m;
  const double previous_position_along_axis = axis.dot(previous_origin);
  return (true_position_along_axis - previous_position_along_axis) / dt;
}

} // namespace rko_lio::core
