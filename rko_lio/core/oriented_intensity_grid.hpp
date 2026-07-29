/*
 * MIT License
 *
 * Copyright (c) 2026 Sasaki
 */

#pragma once

#include "correlation_peak_selector.hpp"
#include "correlation_score.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace rko_lio::core {

struct OrientedIntensityGridConfig {
  double bin_size_m = 0.25;
  double half_length_m = 30.0;
  double half_width_m = 5.0;
  double max_longitudinal_shift_m = 1.5;
  double max_lateral_shift_m = 0.5;
  double intensity_weight = 1.0;
  double height_weight = 0.25;
  double min_correlation = 0.6;
  double min_peak_margin = 0.0;
  std::size_t peak_exclusion_radius_bins = 1;
  std::size_t min_filled_cells = 40;
};

/** Mean reflectivity and height in a fixed, oriented world-frame grid. */
struct OrientedIntensityGrid {
  std::size_t longitudinal_bins = 0;
  std::size_t lateral_bins = 0;
  std::vector<double> intensity;
  std::vector<double> height;
  std::vector<bool> filled;
  std::size_t filled_count = 0;
  bool valid = false;

  [[nodiscard]] std::size_t index(
      const std::size_t longitudinal,
      const std::size_t lateral) const {
    return longitudinal * lateral_bins + lateral;
  }
};

struct OrientedGridShiftResult {
  bool valid = false;
  double longitudinal_shift_m = 0.0;
  double lateral_shift_m = 0.0;
  double correlation = -1.0;
  double second_best_correlation = -1.0;
  double peak_margin = 0.0;
  std::size_t overlap_cells = 0;
  bool has_competing_peak = false;
  bool ambiguous = false;
  /** Channel scores at the selected integer-bin peak. The combined score is
   *  used for peak selection and sub-bin refinement. Keeping the raw channel
   *  scores makes physical-scene confidence policies possible without
   *  coupling them to the matcher. */
  double intensity_correlation = -2.0;
  double height_correlation = -2.0;
};

struct OrientedGridCorrelationScore {
  double combined = -2.0;
  double intensity = -2.0;
  double height = -2.0;
  std::size_t support = 0;
};

inline OrientedIntensityGrid build_oriented_intensity_grid(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<float>& intensities,
    const Eigen::Vector3d& longitudinal_axis,
    const Eigen::Vector3d& lateral_axis,
    const Eigen::Vector3d& origin,
    const OrientedIntensityGridConfig& config) {
  OrientedIntensityGrid grid;
  if (points.empty() || points.size() != intensities.size() ||
      config.bin_size_m <= 0.0 || config.half_length_m <= 0.0 ||
      config.half_width_m <= 0.0 ||
      longitudinal_axis.squaredNorm() <= 1.0e-12 ||
      lateral_axis.squaredNorm() <= 1.0e-12) {
    return grid;
  }

  const Eigen::Vector3d longitudinal = longitudinal_axis.normalized();
  Eigen::Vector3d lateral =
      lateral_axis - longitudinal * longitudinal.dot(lateral_axis);
  if (lateral.squaredNorm() <= 1.0e-12) {
    return grid;
  }
  lateral.normalize();
  const Eigen::Vector3d normal = longitudinal.cross(lateral).normalized();

  const auto longitudinal_radius = static_cast<long>(
      std::ceil(config.half_length_m / config.bin_size_m));
  const auto lateral_radius = static_cast<long>(
      std::ceil(config.half_width_m / config.bin_size_m));
  grid.longitudinal_bins =
      static_cast<std::size_t>(2 * longitudinal_radius + 1);
  grid.lateral_bins = static_cast<std::size_t>(2 * lateral_radius + 1);
  const std::size_t cell_count =
      grid.longitudinal_bins * grid.lateral_bins;
  std::vector<double> intensity_sums(cell_count, 0.0);
  std::vector<double> height_sums(cell_count, 0.0);
  std::vector<std::size_t> counts(cell_count, 0);

  for (std::size_t point_index = 0; point_index < points.size();
       ++point_index) {
    const Eigen::Vector3d relative = points[point_index] - origin;
    const long longitudinal_index =
        static_cast<long>(std::llround(
            longitudinal.dot(relative) / config.bin_size_m)) +
        longitudinal_radius;
    const long lateral_index =
        static_cast<long>(
            std::llround(lateral.dot(relative) / config.bin_size_m)) +
        lateral_radius;
    if (longitudinal_index < 0 || lateral_index < 0 ||
        longitudinal_index >=
            static_cast<long>(grid.longitudinal_bins) ||
        lateral_index >= static_cast<long>(grid.lateral_bins)) {
      continue;
    }
    const std::size_t cell = grid.index(
        static_cast<std::size_t>(longitudinal_index),
        static_cast<std::size_t>(lateral_index));
    intensity_sums[cell] += static_cast<double>(intensities[point_index]);
    height_sums[cell] += normal.dot(relative);
    ++counts[cell];
  }

  grid.intensity.resize(cell_count, 0.0);
  grid.height.resize(cell_count, 0.0);
  grid.filled.resize(cell_count, false);
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    if (counts[cell] == 0) {
      continue;
    }
    const double count = static_cast<double>(counts[cell]);
    grid.intensity[cell] = intensity_sums[cell] / count;
    grid.height[cell] = height_sums[cell] / count;
    grid.filled[cell] = true;
    ++grid.filled_count;
  }
  grid.valid = grid.filled_count >= config.min_filled_cells;
  return grid;
}

inline OrientedGridShiftResult estimate_oriented_grid_shift(
    const OrientedIntensityGrid& grid_a,
    const OrientedIntensityGrid& grid_b,
    const OrientedIntensityGridConfig& config) {
  OrientedGridShiftResult result;
  if (!grid_a.valid || !grid_b.valid ||
      grid_a.longitudinal_bins != grid_b.longitudinal_bins ||
      grid_a.lateral_bins != grid_b.lateral_bins ||
      grid_a.intensity.size() != grid_b.intensity.size() ||
      config.bin_size_m <= 0.0 ||
      (config.intensity_weight <= 0.0 && config.height_weight <= 0.0)) {
    return result;
  }

  const auto longitudinal_bins =
      static_cast<long>(grid_a.longitudinal_bins);
  const auto lateral_bins = static_cast<long>(grid_a.lateral_bins);
  const long max_longitudinal_shift_bins = std::max<long>(
      0,
      static_cast<long>(std::round(
          config.max_longitudinal_shift_m / config.bin_size_m)));
  const long max_lateral_shift_bins = std::max<long>(
      0,
      static_cast<long>(
          std::round(config.max_lateral_shift_m / config.bin_size_m)));

  auto correlation_at =
      [&](const long longitudinal_shift,
          const long lateral_shift) -> OrientedGridCorrelationScore {
    PearsonCorrelationAccumulator intensity_correlation;
    PearsonCorrelationAccumulator height_correlation;
    for (long longitudinal = 0; longitudinal < longitudinal_bins;
         ++longitudinal) {
      const long shifted_longitudinal =
          longitudinal - longitudinal_shift;
      if (shifted_longitudinal < 0 ||
          shifted_longitudinal >= longitudinal_bins) {
        continue;
      }
      for (long lateral = 0; lateral < lateral_bins; ++lateral) {
        const long shifted_lateral = lateral - lateral_shift;
        if (shifted_lateral < 0 || shifted_lateral >= lateral_bins) {
          continue;
        }
        const std::size_t cell_a = grid_a.index(
            static_cast<std::size_t>(longitudinal),
            static_cast<std::size_t>(lateral));
        const std::size_t cell_b = grid_b.index(
            static_cast<std::size_t>(shifted_longitudinal),
            static_cast<std::size_t>(shifted_lateral));
        if (!grid_a.filled[cell_a] || !grid_b.filled[cell_b]) {
          continue;
        }
        intensity_correlation.add(
            grid_a.intensity[cell_a],
            grid_b.intensity[cell_b]);
        height_correlation.add(
            grid_a.height[cell_a],
            grid_b.height[cell_b]);
      }
    }

    const std::optional<double> intensity_score =
        intensity_correlation.score();
    const std::optional<double> height_score = height_correlation.score();
    double weighted_score = 0.0;
    double active_weight = 0.0;
    if (config.intensity_weight > 0.0 && intensity_score.has_value()) {
      weighted_score += config.intensity_weight * *intensity_score;
      active_weight += config.intensity_weight;
    }
    if (config.height_weight > 0.0 && height_score.has_value()) {
      weighted_score += config.height_weight * *height_score;
      active_weight += config.height_weight;
    }
    OrientedGridCorrelationScore score;
    score.combined =
        active_weight > 0.0 ? weighted_score / active_weight : -2.0;
    score.intensity = intensity_score.value_or(-2.0);
    score.height = height_score.value_or(-2.0);
    score.support = intensity_correlation.support();
    return score;
  };

  std::vector<CorrelationPeakCandidate> candidates;
  const std::size_t longitudinal_candidate_count =
      static_cast<std::size_t>(2 * max_longitudinal_shift_bins + 1);
  const std::size_t lateral_candidate_count =
      static_cast<std::size_t>(2 * max_lateral_shift_bins + 1);
  candidates.reserve(
      longitudinal_candidate_count * lateral_candidate_count);
  long candidate_index = 0;
  for (long longitudinal_shift = -max_longitudinal_shift_bins;
       longitudinal_shift <= max_longitudinal_shift_bins;
       ++longitudinal_shift) {
    for (long lateral_shift = -max_lateral_shift_bins;
         lateral_shift <= max_lateral_shift_bins;
         ++lateral_shift) {
      const OrientedGridCorrelationScore score =
          correlation_at(longitudinal_shift, lateral_shift);
      candidates.push_back({
          candidate_index++,
          score.combined,
          score.support,
          {longitudinal_shift, lateral_shift, 0},
          2,
      });
    }
  }

  CorrelationPeakPolicy policy;
  policy.minimum_score = config.min_correlation;
  policy.minimum_support = config.min_filled_cells;
  policy.secondary_exclusion_radius =
      config.peak_exclusion_radius_bins;
  policy.minimum_peak_margin = config.min_peak_margin;
  const CorrelationPeakSelection selection =
      select_correlation_peak(candidates, policy);

  result.correlation = selection.best.score;
  result.overlap_cells = selection.best.support;
  result.has_competing_peak = selection.second_best.has_value();
  result.second_best_correlation = selection.second_best.has_value()
                                       ? selection.second_best->score
                                       : -1.0;
  result.peak_margin = selection.peak_margin;
  result.ambiguous =
      selection.rejection == CorrelationPeakRejection::ambiguous;
  const long best_longitudinal =
      selection.best.offset_coordinates[0];
  const long best_lateral = selection.best.offset_coordinates[1];
  const OrientedGridCorrelationScore best_channel_scores =
      correlation_at(best_longitudinal, best_lateral);
  result.intensity_correlation = best_channel_scores.intensity;
  result.height_correlation = best_channel_scores.height;
  if (!selection.accepted) {
    return result;
  }

  auto refined_offset =
      [&](const long center,
          const long orthogonal,
          const bool refine_longitudinal) {
    const auto score_at = [&](const long offset) {
      return refine_longitudinal
                 ? correlation_at(offset, orthogonal).combined
                 : correlation_at(orthogonal, offset).combined;
    };
    const double previous = score_at(center - 1);
    const double current = score_at(center);
    const double next = score_at(center + 1);
    if (previous < -1.0 || current < -1.0 || next < -1.0) {
      return static_cast<double>(center);
    }
    const double denominator = previous - 2.0 * current + next;
    if (std::abs(denominator) <= 1.0e-9) {
      return static_cast<double>(center);
    }
    return static_cast<double>(center) +
           std::clamp(
               0.5 * (previous - next) / denominator,
               -0.5,
               0.5);
  };

  const double refined_longitudinal =
      best_longitudinal > -max_longitudinal_shift_bins &&
              best_longitudinal < max_longitudinal_shift_bins
          ? refined_offset(best_longitudinal, best_lateral, true)
          : static_cast<double>(best_longitudinal);
  const double refined_lateral =
      best_lateral > -max_lateral_shift_bins &&
              best_lateral < max_lateral_shift_bins
          ? refined_offset(best_lateral, best_longitudinal, false)
          : static_cast<double>(best_lateral);
  result.longitudinal_shift_m =
      refined_longitudinal * config.bin_size_m;
  result.lateral_shift_m = refined_lateral * config.bin_size_m;
  result.valid = true;
  return result;
}

}  // namespace rko_lio::core
