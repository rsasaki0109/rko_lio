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

// Per-scan registration statistics that use no ground truth: how far the keypoints of the
// registered scan are from their nearest map points, and how strongly the normals of the
// matched map surfaces constrain translation, overall and along the scan's own motion. They are written for
// analysis only (Config::scan_statistics) and never read back into the estimate, so that a
// rule meant to replace a per-dataset setting can first be checked against what the sensor
// data alone shows. Pure functions only, unit-testable in isolation.

#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace rko_lio::core {

struct ScanStatistics {
  double time_sec = 0.0;
  int keypoints = 0;
  /** Keypoints with a map point closer than the correspondence cap at the registered pose. */
  int correspondences = 0;
  /** Quantiles 0.1, 0.25, 0.5, 0.75 and 0.9 of the keypoint-to-nearest-map-point distance.
   *  A keypoint with no map point within the search distance counts as that distance. */
  std::array<double, 5> nn_distance_quantiles{};
  /** Fractions of keypoints with a map point closer than 0.25 m and 0.5 m. */
  double fraction_within_0_25 = 0.0;
  double fraction_within_0_5 = 0.0;
  /** Correspondences whose map voxel has a surface normal. */
  int matched_normals = 0;
  /** Correspondences on horizontal map surfaces (normal within about 25 deg of the vertical) more
   *  than 0.5 m above the sensor (a ceiling) and more than 0.5 m below it (a floor, the ground). */
  int ceiling_correspondences = 0;
  int floor_correspondences = 0;
  /** Median distance of the keypoints from the sensor (m). */
  double median_keypoint_range = std::numeric_limits<double>::quiet_NaN();
  /** Eigenvalues of the mean n n^T over the matched surface normals, ascending. */
  std::array<double, 3> normal_eigenvalues{};
  /** axis^T N axis / max eigenvalue of N for the unit motion axis of the scan; NaN when the scan
   *  moved too little to define one. */
  double motion_axis_information_ratio = std::numeric_limits<double>::quiet_NaN();
};

inline constexpr std::array<double, 5> kScanStatisticsQuantiles{0.1, 0.25, 0.5, 0.75, 0.9};

/** Linear-interpolated quantile of an ascending, non-empty vector. */
inline double sorted_quantile(const std::vector<double>& sorted, const double q) {
  if (sorted.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double position = std::clamp(q, 0.0, 1.0) * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

/** Summarise the nearest-neighbour distances of one scan's keypoints (each already capped at
 *  the search distance) and the normal information N (mean n n^T of the matched map surface
 *  normals). `distances` is sorted in place. */
inline ScanStatistics summarize_scan(const double time_sec,
                                     std::vector<double>& distances,
                                     const double correspondence_cap,
                                     const Eigen::Matrix3d& normal_information,
                                     const std::optional<Eigen::Vector3d>& motion_axis) {
  ScanStatistics stats;
  stats.time_sec = time_sec;
  stats.keypoints = static_cast<int>(distances.size());
  std::sort(distances.begin(), distances.end());
  if (!distances.empty()) {
    const auto count_below = [&distances](const double limit) {
      return static_cast<int>(std::lower_bound(distances.begin(), distances.end(), limit) - distances.begin());
    };
    stats.correspondences = count_below(correspondence_cap);
    const double n = static_cast<double>(distances.size());
    stats.fraction_within_0_25 = static_cast<double>(count_below(0.25)) / n;
    stats.fraction_within_0_5 = static_cast<double>(count_below(0.5)) / n;
    for (std::size_t i = 0; i < kScanStatisticsQuantiles.size(); ++i) {
      stats.nn_distance_quantiles[i] = sorted_quantile(distances, kScanStatisticsQuantiles[i]);
    }
  } else {
    stats.nn_distance_quantiles.fill(std::numeric_limits<double>::quiet_NaN());
  }
  const Eigen::Matrix3d symmetric = 0.5 * (normal_information + normal_information.transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(symmetric, Eigen::EigenvaluesOnly);
  const Eigen::Vector3d eigenvalues = solver.eigenvalues();
  for (int i = 0; i < 3; ++i) {
    stats.normal_eigenvalues[static_cast<std::size_t>(i)] = eigenvalues(i);
  }
  const double max_eigenvalue = eigenvalues(2);
  if (motion_axis.has_value() && max_eigenvalue > 0.0) {
    stats.motion_axis_information_ratio = motion_axis->dot(symmetric * *motion_axis) / max_eigenvalue;
  }
  return stats;
}

} // namespace rko_lio::core
