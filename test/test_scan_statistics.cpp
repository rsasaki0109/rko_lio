// Unit tests for the GT-free per-scan registration statistics (scan_statistics.hpp).
#include <gtest/gtest.h>

#include "rko_lio/core/scan_statistics.hpp"

#include <Eigen/Core>

#include <cmath>
#include <optional>
#include <vector>

namespace {

using rko_lio::core::ScanStatistics;
using rko_lio::core::sorted_quantile;
using rko_lio::core::summarize_scan;

TEST(ScanStatistics, QuantilesInterpolateBetweenSortedValues) {
  const std::vector<double> sorted{0.0, 1.0, 2.0, 3.0, 4.0};
  EXPECT_DOUBLE_EQ(sorted_quantile(sorted, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(sorted_quantile(sorted, 0.5), 2.0);
  EXPECT_DOUBLE_EQ(sorted_quantile(sorted, 0.1), 0.4);
  EXPECT_DOUBLE_EQ(sorted_quantile(sorted, 1.0), 4.0);
  EXPECT_TRUE(std::isnan(sorted_quantile({}, 0.5)));
}

TEST(ScanStatistics, CountsDistancesBelowTheCapAndFixedLimits) {
  std::vector<double> distances{0.6, 0.1, 0.3, 0.2, 0.75};
  const ScanStatistics stats =
      summarize_scan(12.5, distances, 0.25, Eigen::Matrix3d::Identity(), std::nullopt);
  EXPECT_DOUBLE_EQ(stats.time_sec, 12.5);
  EXPECT_EQ(stats.keypoints, 5);
  EXPECT_EQ(stats.correspondences, 2);
  EXPECT_DOUBLE_EQ(stats.fraction_within_0_25, 0.4);
  EXPECT_DOUBLE_EQ(stats.fraction_within_0_5, 0.6);
  EXPECT_DOUBLE_EQ(stats.nn_distance_quantiles[2], 0.3);
  EXPECT_TRUE(std::isnan(stats.motion_axis_information_ratio));
}

TEST(ScanStatistics, MotionAxisRatioIsTheShareOfTheStrongestDirection) {
  // A corridor along x: little information along x, plenty across it.
  Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
  information.diagonal() << 0.01, 1.0, 2.0;
  std::vector<double> distances{0.1};
  const ScanStatistics along =
      summarize_scan(0.0, distances, 0.5, information, Eigen::Vector3d(1.0, 0.0, 0.0));
  EXPECT_DOUBLE_EQ(along.normal_eigenvalues[0], 0.01);
  EXPECT_DOUBLE_EQ(along.normal_eigenvalues[2], 2.0);
  EXPECT_DOUBLE_EQ(along.motion_axis_information_ratio, 0.005);
  const ScanStatistics across =
      summarize_scan(0.0, distances, 0.5, information, Eigen::Vector3d(0.0, 0.0, 1.0));
  EXPECT_DOUBLE_EQ(across.motion_axis_information_ratio, 1.0);
}

TEST(ScanStatistics, EmptyScanHasNoQuantiles) {
  std::vector<double> distances;
  const ScanStatistics stats =
      summarize_scan(0.0, distances, 0.5, Eigen::Matrix3d::Identity(), std::nullopt);
  EXPECT_EQ(stats.keypoints, 0);
  EXPECT_EQ(stats.correspondences, 0);
  EXPECT_TRUE(std::isnan(stats.nn_distance_quantiles[0]));
}

} // namespace
