// Unit tests for taking the keypoints of a small scene again at a finer voxel (refine_keypoints,
// LIO::Config::keypoint_shortfall_voxel_multiplier).
#include <gtest/gtest.h>

#include "rko_lio/core/preprocess_scan.hpp"

#include <Eigen/Core>

namespace {

using rko_lio::core::LIO;
using rko_lio::core::preprocess_scan;
using rko_lio::core::refine_keypoints;
using rko_lio::core::Vector3dVector;

// The walls, floor and ceiling of a 3 m x 2 m x 2.5 m room around the sensor, sampled every 2 cm.
Vector3dVector small_room() {
  Vector3dVector points;
  const double step = 0.02;
  for (double a = -1.5; a <= 1.5; a += step) {
    for (double b = -1.0; b <= 1.5; b += step) {
      points.emplace_back(a, -1.0, b);
      points.emplace_back(a, 1.0, b);
    }
    for (double c = -1.0; c <= 1.0; c += step) {
      points.emplace_back(a, c, -1.0);
      points.emplace_back(a, c, 1.5);
    }
  }
  for (double c = -1.0; c <= 1.0; c += step) {
    for (double b = -1.0; b <= 1.5; b += step) {
      points.emplace_back(-1.5, c, b);
      points.emplace_back(1.5, c, b);
    }
  }
  return points;
}

LIO::Config room_config() {
  LIO::Config config;
  config.voxel_size = 0.25;
  config.min_range = 0.7;
  config.max_range = 100.0;
  config.double_downsample = true;
  config.icp_keypoint_voxel_multiplier = 2.25;
  return config;
}

TEST(RefineKeypoints, FinerVoxelGivesSeveralTimesTheKeypointsOfASmallRoom) {
  const LIO::Config config = room_config();
  const auto result = preprocess_scan(small_room(), config);
  const Vector3dVector finer = refine_keypoints(result, config, 1.0);
  ASSERT_FALSE(result.keypoints.empty());
  EXPECT_GT(finer.size(), 3 * result.keypoints.size());
  EXPECT_LE(finer.size(), result.map_frame.size());
}

TEST(RefineKeypoints, UsesTheClippedFrameWithoutDoubleDownsample) {
  LIO::Config config = room_config();
  config.double_downsample = false;
  const auto result = preprocess_scan(small_room(), config);
  EXPECT_TRUE(result.map_frame.empty());
  const Vector3dVector finer = refine_keypoints(result, config, 0.5);
  EXPECT_GT(finer.size(), result.keypoints.size());
}

TEST(RefineKeypoints, LegacyDownsampleIsHonoured) {
  LIO::Config config = room_config();
  config.legacy_voxel_downsample = true;
  const auto result = preprocess_scan(small_room(), config);
  EXPECT_GT(refine_keypoints(result, config, 1.0).size(), result.keypoints.size());
}

} // namespace
