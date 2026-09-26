#include <gtest/gtest.h>
#include "rko_lio/core/lio.hpp"

using namespace rko_lio::core;
using namespace std::chrono_literals;

static Vector3dVector cloud() {
  Vector3dVector points;
  for (int x = -4; x <= 4; ++x) {
    for (int y = -4; y <= 4; ++y) points.emplace_back(x * .4, y * .4, 2.0);
  }
  return points;
}

TEST(InitializedOutput, BootstrapWaitsForWorldOrientationWithoutLosingStoredPose) {
  LIO::Config config;
  config.initialization_phase = true;
  config.max_num_threads = 1;
  LIO lio(config);
  const auto scan = cloud();
  EXPECT_TRUE(lio.register_scan(scan, TimestampVector(scan.size(), 1s)).empty());
  EXPECT_TRUE(lio.poses_with_timestamps.empty());
  for (int i = 1; i <= 10; ++i) {
    ImuControl imu;
    imu.time = 1s + i * 5ms;
    imu.angular_velocity = Eigen::Vector3d::Zero();
    imu.acceleration = Eigen::Vector3d(0, 0, -GRAVITY_MAG);
    lio.add_imu_measurement(imu);
  }
  EXPECT_FALSE(lio.register_scan(scan, TimestampVector(scan.size(), 1050ms)).empty());
  ASSERT_EQ(lio.poses_with_timestamps.size(), 2u);
  EXPECT_EQ(lio.poses_with_timestamps[0].first, 1s);
  EXPECT_EQ(lio.poses_with_timestamps[1].first, 1050ms);
  const auto& pose = lio.lidar_state.pose;
  EXPECT_LT((pose.so3() * Eigen::Vector3d(0, 0, -1) - Eigen::Vector3d::UnitZ()).norm(), 1e-10);
  EXPECT_LT((pose.inverse() * lio.poses_with_timestamps[0].second).log().norm(), 1e-10);
}

TEST(InitializedOutput, ExplicitNoInitializationStillPublishesFirstScan) {
  LIO::Config config;
  config.initialization_phase = false;
  config.max_num_threads = 1;
  LIO lio(config);
  const auto scan = cloud();
  EXPECT_FALSE(lio.register_scan(scan, TimestampVector(scan.size(), 1s)).empty());
  ASSERT_EQ(lio.poses_with_timestamps.size(), 1u);
  EXPECT_LT(lio.lidar_state.pose.log().norm(), 1e-12);
}
