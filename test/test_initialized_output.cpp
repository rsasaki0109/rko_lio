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
  EXPECT_FALSE(lio.has_initialized_pose());
  const auto scan = cloud();
  EXPECT_TRUE(lio.register_scan(scan, TimestampVector(scan.size(), 1s)).empty());
  EXPECT_TRUE(lio.poses_with_timestamps.empty());
  EXPECT_GT(lio.imu_state.time, 0ns);
  EXPECT_FALSE(lio.has_initialized_pose());
  for (int i = 1; i <= 10; ++i) {
    ImuControl imu;
    imu.time = 1s + i * 5ms;
    imu.angular_velocity = Eigen::Vector3d::Zero();
    imu.acceleration = Eigen::Vector3d(0, 0, -GRAVITY_MAG);
    lio.add_imu_measurement(imu);
    EXPECT_FALSE(lio.has_initialized_pose());
  }
  EXPECT_FALSE(lio.register_scan(scan, TimestampVector(scan.size(), 1050ms)).empty());
  EXPECT_TRUE(lio.has_initialized_pose());
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
  EXPECT_FALSE(lio.has_initialized_pose());
  const auto scan = cloud();
  EXPECT_FALSE(lio.register_scan(scan, TimestampVector(scan.size(), 1s)).empty());
  EXPECT_TRUE(lio.has_initialized_pose());
  ASSERT_EQ(lio.poses_with_timestamps.size(), 1u);
  EXPECT_LT(lio.lidar_state.pose.log().norm(), 1e-12);
}

TEST(InitializedOutput, InitializationWindowAveragesImuBeforeTheFirstPose) {
  LIO::Config config;
  config.initialization_phase = true;
  config.initialization_window_sec = 0.2;
  config.max_num_threads = 1;
  LIO lio(config);
  const auto scan = cloud();
  EXPECT_TRUE(lio.register_scan(scan, TimestampVector(scan.size(), 1s)).empty());
  // Gravity along -z with alternating +-0.1 rad of roll-axis noise per interval: a
  // single interval would tilt the frame by 0.1 rad, the window average does not.
  for (int interval = 0; interval < 4; ++interval) {
    const double tilt = interval % 2 == 0 ? 0.1 : -0.1;
    for (int i = 1; i <= 10; ++i) {
      ImuControl imu;
      imu.time = 1s + interval * 50ms + i * 5ms;
      imu.angular_velocity = Eigen::Vector3d::Zero();
      imu.acceleration = GRAVITY_MAG * Eigen::Vector3d(0, std::sin(tilt), -std::cos(tilt));
      lio.add_imu_measurement(imu);
    }
    const Nsec stamp = 1s + (interval + 1) * 50ms;
    const bool published = !lio.register_scan(scan, TimestampVector(scan.size(), stamp)).empty();
    EXPECT_EQ(published, interval == 3) << "interval " << interval;
    EXPECT_EQ(lio.has_initialized_pose(), interval == 3) << "interval " << interval;
  }
  const auto& pose = lio.lidar_state.pose;
  EXPECT_LT((pose.so3() * Eigen::Vector3d(0, 0, -1) - Eigen::Vector3d::UnitZ()).norm(), 1e-3);
}
