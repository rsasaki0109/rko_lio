#include <gtest/gtest.h>

#include "rko_lio/core/direct_visual_odometry.hpp"
#include "rko_lio/core/lio.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

rko_lio::core::FisheyeCameraModel camera() {
  rko_lio::core::FisheyeCameraModel model;
  model.width = 80;
  model.height = 60;
  model.fx = 65.0;
  model.fy = 65.0;
  model.cx = 39.5;
  model.cy = 29.5;
  model.distortion = {-0.02, 0.003, 0.0, 0.0};
  return model;
}

rko_lio::core::SparseDepthImage planar_depth(
    const rko_lio::core::FisheyeCameraModel& model, double z) {
  std::vector<Eigen::Vector3d> points;
  points.reserve(static_cast<std::size_t>(model.width * model.height));
  for (int y = 0; y < model.height; ++y) {
    for (int x = 0; x < model.width; ++x) {
      Eigen::Vector3d point;
      if (model.backproject(Eigen::Vector2d(x, y), z, point)) {
        points.push_back(point);
      }
    }
  }
  return rko_lio::core::project_sparse_depth(
      points, Sophus::SE3d(), model);
}

TEST(DirectVisualOdometry, FisheyeProjectionRoundTrips) {
  const auto model = camera();
  for (const Eigen::Vector3d point : {
           Eigen::Vector3d(0.0, 0.0, 3.0),
           Eigen::Vector3d(0.4, -0.2, 2.0),
           Eigen::Vector3d(-0.8, 0.3, 4.0)}) {
    Eigen::Vector2d pixel;
    ASSERT_TRUE(model.project(point, pixel));
    Eigen::Vector3d reconstructed;
    ASSERT_TRUE(model.backproject(pixel, point.z(), reconstructed));
    EXPECT_TRUE(reconstructed.isApprox(point, 1.0e-10));
  }
}

TEST(DirectVisualOdometry, PlumbBobProjectionRoundTrips) {
  auto model = camera();
  model.width = 752;
  model.height = 480;
  model.fx = 425.0258563372763;
  model.fy = 426.7976260903337;
  model.cx = 386.015186655088;
  model.cy = 241.913033674344;
  model.distortion_model =
      rko_lio::core::CameraDistortionModel::plumb_bob;
  model.distortion = {-0.288105327549552, 0.074578284234601,
                      0.0007784489598138802, -0.0002277853975035461};
  for (const Eigen::Vector3d point : {
           Eigen::Vector3d(0.0, 0.0, 3.0),
           Eigen::Vector3d(0.4, -0.2, 2.0),
           Eigen::Vector3d(-0.8, 0.3, 4.0)}) {
    Eigen::Vector2d pixel;
    ASSERT_TRUE(model.project(point, pixel));
    Eigen::Vector3d reconstructed;
    ASSERT_TRUE(model.backproject(pixel, point.z(), reconstructed));
    EXPECT_TRUE(reconstructed.isApprox(point, 1.0e-10));
  }
  Eigen::Vector2d opencv_pixel;
  ASSERT_TRUE(model.project(Eigen::Vector3d(0.4, -0.2, 2.0), opencv_pixel));
  EXPECT_NEAR(opencv_pixel.x(), 469.78586429, 1.0e-8);
  EXPECT_NEAR(opencv_pixel.y(), 199.86727250, 1.0e-8);
}

TEST(DirectVisualOdometry, MetricDirectAlignmentReducesPhotometricError) {
  const auto model = camera();
  rko_lio::core::GrayImage previous{model.width, model.height, {}};
  previous.pixels.resize(static_cast<std::size_t>(model.width * model.height));
  for (int y = 0; y < model.height; ++y) {
    for (int x = 0; x < model.width; ++x) {
      const double value = 120.0 + 45.0 * std::sin(0.21 * x) +
                           38.0 * std::cos(0.17 * y) +
                           20.0 * std::sin(0.11 * (x + y));
      previous.pixels[static_cast<std::size_t>(y * model.width + x)] =
          static_cast<std::uint8_t>(std::clamp(value, 0.0, 255.0));
    }
  }
  const Sophus::SE3d truth(
      Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.002, -0.003)),
      Eigen::Vector3d(0.045, -0.015, 0.0));
  rko_lio::core::GrayImage current{model.width, model.height, {}};
  current.pixels.resize(previous.pixels.size());
  for (int y = 0; y < model.height; ++y) {
    for (int x = 0; x < model.width; ++x) {
      Eigen::Vector3d current_point;
      ASSERT_TRUE(model.backproject(Eigen::Vector2d(x, y), 5.0, current_point));
      Eigen::Vector2d previous_pixel;
      const bool visible = model.project(truth.inverse() * current_point, previous_pixel);
      const double value = visible ? 1.08 * previous.bilinear(
                                           previous_pixel.x(), previous_pixel.y()) + 4.0
                                   : 0.0;
      current.pixels[static_cast<std::size_t>(y * model.width + x)] =
          static_cast<std::uint8_t>(std::clamp(value, 0.0, 255.0));
    }
  }
  const auto previous_depth = planar_depth(model, 5.0);
  auto current_depth = planar_depth(model, 5.0);
  std::fill(current_depth.depth.begin(), current_depth.depth.end(),
            std::numeric_limits<float>::infinity());
  rko_lio::core::DirectVisualOdometryConfig config;
  config.max_features = 120;
  config.grid_cell_size = 8;
  config.min_gradient = 5.0;
  config.min_residuals = 100;
  config.max_rmse = 30.0;
  config.max_pose_step = 0.2;
  const auto result = rko_lio::core::align_direct_visual(
      previous, previous_depth, current, current_depth, model,
      Sophus::SE3d(), config);
  ASSERT_TRUE(result.valid);
  EXPECT_LT(result.final_rmse, 0.45 * result.initial_rmse);
  EXPECT_GT(result.residuals, 100);
  EXPECT_GT(result.inlier_ratio, 0.55);
  EXPECT_NEAR(result.exposure_gain, 1.08, 0.08);
  EXPECT_TRUE(result.pose_information.allFinite());
  EXPECT_GT(result.pose_information.eigenvalues().real().maxCoeff(), 0.0);
}

TEST(DirectVisualOdometry, CameraPredictionUsesCurrentImuInterval) {
  rko_lio::core::LIO::Config config;
  rko_lio::core::LIO lio(config);
  lio.lidar_state.time = rko_lio::core::Secondsd(10.0);
  lio.lidar_state.velocity = Eigen::Vector3d(1.0, 0.0, 0.0);
  lio.interval_stats.imu_count = 2;
  lio.interval_stats.body_acceleration_sum = Eigen::Vector3d(2.0, 0.0, 0.0);
  lio.interval_stats.angular_velocity_sum = Eigen::Vector3d(0.0, 0.0, 0.4);
  Eigen::Vector6d expected_motion = Eigen::Vector6d::Zero();
  expected_motion.head<3>() = Eigen::Vector3d(4.0, 0.0, 0.0);
  expected_motion.tail<3>() = Eigen::Vector3d(0.0, 0.0, 0.4);
  const Sophus::SE3d expected = Sophus::SE3d::exp(expected_motion);
  EXPECT_TRUE(lio.predict_pose_at(rko_lio::core::Secondsd(12.0))
                  .matrix().isApprox(expected.matrix(), 1.0e-12));
}

TEST(DirectVisualOdometry, RejectsPoorInitialFitBeforeIterations) {
  const auto model = camera();
  rko_lio::core::GrayImage previous{model.width, model.height, {}};
  rko_lio::core::GrayImage current{model.width, model.height, {}};
  previous.pixels.resize(static_cast<std::size_t>(model.width * model.height));
  current.pixels.resize(previous.pixels.size());
  for (int y = 0; y < model.height; ++y) {
    for (int x = 0; x < model.width; ++x) {
      const std::size_t index = static_cast<std::size_t>(y * model.width + x);
      previous.pixels[index] = static_cast<std::uint8_t>((7 * x + 11 * y) % 256);
      current.pixels[index] = static_cast<std::uint8_t>(255 - previous.pixels[index]);
    }
  }
  const auto depth = planar_depth(model, 5.0);
  rko_lio::core::DirectVisualOdometryConfig config;
  config.grid_cell_size = 8;
  config.min_gradient = 2.0;
  config.max_initial_rmse = 1.0;
  const auto result = rko_lio::core::align_direct_visual(
      previous, depth, current, depth, model, Sophus::SE3d(), config);
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.failure_reason,
            rko_lio::core::DirectVisualFailureReason::poor_initial_photometric_fit);
  EXPECT_DOUBLE_EQ(result.final_rmse, result.initial_rmse);
}

TEST(DirectVisualOdometry, CountsOnlyConfiguredWeakHessianDirections) {
  Eigen::Matrix6d H = Eigen::Matrix6d::Zero();
  H.diagonal() << 0.01, 0.02, 1.0, 1.0, 1.0, 1000.0;
  rko_lio::core::SelectiveVisualFusionConfig config;
  config.weak_information_ratio = 1.0e-4;
  config.max_weak_directions = 2;
  EXPECT_EQ(rko_lio::core::count_weak_information_directions(H, config), 2U);
  config.weak_information_ratio = 1.0e-6;
  EXPECT_EQ(rko_lio::core::count_weak_information_directions(H, config), 0U);
}

TEST(DirectVisualOdometry, RejectsLidarWeakAxisUnobservedByVision) {
  Eigen::Matrix6d H = Eigen::Matrix6d::Identity();
  H(0, 0) = 1.0e-6;
  const Eigen::Vector6d b = Eigen::Vector6d::Zero();
  Eigen::Vector6d update = Eigen::Vector6d::Zero();
  update(0) = 0.01;
  rko_lio::core::VisualConstraintConfidence confidence;
  confidence.tracks = 200;
  confidence.inliers = 180;
  confidence.rotation_error_deg = 0.1;
  confidence.translation_cosine = 0.99;
  confidence.baseline_m = 0.1;
  confidence.visual_information = Eigen::Matrix6d::Identity();
  confidence.visual_information(0, 0) = 0.0;
  rko_lio::core::SelectiveVisualFusionConfig config;
  config.enabled = true;
  config.weak_information_ratio = 1.0e-4;
  config.min_visual_directional_information_ratio = 0.1;
  auto result = rko_lio::core::fuse_visual_in_weak_directions(
      H, b, update, confidence, config);
  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.visual_unobservable_directions, 1U);
  confidence.visual_information(0, 0) = 1.0;
  result = rko_lio::core::fuse_visual_in_weak_directions(
      H, b, update, confidence, config);
  EXPECT_TRUE(result.accepted);
  EXPECT_EQ(result.fused_directions, 1U);
}

}  // namespace
