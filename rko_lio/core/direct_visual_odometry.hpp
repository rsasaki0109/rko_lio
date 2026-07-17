#pragma once

#include <sophus/se3.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace rko_lio::core {

enum class CameraDistortionModel {
  equidistant,
  plumb_bob,
};

struct FisheyeCameraModel {
  int width = 0;
  int height = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  CameraDistortionModel distortion_model = CameraDistortionModel::equidistant;
  std::array<double, 4> distortion{};

  bool valid() const;
  bool project(const Eigen::Vector3d& point, Eigen::Vector2d& pixel) const;
  bool backproject(const Eigen::Vector2d& pixel, double depth,
                   Eigen::Vector3d& point) const;
};

struct GrayImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels;

  bool valid() const;
  double bilinear(double x, double y) const;
  double gradient_squared(int x, int y) const;
};

struct SparseDepthImage {
  int width = 0;
  int height = 0;
  std::vector<float> depth;
  // Preserve the winning z-buffer sample instead of reconstructing it from
  // the rounded depth pixel.  The latter introduces an avoidable angular
  // error at every sparse LiDAR feature.
  std::vector<Eigen::Vector3d> camera_points;
  std::vector<Eigen::Vector2d> projected_pixels;

  bool valid() const;
  float nearest_depth(int x, int y, int radius) const;
};

SparseDepthImage project_sparse_depth(
    const std::vector<Eigen::Vector3d>& lidar_points,
    const Sophus::SE3d& camera_T_lidar,
    const FisheyeCameraModel& camera);

struct DirectVisualOdometryConfig {
  int max_features = 240;
  int grid_cell_size = 16;
  int patch_radius = 1;
  int max_iterations = 8;
  int occlusion_search_radius = 2;
  double min_gradient = 12.0;
  double huber_delta = 12.0;
  double occlusion_tolerance_m = 0.35;
  double finite_difference_epsilon = 1.0e-5;
  int min_residuals = 180;
  double min_inlier_ratio = 0.55;
  double max_rmse = 20.0;
  double max_initial_rmse = 70.0;
  double max_pose_step = 0.10;
  double max_translation_correction_m = 0.08;
  double max_rotation_correction_rad = 0.05;
};

enum class DirectVisualFailureReason {
  none = 0,
  no_features,
  insufficient_residuals,
  nonfinite_update,
  oversized_update,
  poor_initial_photometric_fit,
  low_inlier_ratio,
  high_rmse,
  no_photometric_improvement,
};

struct DirectVisualOdometryResult {
  bool valid = false;
  DirectVisualFailureReason failure_reason = DirectVisualFailureReason::no_features;
  Sophus::SE3d current_T_previous_camera;
  int features = 0;
  int tracked_features = 0;
  int residuals = 0;
  int inliers = 0;
  double inlier_ratio = 0.0;
  double initial_rmse = std::numeric_limits<double>::infinity();
  double final_rmse = std::numeric_limits<double>::infinity();
  double exposure_gain = 1.0;
  double exposure_bias = 0.0;
  Eigen::Matrix<double, 6, 6> pose_information =
      Eigen::Matrix<double, 6, 6>::Zero();
};

DirectVisualOdometryResult align_direct_visual(
    const GrayImage& previous_image,
    const SparseDepthImage& previous_depth,
    const GrayImage& current_image,
    const SparseDepthImage& current_depth,
    const FisheyeCameraModel& camera,
    const Sophus::SE3d& initial_current_T_previous_camera,
    const DirectVisualOdometryConfig& config);

}  // namespace rko_lio::core
