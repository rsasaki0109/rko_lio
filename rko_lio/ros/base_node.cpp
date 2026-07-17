/*
 * MIT License
 *
 * Copyright (c) 2025 Meher V.R. Malladi.
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

#include "base_node.hpp"
#include "rko_lio/core/process_timestamps.hpp"
#include "rko_lio/ros/utils/utils.hpp"
// other
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <stdexcept>

namespace {
using namespace std::literals;

// ============================================================================
// [v0.8 Phase 1, diagnostic-only] anisotropic nav_msgs/Odometry.pose.covariance
// fill, derived from LIO::State::icp_diagnostics (rko_lio/core/util.hpp).
//
// This does not change odom_msg.pose.pose or odom_msg.twist.twist in any way
// -- those are still set exactly as before (see BaseNode::publish_odometry
// below). Only the covariance field, which upstream leaves at its
// message-default (all zeros), is filled here.
//
// Scale rationale: at a Gauss-Newton optimum, the accumulated Hessian `H`
// (already averaged over correspondences by build_icp_linear_system, see
// lio.cpp) is the standard asymptotic information matrix of the ICP solve;
// its inverse is the corresponding pose-covariance approximation -- the same
// H <-> information-matrix relationship Zhang & Singh ("On Degeneracy of
// Optimization-based State Estimation Problems", ICRA 2016) use to define
// per-direction degeneracy. `H`'s eigenvectors/eigenvalues are reused
// directly from LocalizabilitySummary (already computed once in core, not
// recomputed here) to build Cov = V * diag(1/lambda) * V^T.
//
// Near-exactly-degenerate directions (eigenvalue ~ 0) would blow up to +inf
// under a literal inverse; each eigenvalue is floored at
// max(kMinEigenvalueFloor, kRelativeEigenvalueFloor * lambda_max) before
// inverting, so the reported covariance stays finite while still reporting a
// very large (i.e. "not informative") uncertainty along that direction,
// rather than a numerically meaningless inf/nan on the wire.
constexpr double kMinEigenvalueFloor = 1e-9;
constexpr double kRelativeEigenvalueFloor = 1e-6;
// Fallback diagonal covariance (m^2 on the translation block, rad^2 on the
// rotation block) used when no ICP solve happened for this scan (first
// frame, a dropped scan, or a kidnap-recovery/local-reset scan): "no
// information available", deliberately not "perfectly known" (which a
// covariance of all zeros would imply to a consumer such as robot_localization).
constexpr double kNoDiagnosticsFallbackVariance = 1e6;

Sophus::SE3d desired_base_pose_from_monocular_relative(const Sophus::SE3d& world_T_previous_base,
                                                       const Sophus::SE3d& world_T_predicted_current_base,
                                                       const Sophus::SE3d& base_T_camera,
                                                       const Eigen::Matrix3d& current_R_previous_camera,
                                                       const Eigen::Vector3d& current_t_previous_direction) {
  const Sophus::SE3d world_T_previous_camera = world_T_previous_base * base_T_camera;
  const Sophus::SE3d world_T_predicted_current_camera = world_T_predicted_current_base * base_T_camera;
  const Sophus::SE3d predicted_current_T_previous_camera =
      world_T_predicted_current_camera.inverse() * world_T_previous_camera;
  const double baseline = predicted_current_T_previous_camera.translation().norm();
  const Sophus::SE3d measured_current_T_previous_camera(Sophus::SO3d(current_R_previous_camera),
                                                        baseline * current_t_previous_direction.normalized());
  return world_T_previous_camera * measured_current_T_previous_camera.inverse() * base_T_camera.inverse();
}

std::array<double, 36> pose_covariance_from_state(const rko_lio::core::State& state) {
  std::array<double, 36> covariance{}; // zero-initialized
  if (!state.icp_diagnostics.has_value()) {
    for (int i = 0; i < 6; ++i) {
      covariance[static_cast<size_t>(i * 6 + i)] = kNoDiagnosticsFallbackVariance;
    }
    return covariance;
  }
  const rko_lio::core::LocalizabilitySummary& localizability = state.icp_diagnostics->localizability;
  const double lambda_max = localizability.eigenvalues.maxCoeff();
  const double eigenvalue_floor = std::max(kMinEigenvalueFloor, kRelativeEigenvalueFloor * lambda_max);
  Eigen::Vector6d inv_eigenvalues;
  for (int i = 0; i < 6; ++i) {
    inv_eigenvalues(i) = 1.0 / std::max(localizability.eigenvalues(i), eigenvalue_floor);
  }
  // Sophus::SE3d's se(3) tangent order (translation rho, then rotation phi) matches
  // geometry_msgs/PoseWithCovariance's documented covariance order ([x, y, z, rot x, rot y,
  // rot z]) exactly, so no axis permutation is needed between H's ordering and the wire
  // covariance.
  const Eigen::Matrix6d cov =
      localizability.eigenvectors * inv_eigenvalues.asDiagonal() * localizability.eigenvectors.transpose();
  for (int row = 0; row < 6; ++row) {
    for (int col = 0; col < 6; ++col) {
      covariance[static_cast<size_t>(row * 6 + col)] = cov(row, col);
    }
  }
  return covariance;
}
} // namespace

namespace rko_lio::core {
// necessary for serializing the config, including the namespacing
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SelectiveVisualFusionConfig,
                                   enabled,
                                   min_tracks,
                                   min_inliers,
                                   min_inlier_ratio,
                                   max_rotation_error_deg,
                                   min_translation_cosine,
                                   min_baseline_m,
                                   max_baseline_m,
                                   weak_information_ratio,
                                   relative_information_weight,
                                   min_visual_directional_information_ratio,
                                   max_weak_directions,
                                   max_translation_update_m,
                                   max_rotation_update_rad)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LIO::Config,
                                   deskew,
                                   max_iterations,
                                   voxel_size,
                                   max_points_per_voxel,
                                   max_range,
                                   min_range,
                                   convergence_criterion,
                                   max_correspondence_distance,
                                   max_num_threads,
                                   initialization_phase,
                                   max_expected_jerk,
                                   double_downsample,
                                   icp_keypoint_voxel_multiplier,
                                   min_beta,
                                   degeneracy_aware_solve,
                                   degeneracy_well_conditioned_ratio,
                                   degeneracy_multiplicity_relative_gap,
                                   degeneracy_prior_weight,
                                   degeneracy_persistence_gate,
                                   degeneracy_persistence_min_scans,
                                   degeneracy_persistence_tracking_ratio,
                                   degeneracy_persistence_min_absolute_cosine,
                                   degeneracy_persistence_min_translation_fraction,
                                   degeneracy_adaptive_iteration_budget,
                                   degeneracy_adaptive_max_iterations,
                                   degeneracy_adaptive_iteration_ratio,
                                   degeneracy_adaptive_hold_scans,
                                   degeneracy_multiscan_observability_gate,
                                   degeneracy_observability_window_scans,
                                   degeneracy_observability_min_scans,
                                   degeneracy_observability_max_directional_ratio,
                                   visual_fusion,
                                   visual_prior_max_time_offset_sec,
                                   radar_velocity_fusion,
                                   radar_prior_max_time_offset_sec,
                                   radar_min_speed,
                                   radar_disagreement_gate,
                                   radar_disagreement_min_mps,
                                   radar_disagreement_min_scans,
                                   radar_disagreement_weight,
                                   intensity_constraint,
                                   intensity_bin_size_m,
                                   intensity_profile_half_length_m,
                                   intensity_max_shift_m,
                                   intensity_min_correlation,
                                   intensity_min_filled_bins,
                                   intensity_disagreement_gate,
                                   intensity_disagreement_min_mps,
                                   intensity_disagreement_min_scans,
                                   intensity_disagreement_weight,
                                   max_scan_delta_sec,
                                   enable_kidnap_relocalization,
                                   reset_on_registration_failure,
                                   recovery_min_failures,
                                   relocalize_after_scan_gap,
                                   relocalization_min_correspondences,
                                   relocalization_min_inlier_ratio,
                                   relocalization_max_mean_error,
                                   relocalization_max_correspondence_distance,
                                   relocalization_yaw_samples,
                                   relocalization_pose_stride,
                                   relocalization_min_pose_separation,
                                   relocalization_max_iterations)
} // namespace rko_lio::core

namespace rko_lio::ros {

core::ImuControl imu_msg_to_imu_data(const sensor_msgs::msg::Imu& imu_msg) {
  core::ImuControl imu_data;
  imu_data.time = utils::to_ns(imu_msg.header.stamp);
  imu_data.angular_velocity = utils::ros_xyz_to_eigen_vector3d(imu_msg.angular_velocity);
  imu_data.acceleration = utils::ros_xyz_to_eigen_vector3d(imu_msg.linear_acceleration);
  return imu_data;
}

BaseNode::BaseNode(const std::string& node_name, const rclcpp::NodeOptions& options) {
  node = rclcpp::Node::make_shared(node_name, options);
  imu_topic = node->declare_parameter<std::string>("imu_topic");     // required
  lidar_topic = node->declare_parameter<std::string>("lidar_topic"); // required
  base_frame = node->declare_parameter<std::string>("base_frame");   // required
  imu_frame = node->declare_parameter<std::string>("imu_frame", imu_frame);
  lidar_frame = node->declare_parameter<std::string>("lidar_frame", lidar_frame);
  odom_frame = node->declare_parameter<std::string>("odom_frame", odom_frame);
  odom_topic = node->declare_parameter<std::string>("odom_topic", odom_topic);

  // tf
  invert_odom_tf = node->declare_parameter<bool>("invert_odom_tf", invert_odom_tf);
  tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
  tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*node);

  // publishing
  const rclcpp::QoS publisher_qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
  odom_publisher = node->create_publisher<nav_msgs::msg::Odometry>(odom_topic, publisher_qos);

  publish_lidar_acceleration = node->declare_parameter<bool>("publish_lidar_acceleration", publish_lidar_acceleration);
  if (publish_lidar_acceleration) {
    lidar_accel_publisher =
        node->create_publisher<geometry_msgs::msg::AccelStamped>("rko_lio/lidar_acceleration", publisher_qos);
  }

  publish_deskewed_scan = node->declare_parameter<bool>("publish_deskewed_scan", publish_deskewed_scan);
  if (publish_deskewed_scan) {
    deskewed_scan_topic = node->declare_parameter<std::string>("deskewed_scan_topic", deskewed_scan_topic);
    frame_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(deskewed_scan_topic, publisher_qos);
  }

  publish_local_map = node->declare_parameter<bool>("publish_local_map", publish_local_map);
  if (publish_local_map) {
    map_topic = node->declare_parameter<std::string>("map_topic", map_topic);
    const double publish_map_after_seconds =
        node->declare_parameter<double>("publish_map_after", core::to_seconds(publish_map_after));
    publish_map_after = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(publish_map_after_seconds));
    map_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(map_topic, publisher_qos);
    map_publish_thead = std::jthread([this]() { publish_map_loop(); });
  }

  // lio params
  core::LIO::Config lio_config{};
  lio_config.deskew = node->declare_parameter<bool>("deskew", lio_config.deskew);
  lio_config.max_iterations =
      static_cast<size_t>(node->declare_parameter<int>("max_iterations", static_cast<int>(lio_config.max_iterations)));
  lio_config.voxel_size = node->declare_parameter<double>("voxel_size", lio_config.voxel_size);
  lio_config.max_points_per_voxel =
      static_cast<int>(node->declare_parameter<int>("max_points_per_voxel", lio_config.max_points_per_voxel));
  lio_config.max_range = node->declare_parameter<double>("max_range", lio_config.max_range);
  lio_config.min_range = node->declare_parameter<double>("min_range", lio_config.min_range);
  lio_config.convergence_criterion =
      node->declare_parameter<double>("convergence_criterion", lio_config.convergence_criterion);
  lio_config.max_correspondence_distance =
      node->declare_parameter<double>("max_correspondence_distance", lio_config.max_correspondence_distance);
  lio_config.max_num_threads =
      static_cast<int>(node->declare_parameter<int>("max_num_threads", lio_config.max_num_threads));
  lio_config.initialization_phase =
      node->declare_parameter<bool>("initialization_phase", lio_config.initialization_phase);
  lio_config.max_expected_jerk = node->declare_parameter<double>("max_expected_jerk", lio_config.max_expected_jerk);
  lio_config.double_downsample = node->declare_parameter<bool>("double_downsample", lio_config.double_downsample);
  lio_config.min_beta = node->declare_parameter<double>("min_beta", lio_config.min_beta);
  lio_config.icp_keypoint_voxel_multiplier =
      node->declare_parameter<double>("icp_keypoint_voxel_multiplier", lio_config.icp_keypoint_voxel_multiplier);

  // ---- degeneracy-aware ICP solve (fork addition; default-off) ----
  lio_config.degeneracy_aware_solve =
      node->declare_parameter<bool>("degeneracy_aware_solve", lio_config.degeneracy_aware_solve);
  lio_config.degeneracy_well_conditioned_ratio = node->declare_parameter<double>(
      "degeneracy_well_conditioned_ratio", lio_config.degeneracy_well_conditioned_ratio);
  lio_config.degeneracy_multiplicity_relative_gap = node->declare_parameter<double>(
      "degeneracy_multiplicity_relative_gap", lio_config.degeneracy_multiplicity_relative_gap);
  lio_config.degeneracy_prior_weight =
      node->declare_parameter<double>("degeneracy_prior_weight", lio_config.degeneracy_prior_weight);
  lio_config.degeneracy_persistence_gate =
      node->declare_parameter<bool>("degeneracy_persistence_gate", lio_config.degeneracy_persistence_gate);
  const auto degeneracy_persistence_min_scans = node->declare_parameter<int>(
      "degeneracy_persistence_min_scans", static_cast<int>(lio_config.degeneracy_persistence_min_scans));
  lio_config.degeneracy_persistence_min_scans =
      static_cast<size_t>(degeneracy_persistence_min_scans > 0 ? degeneracy_persistence_min_scans : 1);
  lio_config.degeneracy_persistence_tracking_ratio = node->declare_parameter<double>(
      "degeneracy_persistence_tracking_ratio", lio_config.degeneracy_persistence_tracking_ratio);
  lio_config.degeneracy_persistence_min_absolute_cosine = node->declare_parameter<double>(
      "degeneracy_persistence_min_absolute_cosine", lio_config.degeneracy_persistence_min_absolute_cosine);
  lio_config.degeneracy_persistence_min_translation_fraction = node->declare_parameter<double>(
      "degeneracy_persistence_min_translation_fraction", lio_config.degeneracy_persistence_min_translation_fraction);
  lio_config.degeneracy_adaptive_iteration_budget = node->declare_parameter<bool>(
      "degeneracy_adaptive_iteration_budget", lio_config.degeneracy_adaptive_iteration_budget);
  const auto degeneracy_adaptive_max_iterations = node->declare_parameter<int>(
      "degeneracy_adaptive_max_iterations", static_cast<int>(lio_config.degeneracy_adaptive_max_iterations));
  lio_config.degeneracy_adaptive_max_iterations =
      static_cast<size_t>(degeneracy_adaptive_max_iterations > 0 ? degeneracy_adaptive_max_iterations : 1);
  lio_config.degeneracy_adaptive_iteration_ratio = node->declare_parameter<double>(
      "degeneracy_adaptive_iteration_ratio", lio_config.degeneracy_adaptive_iteration_ratio);
  const auto degeneracy_adaptive_hold_scans = node->declare_parameter<int>(
      "degeneracy_adaptive_hold_scans", static_cast<int>(lio_config.degeneracy_adaptive_hold_scans));
  lio_config.degeneracy_adaptive_hold_scans =
      static_cast<size_t>(degeneracy_adaptive_hold_scans > 0 ? degeneracy_adaptive_hold_scans : 1);
  lio_config.degeneracy_multiscan_observability_gate = node->declare_parameter<bool>(
      "degeneracy_multiscan_observability_gate", lio_config.degeneracy_multiscan_observability_gate);
  const auto degeneracy_observability_window_scans = node->declare_parameter<int>(
      "degeneracy_observability_window_scans", static_cast<int>(lio_config.degeneracy_observability_window_scans));
  lio_config.degeneracy_observability_window_scans =
      static_cast<size_t>(degeneracy_observability_window_scans > 0 ? degeneracy_observability_window_scans : 1);
  const auto degeneracy_observability_min_scans = node->declare_parameter<int>(
      "degeneracy_observability_min_scans", static_cast<int>(lio_config.degeneracy_observability_min_scans));
  lio_config.degeneracy_observability_min_scans =
      static_cast<size_t>(degeneracy_observability_min_scans > 0 ? degeneracy_observability_min_scans : 1);
  lio_config.degeneracy_observability_max_directional_ratio = node->declare_parameter<double>(
      "degeneracy_observability_max_directional_ratio", lio_config.degeneracy_observability_max_directional_ratio);

  // ---- selective visual fusion (fork addition; default-off) ----
  lio_config.visual_fusion.enabled =
      node->declare_parameter<bool>("selective_visual_fusion", lio_config.visual_fusion.enabled);
  const auto visual_min_tracks =
      node->declare_parameter<int>("visual_min_tracks", static_cast<int>(lio_config.visual_fusion.min_tracks));
  lio_config.visual_fusion.min_tracks = static_cast<std::size_t>(visual_min_tracks > 0 ? visual_min_tracks : 1);
  const auto visual_min_inliers =
      node->declare_parameter<int>("visual_min_inliers", static_cast<int>(lio_config.visual_fusion.min_inliers));
  lio_config.visual_fusion.min_inliers = static_cast<std::size_t>(visual_min_inliers > 0 ? visual_min_inliers : 1);
  lio_config.visual_fusion.min_inlier_ratio =
      node->declare_parameter<double>("visual_min_inlier_ratio", lio_config.visual_fusion.min_inlier_ratio);
  lio_config.visual_fusion.max_rotation_error_deg = node->declare_parameter<double>(
      "visual_max_rotation_error_deg", lio_config.visual_fusion.max_rotation_error_deg);
  lio_config.visual_fusion.min_translation_cosine = node->declare_parameter<double>(
      "visual_min_translation_cosine", lio_config.visual_fusion.min_translation_cosine);
  lio_config.visual_fusion.min_baseline_m =
      node->declare_parameter<double>("visual_min_baseline_m", lio_config.visual_fusion.min_baseline_m);
  lio_config.visual_fusion.max_baseline_m =
      node->declare_parameter<double>("visual_max_baseline_m", lio_config.visual_fusion.max_baseline_m);
  lio_config.visual_fusion.weak_information_ratio = node->declare_parameter<double>(
      "visual_weak_information_ratio", lio_config.visual_fusion.weak_information_ratio);
  lio_config.visual_fusion.relative_information_weight = node->declare_parameter<double>(
      "visual_relative_information_weight", lio_config.visual_fusion.relative_information_weight);
  lio_config.visual_fusion.min_visual_directional_information_ratio =
      node->declare_parameter<double>("visual_min_directional_information_ratio",
                                      lio_config.visual_fusion.min_visual_directional_information_ratio);
  const auto visual_max_weak_directions = node->declare_parameter<int>(
      "visual_max_weak_directions", static_cast<int>(lio_config.visual_fusion.max_weak_directions));
  lio_config.visual_fusion.max_weak_directions =
      static_cast<std::size_t>(visual_max_weak_directions > 0 ? visual_max_weak_directions : 0);
  lio_config.visual_fusion.max_translation_update_m = node->declare_parameter<double>(
      "visual_max_translation_update_m", lio_config.visual_fusion.max_translation_update_m);
  lio_config.visual_fusion.max_rotation_update_rad = node->declare_parameter<double>(
      "visual_max_rotation_update_rad", lio_config.visual_fusion.max_rotation_update_rad);
  lio_config.visual_prior_max_time_offset_sec = node->declare_parameter<double>(
      "visual_prior_max_time_offset_sec", lio_config.visual_prior_max_time_offset_sec);

  // ---- radar ego-velocity fusion (fork addition; default-off) ----
  lio_config.radar_velocity_fusion =
      node->declare_parameter<bool>("radar_velocity_fusion", lio_config.radar_velocity_fusion);
  lio_config.radar_prior_max_time_offset_sec =
      node->declare_parameter<double>("radar_prior_max_time_offset_sec", lio_config.radar_prior_max_time_offset_sec);
  lio_config.radar_min_speed = node->declare_parameter<double>("radar_min_speed", lio_config.radar_min_speed);
  lio_config.radar_disagreement_gate =
      node->declare_parameter<bool>("radar_disagreement_gate", lio_config.radar_disagreement_gate);
  lio_config.radar_disagreement_min_mps =
      node->declare_parameter<double>("radar_disagreement_min_mps", lio_config.radar_disagreement_min_mps);
  const auto radar_disagreement_min_scans = node->declare_parameter<int>(
      "radar_disagreement_min_scans", static_cast<int>(lio_config.radar_disagreement_min_scans));
  lio_config.radar_disagreement_min_scans =
      static_cast<size_t>(radar_disagreement_min_scans > 0 ? radar_disagreement_min_scans : 1);
  lio_config.radar_disagreement_weight =
      node->declare_parameter<double>("radar_disagreement_weight", lio_config.radar_disagreement_weight);

  // ---- intensity/reflectivity texture constraint (fork addition; default-off) ----
  lio_config.intensity_constraint =
      node->declare_parameter<bool>("intensity_constraint", lio_config.intensity_constraint);
  lio_config.intensity_bin_size_m =
      node->declare_parameter<double>("intensity_bin_size_m", lio_config.intensity_bin_size_m);
  lio_config.intensity_profile_half_length_m = node->declare_parameter<double>(
      "intensity_profile_half_length_m", lio_config.intensity_profile_half_length_m);
  lio_config.intensity_max_shift_m =
      node->declare_parameter<double>("intensity_max_shift_m", lio_config.intensity_max_shift_m);
  lio_config.intensity_min_correlation =
      node->declare_parameter<double>("intensity_min_correlation", lio_config.intensity_min_correlation);
  const auto intensity_min_filled_bins = node->declare_parameter<int>(
      "intensity_min_filled_bins", static_cast<int>(lio_config.intensity_min_filled_bins));
  lio_config.intensity_min_filled_bins =
      static_cast<size_t>(intensity_min_filled_bins > 0 ? intensity_min_filled_bins : 1);
  lio_config.intensity_disagreement_gate =
      node->declare_parameter<bool>("intensity_disagreement_gate", lio_config.intensity_disagreement_gate);
  lio_config.intensity_disagreement_min_mps =
      node->declare_parameter<double>("intensity_disagreement_min_mps", lio_config.intensity_disagreement_min_mps);
  const auto intensity_disagreement_min_scans = node->declare_parameter<int>(
      "intensity_disagreement_min_scans", static_cast<int>(lio_config.intensity_disagreement_min_scans));
  lio_config.intensity_disagreement_min_scans =
      static_cast<size_t>(intensity_disagreement_min_scans > 0 ? intensity_disagreement_min_scans : 1);
  lio_config.intensity_disagreement_weight = node->declare_parameter<double>(
      "intensity_disagreement_weight", lio_config.intensity_disagreement_weight);

  // ---- scan-gap re-anchor + kidnap relocalization (fork addition; default preserves legacy
  // behavior for max_scan_delta_sec, kidnap features themselves default-off) ----
  lio_config.max_scan_delta_sec =
      node->declare_parameter<double>("max_scan_delta_sec", lio_config.max_scan_delta_sec);
  lio_config.enable_kidnap_relocalization =
      node->declare_parameter<bool>("enable_kidnap_relocalization", lio_config.enable_kidnap_relocalization);
  lio_config.reset_on_registration_failure =
      node->declare_parameter<bool>("reset_on_registration_failure", lio_config.reset_on_registration_failure);
  lio_config.recovery_min_failures =
      node->declare_parameter<int>("recovery_min_failures", lio_config.recovery_min_failures);
  lio_config.relocalize_after_scan_gap =
      node->declare_parameter<bool>("relocalize_after_scan_gap", lio_config.relocalize_after_scan_gap);
  lio_config.relocalization_min_correspondences = node->declare_parameter<int>(
      "relocalization_min_correspondences", lio_config.relocalization_min_correspondences);
  lio_config.relocalization_min_inlier_ratio =
      node->declare_parameter<double>("relocalization_min_inlier_ratio", lio_config.relocalization_min_inlier_ratio);
  lio_config.relocalization_max_mean_error =
      node->declare_parameter<double>("relocalization_max_mean_error", lio_config.relocalization_max_mean_error);
  lio_config.relocalization_max_correspondence_distance = node->declare_parameter<double>(
      "relocalization_max_correspondence_distance", lio_config.relocalization_max_correspondence_distance);
  lio_config.relocalization_yaw_samples =
      node->declare_parameter<int>("relocalization_yaw_samples", lio_config.relocalization_yaw_samples);
  lio_config.relocalization_pose_stride =
      node->declare_parameter<int>("relocalization_pose_stride", lio_config.relocalization_pose_stride);
  lio_config.relocalization_min_pose_separation = node->declare_parameter<int>(
      "relocalization_min_pose_separation", lio_config.relocalization_min_pose_separation);
  lio_config.relocalization_max_iterations =
      node->declare_parameter<int>("relocalization_max_iterations", lio_config.relocalization_max_iterations);

  lio = std::make_unique<core::LIO>(lio_config);

  // ---- direct visual odometry frontend params (fork addition; default-off) ----
  direct_visual_frontend = node->declare_parameter<bool>("direct_visual_frontend", direct_visual_frontend);
  direct_visual_require_previous_weak_direction = node->declare_parameter<bool>(
      "direct_visual_require_previous_weak_direction", direct_visual_require_previous_weak_direction);

  // ---- radar topic + ego-velocity estimator params (fork addition; default-off) ----
  radar_topic = node->declare_parameter<std::string>("radar_topic", radar_topic);
  if (!radar_topic.empty()) {
    radar_ego_velocity_config.doppler_sign =
        node->declare_parameter<double>("radar_doppler_sign", radar_ego_velocity_config.doppler_sign);
    radar_ego_velocity_config.ransac_inlier_threshold = node->declare_parameter<double>(
        "radar_ransac_inlier_threshold", radar_ego_velocity_config.ransac_inlier_threshold);
    const auto radar_ransac_iterations = node->declare_parameter<int>(
        "radar_ransac_iterations", static_cast<int>(radar_ego_velocity_config.ransac_iterations));
    radar_ego_velocity_config.ransac_iterations =
        static_cast<std::size_t>(radar_ransac_iterations > 0 ? radar_ransac_iterations : 1);
    const auto radar_min_inliers = node->declare_parameter<int>(
        "radar_min_inliers", static_cast<int>(radar_ego_velocity_config.min_inliers));
    radar_ego_velocity_config.min_inliers = static_cast<std::size_t>(radar_min_inliers > 0 ? radar_min_inliers : 1);
    radar_velocity_scale = node->declare_parameter<double>("radar_velocity_scale", radar_velocity_scale);
  }

  // Lidar per-point timestamp processing params, namespaced under lidar_timestamps.*
  timestamp_proc_config.multiplier_to_seconds = node->declare_parameter<double>(
      "lidar_timestamps.multiplier_to_seconds", timestamp_proc_config.multiplier_to_seconds);
  timestamp_proc_config.offset_seconds =
      node->declare_parameter<double>("lidar_timestamps.offset_seconds", timestamp_proc_config.offset_seconds);
  timestamp_proc_config.force_absolute =
      node->declare_parameter<bool>("lidar_timestamps.force_absolute", timestamp_proc_config.force_absolute);
  timestamp_proc_config.force_relative =
      node->declare_parameter<bool>("lidar_timestamps.force_relative", timestamp_proc_config.force_relative);

  // manually, if, define extrinsics
  parse_cli_extrinsics();
  if (lio->config.visual_fusion.enabled) {
    if (direct_visual_frontend) {
      configure_direct_visual_frontend();
    } else {
      load_visual_constraints();
    }
  }

  RCLCPP_INFO_STREAM(node->get_logger(),
                     "Subscribed to IMU: "
                         << imu_topic << (!imu_frame.empty() ? " (frame " + imu_frame + ")" : "") << " and LiDAR: "
                         << lidar_topic << (!lidar_frame.empty() ? " (frame " + lidar_frame + ")" : "")
                         << ". Max number of threads: " << lio_config.max_num_threads << ". Publishing odometry to "
                         << odom_topic << " ( " << odom_frame
                         << " ) and acceleration "
                            "estimates to rko_lio/lidar_acceleration. Deskewing is "
                         << (lio->config.deskew ? "enabled" : "disabled") << "."
                         << (publish_deskewed_scan ? (" Publishing deskewed_cloud to " + deskewed_scan_topic + ".")
                                                   : ""));

  // disk logging
  dump_results = node->declare_parameter<bool>("dump_results", dump_results);
  results_dir = node->declare_parameter<std::string>("results_dir", results_dir);
  run_name = node->declare_parameter<std::string>("run_name", run_name);
  rclcpp::on_shutdown([this]() {
    // i'll need to look into rclcpp::Context a bit more, but for now i think this callback should be called before
    // anything gets destroyed.
    if (dump_results) {
      // it is probably still a veery good idea to make dump_results_to_disk noexcept
      dump_results_to_disk(results_dir, run_name);
    }
  });

  RCLCPP_INFO(node->get_logger(), "RKO LIO Node is up!");
}

void BaseNode::parse_cli_extrinsics() {
  auto parse_extrinsic = [this](const std::string& name, Sophus::SE3d& extrinsic) {
    const std::string param_name = "extrinsic_" + name + "2base_quat_xyzw_xyz";
    const std::vector<double> vec = node->declare_parameter<std::vector<double>>(param_name, std::vector<double>{});

    if (vec.size() != 7) {
      if (!vec.empty()) {
        RCLCPP_WARN_STREAM(node->get_logger(),
                           "Parameter 'extrinsic_"
                               << name << "2base_quat_xyzw_xyz' is set but has wrong size: " << vec.size()
                               << ". Expected 7 (qx, qy, qz, qw, x, y, z). check the value: "
                               << Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size()).transpose());
      }
      return false;
    }
    Eigen::Quaterniond q(vec[3], vec[0], vec[1], vec[2]); // qw, qx, qy, qz
    if (q.norm() < 1e-6) {
      throw std::runtime_error(name + " extrinsic quaternion has zero norm");
    }
    extrinsic = Sophus::SE3d(q, Eigen::Vector3d(vec[4], vec[5], vec[6]));
    RCLCPP_INFO_STREAM(node->get_logger(), "Parsed " << name << " extrinsic as: " << extrinsic.log().transpose());
    return true;
  };
  const bool imu_ok = parse_extrinsic("imu", extrinsic_imu2base);
  const bool lidar_ok = parse_extrinsic("lidar", extrinsic_lidar2base);
  if (lio->config.visual_fusion.enabled) {
    visual_extrinsic_set = parse_extrinsic("cam", extrinsic_cam2base);
  }
  if (lio->config.radar_velocity_fusion) {
    radar_extrinsic_set = parse_extrinsic("radar", extrinsic_radar2base);
    if (!radar_extrinsic_set) {
      RCLCPP_WARN_STREAM(node->get_logger(),
                         "radar_velocity_fusion is enabled but extrinsic_radar2base_quat_xyzw_xyz was not set; "
                         "assuming identity rotation between radar and base.");
    }
  }
  extrinsics_set = imu_ok && lidar_ok;
}

bool BaseNode::ensure_frame_and_extrinsics(std::string& target_frame,
                                           const std::string& msg_frame,
                                           std::string_view kind) {
  if (target_frame.empty()) {
    if (msg_frame.empty() && !extrinsics_set) {
      throw std::runtime_error(std::string(kind) +
                               " message header has no frame id and we need it to query TF for the extrinsics. "
                               "Either specify the frame id or the extrinsic manually.");
    }
    target_frame = msg_frame;
    RCLCPP_INFO_STREAM(node->get_logger(), "Parsed the " << kind << " frame id as: " << target_frame);
  }
  return check_and_set_extrinsics();
}

bool BaseNode::check_and_set_extrinsics() {
  if (extrinsics_set) {
    return true;
  }
  const std::optional<Sophus::SE3d> imu_transform = utils::get_transform(tf_buffer, imu_frame, base_frame, 0s);
  if (!imu_transform) {
    return false;
  }
  const std::optional<Sophus::SE3d> lidar_transform = utils::get_transform(tf_buffer, lidar_frame, base_frame, 0s);
  if (!lidar_transform) {
    return false;
  }
  extrinsic_imu2base = imu_transform.value();
  extrinsic_lidar2base = lidar_transform.value();
  extrinsics_set = true;
  return true;
}

std::tuple<core::Timestamps, core::Vector3dVector>
BaseNode::process_lidar_msg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) const {
  const core::Nsec header_stamp = utils::to_ns(lidar_msg->header.stamp);
  if (lio->config.deskew) {
    const auto& [scan, raw_timestamps] = utils::point_cloud2_to_eigen_with_timestamps(lidar_msg);
    const core::Timestamps& timestamps = core::process_timestamps(raw_timestamps, header_stamp, timestamp_proc_config);
    return {timestamps, scan};
  }
  RCLCPP_WARN_STREAM_ONCE(node->get_logger(), "Deskewing is disabled. Populating timestamps with static header time.");
  const core::Vector3dVector scan = utils::point_cloud2_to_eigen(lidar_msg);
  const core::Nsec corrected_stamp =
      header_stamp + std::chrono::duration_cast<core::Nsec>(std::chrono::duration<double>(
                         timestamp_proc_config.offset_seconds));
  return {{.min = corrected_stamp, .max = corrected_stamp, .times = core::TimestampVector(scan.size(), corrected_stamp)},
          scan};
}

std::vector<float>
BaseNode::process_lidar_intensity(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) const {
  // Only parsed when an intensity-based feature is enabled: point_cloud2_to_intensity() still
  // iterates the whole cloud, so skip it entirely on the (default) common path.
  const bool intensity_needed = lio->config.intensity_constraint || lio->config.intensity_disagreement_gate;
  if (!intensity_needed) {
    return {};
  }
  std::vector<float> intensities = utils::point_cloud2_to_intensity(lidar_msg);
  if (intensities.empty()) {
    RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                            "intensity_constraint/intensity_disagreement_gate is enabled but the point cloud on "
                                << lidar_topic
                                << " has no usable 'reflectivity' (uint16) or 'intensity' (float32) field; the "
                                   "intensity feature(s) will not fire.");
  }
  return intensities;
}

core::Vector3dVector BaseNode::register_scan_locked(const core::Vector3dVector& scan,
                                                    const core::TimestampVector& time_vector,
                                                    const std::vector<float>* intensities) {
  if (publish_local_map) {
    std::lock_guard lock(local_map_mutex); // map publish thread may access map simultaneously
    return lio->register_scan(extrinsic_lidar2base, scan, time_vector, intensities);
  }
  return lio->register_scan(extrinsic_lidar2base, scan, time_vector, intensities);
}

void BaseNode::publish_lidar_outputs(const core::Vector3dVector& deskewed_frame) const {
  if (publish_deskewed_scan) {
    std_msgs::msg::Header header;
    header.frame_id = lidar_frame;
    header.stamp = utils::to_ros_time(lio->lidar_state.time);
    frame_publisher->publish(utils::eigen_to_point_cloud2(deskewed_frame, header));
  }
  publish_odometry(lio->lidar_state, odom_publisher);
  if (publish_lidar_acceleration) {
    publish_lidar_accel(lio->lidar_state);
  }
}

void BaseNode::publish_odometry(const core::State& state,
                                const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr& publisher) const {
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = utils::to_ros_time(state.time);
  odom_msg.header.frame_id = odom_frame;
  odom_msg.child_frame_id = base_frame;
  odom_msg.pose.pose = utils::sophus_to_pose(state.pose);
  // [v0.8 Phase 1, diagnostic-only] anisotropic covariance from the final ICP Hessian; see
  // pose_covariance_from_state's doc comment above. Only this field is new -- pose.pose and
  // twist.twist are set exactly as before.
  odom_msg.pose.covariance = pose_covariance_from_state(state);
  utils::eigen_vector3d_to_ros_xyz(state.velocity, odom_msg.twist.twist.linear);
  utils::eigen_vector3d_to_ros_xyz(state.angular_velocity, odom_msg.twist.twist.angular);
  publisher->publish(odom_msg);
}

void BaseNode::publish_tf(const core::State& state) const {
  geometry_msgs::msg::TransformStamped transform_msg;
  transform_msg.header.stamp = utils::to_ros_time(state.time);
  if (invert_odom_tf) {
    transform_msg.header.frame_id = base_frame;
    transform_msg.child_frame_id = odom_frame;
    transform_msg.transform = utils::sophus_to_transform(state.pose.inverse());
  } else {
    transform_msg.header.frame_id = odom_frame;
    transform_msg.child_frame_id = base_frame;
    transform_msg.transform = utils::sophus_to_transform(state.pose);
  }
  tf_broadcaster->sendTransform(transform_msg);
}

void BaseNode::publish_lidar_accel(const core::State& state) const {
  auto accel_msg = geometry_msgs::msg::AccelStamped();
  accel_msg.header.stamp = utils::to_ros_time(state.time);
  accel_msg.header.frame_id = base_frame;
  utils::eigen_vector3d_to_ros_xyz(state.linear_acceleration, accel_msg.accel.linear);
  lidar_accel_publisher->publish(accel_msg);
}

void BaseNode::publish_map_loop() {
  while (atomic_node_running) {
    std::this_thread::sleep_for(publish_map_after);
    std::unique_lock lock(local_map_mutex);
    if (lio->map.empty()) {
      RCLCPP_WARN_ONCE(node->get_logger(), "Local map publish thread: Local map is empty.");
      continue;
    }
    const core::Vector3dVector map_points = lio->map.pointcloud();
    lock.unlock(); // we don't access the local map anymore
    std_msgs::msg::Header map_header;
    map_header.stamp = node->now();
    map_header.frame_id = odom_frame;
    map_publisher->publish(utils::eigen_to_point_cloud2(map_points, map_header));
  }
}

BaseNode::~BaseNode() { atomic_node_running = false; }

// ---- radar ego-velocity fusion (fork addition) ----

void BaseNode::radar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& radar_msg) {
  if (!lio->config.radar_velocity_fusion) {
    return;
  }
  const bool has_velocity_field =
      std::any_of(radar_msg->fields.cbegin(), radar_msg->fields.cend(), [](const auto& field) {
        return field.name == "velocity" && field.datatype == sensor_msgs::msg::PointField::FLOAT32;
      });
  if (!has_velocity_field) {
    RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                            "Radar point cloud on " << radar_topic << " has no float32 'velocity' field; ignoring.");
    return;
  }

  const std::size_t point_count = static_cast<std::size_t>(radar_msg->height) * radar_msg->width;
  std::vector<core::RadarDopplerMeasurement> measurements;
  measurements.reserve(point_count);
  sensor_msgs::PointCloud2ConstIterator<float> msg_x(*radar_msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> msg_y(*radar_msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> msg_z(*radar_msg, "z");
  sensor_msgs::PointCloud2ConstIterator<float> msg_velocity(*radar_msg, "velocity");
  for (std::size_t i = 0; i < point_count; ++i, ++msg_x, ++msg_y, ++msg_z, ++msg_velocity) {
    const Eigen::Vector3d point(*msg_x, *msg_y, *msg_z);
    const double range = point.norm();
    if (!(range > 1.0e-6) || !std::isfinite(*msg_velocity)) {
      continue;
    }
    measurements.push_back({point / range, static_cast<double>(*msg_velocity)});
  }

  const core::RadarEgoVelocityResult velocity_result =
      core::estimate_radar_ego_velocity(measurements, radar_ego_velocity_config);
  if (!velocity_result.valid) {
    return;
  }

  core::RadarVelocityPrior prior;
  prior.time = utils::to_ns(radar_msg->header.stamp);
  prior.velocity_base = radar_velocity_scale * (extrinsic_radar2base.so3() * velocity_result.velocity);
  {
    std::lock_guard lock(radar_prior_mutex);
    radar_prior_queue.push_back(prior);
    // Bound the queue for online use; offline the registration loop drains it.
    constexpr std::size_t kMaxBufferedRadarPriors = 2000;
    if (radar_prior_queue.size() > kMaxBufferedRadarPriors) {
      radar_prior_queue.pop_front();
    }
  }
}

void BaseNode::prepare_radar_prior(const core::Nsec& lidar_time) {
  if (!lio->config.radar_velocity_fusion) {
    return;
  }
  const double tolerance = lio->config.radar_prior_max_time_offset_sec;
  std::optional<core::RadarVelocityPrior> best;
  {
    std::lock_guard lock(radar_prior_mutex);
    // Drop estimates that can never match a future scan again.
    while (!radar_prior_queue.empty() &&
           core::to_seconds(lidar_time - radar_prior_queue.front().time) > tolerance) {
      radar_prior_queue.pop_front();
    }
    for (const auto& candidate : radar_prior_queue) {
      const double offset = core::to_seconds(candidate.time - lidar_time);
      if (offset > tolerance) {
        break; // queue is time-ordered; later entries are only further away
      }
      if (!best.has_value() || std::abs(offset) < std::abs(core::to_seconds(best->time - lidar_time))) {
        best = candidate;
      }
    }
  }
  if (best.has_value()) {
    lio->set_radar_velocity_prior(*best);
  } else {
    lio->clear_radar_velocity_prior();
  }
}

// ---- selective visual fusion / direct visual odometry frontend (fork addition) ----

void BaseNode::load_visual_constraints() {
  const std::string path = node->declare_parameter<std::string>("visual_constraints_path", "");
  if (path.empty()) {
    throw std::runtime_error("selective_visual_fusion requires visual_constraints_path during artifact-driven "
                             "development");
  }
  if (!visual_extrinsic_set) {
    throw std::runtime_error("selective_visual_fusion requires extrinsic_cam2base_quat_xyzw_xyz");
  }
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("failed to open visual constraint artifact: " + path);
  }
  nlohmann::json report;
  stream >> report;
  if (report.at("schema_version").get<int>() != 1) {
    throw std::runtime_error("unsupported visual constraint schema");
  }
  for (const auto& record : report.at("constraints")) {
    if (!record.value("accepted", false)) {
      continue;
    }
    VisualRelativeConstraint constraint;
    constraint.first_time =
        std::chrono::duration_cast<core::Nsec>(std::chrono::duration<double>(record.at("first_stamp").get<double>()));
    constraint.second_time = std::chrono::duration_cast<core::Nsec>(
        std::chrono::duration<double>(record.at("second_stamp").get<double>()));
    const auto rotation = record.at("rotation").get<std::vector<std::vector<double>>>();
    const auto direction = record.at("translation_direction").get<std::vector<double>>();
    if (rotation.size() != 3 || rotation[0].size() != 3 || rotation[1].size() != 3 || rotation[2].size() != 3 ||
        direction.size() != 3) {
      throw std::runtime_error("invalid visual rotation or translation direction shape");
    }
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        constraint.current_R_previous_camera(row, col) =
            rotation[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
      }
      constraint.current_t_previous_direction(row) = direction[static_cast<std::size_t>(row)];
    }
    constraint.confidence.tracks = record.at("tracks").get<std::size_t>();
    constraint.confidence.inliers = record.at("inliers").get<std::size_t>();
    constraint.confidence.rotation_error_deg = record.at("rotation_error_deg").get<double>();
    constraint.confidence.translation_cosine = record.at("translation_cosine").get<double>();
    constraint.confidence.baseline_m = record.at("predicted_translation_m").get<double>();
    visual_constraints.push_back(constraint);
  }
  std::sort(visual_constraints.begin(), visual_constraints.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.second_time < rhs.second_time; });
  RCLCPP_INFO_STREAM(node->get_logger(),
                     "Loaded " << visual_constraints.size() << " accepted visual constraints from " << path);
}

void BaseNode::configure_direct_visual_frontend() {
  visual_image_topic = node->declare_parameter<std::string>("visual_image_topic", visual_image_topic);
  visual_camera_time_offset_sec =
      node->declare_parameter<double>("visual_camera_time_offset_sec", visual_camera_time_offset_sec);
  visual_max_image_lidar_delta_sec =
      node->declare_parameter<double>("visual_max_image_lidar_delta_sec", visual_max_image_lidar_delta_sec);
  visual_keyframe_interval_sec =
      node->declare_parameter<double>("visual_keyframe_interval_sec", visual_keyframe_interval_sec);
  visual_camera.width = node->declare_parameter<int>("visual_camera_width", 0);
  visual_camera.height = node->declare_parameter<int>("visual_camera_height", 0);
  visual_camera.fx = node->declare_parameter<double>("visual_camera_fx", 0.0);
  visual_camera.fy = node->declare_parameter<double>("visual_camera_fy", 0.0);
  visual_camera.cx = node->declare_parameter<double>("visual_camera_cx", 0.0);
  visual_camera.cy = node->declare_parameter<double>("visual_camera_cy", 0.0);
  const auto distortion_model = node->declare_parameter<std::string>("visual_camera_distortion_model", "equidistant");
  if (distortion_model == "equidistant" || distortion_model == "fisheye") {
    visual_camera.distortion_model = core::CameraDistortionModel::equidistant;
  } else if (distortion_model == "plumb_bob" || distortion_model == "radtan") {
    visual_camera.distortion_model = core::CameraDistortionModel::plumb_bob;
  } else {
    throw std::runtime_error("direct_visual_frontend camera distortion model must be equidistant or plumb_bob");
  }
  const auto distortion = node->declare_parameter<std::vector<double>>("visual_camera_distortion", std::vector<double>{});
  if (distortion.size() != 4 || !visual_camera.valid()) {
    throw std::runtime_error("direct_visual_frontend requires valid intrinsics and four distortion coefficients");
  }
  std::copy(distortion.begin(), distortion.end(), visual_camera.distortion.begin());
  direct_visual_config.max_features =
      node->declare_parameter<int>("direct_visual_max_features", direct_visual_config.max_features);
  direct_visual_config.grid_cell_size =
      node->declare_parameter<int>("direct_visual_grid_cell_size", direct_visual_config.grid_cell_size);
  direct_visual_config.patch_radius =
      node->declare_parameter<int>("direct_visual_patch_radius", direct_visual_config.patch_radius);
  direct_visual_config.max_iterations =
      node->declare_parameter<int>("direct_visual_max_iterations", direct_visual_config.max_iterations);
  direct_visual_config.min_gradient =
      node->declare_parameter<double>("direct_visual_min_gradient", direct_visual_config.min_gradient);
  direct_visual_config.huber_delta =
      node->declare_parameter<double>("direct_visual_huber_delta", direct_visual_config.huber_delta);
  direct_visual_config.occlusion_tolerance_m = node->declare_parameter<double>(
      "direct_visual_occlusion_tolerance_m", direct_visual_config.occlusion_tolerance_m);
  direct_visual_config.min_residuals =
      node->declare_parameter<int>("direct_visual_min_residuals", direct_visual_config.min_residuals);
  direct_visual_config.min_inlier_ratio =
      node->declare_parameter<double>("direct_visual_min_inlier_ratio", direct_visual_config.min_inlier_ratio);
  direct_visual_config.max_rmse =
      node->declare_parameter<double>("direct_visual_max_rmse", direct_visual_config.max_rmse);
  direct_visual_config.max_initial_rmse =
      node->declare_parameter<double>("direct_visual_max_initial_rmse", direct_visual_config.max_initial_rmse);
  direct_visual_config.max_translation_correction_m = node->declare_parameter<double>(
      "direct_visual_max_translation_correction_m", direct_visual_config.max_translation_correction_m);
  direct_visual_config.max_rotation_correction_rad = node->declare_parameter<double>(
      "direct_visual_max_rotation_correction_rad", direct_visual_config.max_rotation_correction_rad);
  RCLCPP_INFO_STREAM(node->get_logger(), "Live direct visual frontend: "
                                             << visual_image_topic << " " << visual_camera.width << "x"
                                             << visual_camera.height << " distortion=" << distortion_model
                                             << " keyframe interval " << visual_keyframe_interval_sec << " s");
}

void BaseNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg) {
  if (!direct_visual_frontend) {
    return;
  }
  if (image_msg->encoding != "mono8" || static_cast<int>(image_msg->width) != visual_camera.width ||
      static_cast<int>(image_msg->height) != visual_camera.height || image_msg->step < image_msg->width) {
    RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                            "Direct visual frontend requires mono8 images matching configured dimensions");
    return;
  }
  VisualImageFrame frame;
  frame.time = utils::to_ns(image_msg->header.stamp) +
               std::chrono::duration_cast<core::Nsec>(std::chrono::duration<double>(visual_camera_time_offset_sec));
  frame.image.width = static_cast<int>(image_msg->width);
  frame.image.height = static_cast<int>(image_msg->height);
  frame.image.pixels.resize(static_cast<std::size_t>(frame.image.width * frame.image.height));
  for (int row = 0; row < frame.image.height; ++row) {
    std::copy_n(image_msg->data.begin() + static_cast<std::ptrdiff_t>(row * image_msg->step), frame.image.width,
               frame.image.pixels.begin() + static_cast<std::ptrdiff_t>(row * frame.image.width));
  }
  std::lock_guard lock(visual_buffer_mutex);
  visual_image_buffer.push(std::move(frame));
  while (visual_image_buffer.size() > 400) {
    visual_image_buffer.pop();
  }
}

std::optional<LiveVisualKeyframe> BaseNode::prepare_direct_visual_prior(const core::Vector3dVector& scan,
                                                                        const core::Nsec& lidar_time) {
  if (!direct_visual_frontend) {
    return std::nullopt;
  }
  std::optional<VisualImageFrame> selected;
  {
    std::lock_guard lock(visual_buffer_mutex);
    while (!visual_image_buffer.empty() && visual_image_buffer.front().time <= lidar_time) {
      selected = std::move(visual_image_buffer.front());
      visual_image_buffer.pop();
    }
  }
  if (!selected.has_value() || core::to_seconds(lidar_time - selected->time) > visual_max_image_lidar_delta_sec ||
      (live_visual_keyframe.has_value() &&
       core::to_seconds(selected->time - live_visual_keyframe->time) < visual_keyframe_interval_sec)) {
    return std::nullopt;
  }
  LiveVisualKeyframe pending;
  pending.time = selected->time;
  pending.image = std::move(selected->image);
  const Sophus::SE3d camera_T_lidar = extrinsic_cam2base.inverse() * extrinsic_lidar2base;
  pending.depth = core::project_sparse_depth(scan, camera_T_lidar, visual_camera);
  if (!live_visual_keyframe.has_value()) {
    return pending;
  }
  if (direct_visual_require_previous_weak_direction &&
      (!lio->lidar_state.icp_diagnostics.has_value() ||
       core::count_weak_information_directions(lio->lidar_state.icp_diagnostics->H, lio->config.visual_fusion) ==
           0)) {
    ++direct_visual_weak_gate_skip_count;
    return pending;
  }

  const Sophus::SE3d predicted_image_base = lio->predict_pose_at(pending.time);
  const Sophus::SE3d predicted_lidar_base = lio->predict_pose_at(lidar_time);
  const Sophus::SE3d previous_camera = live_visual_keyframe->world_T_base * extrinsic_cam2base;
  const Sophus::SE3d predicted_camera = predicted_image_base * extrinsic_cam2base;
  const Sophus::SE3d initial_relative = predicted_camera.inverse() * previous_camera;
  ++direct_visual_attempt_count;
  const auto direct = core::align_direct_visual(live_visual_keyframe->image, live_visual_keyframe->depth,
                                                pending.image, pending.depth, visual_camera, initial_relative,
                                                direct_visual_config);
  if (direct_visual_attempt_count % 50 == 0) {
    RCLCPP_INFO_STREAM(node->get_logger(), "Direct visual probe "
                                               << direct_visual_attempt_count << ": valid=" << direct.valid
                                               << " features=" << direct.features
                                               << " tracked=" << direct.tracked_features
                                               << " inlier_ratio=" << direct.inlier_ratio
                                               << " rmse=" << direct.final_rmse);
  }
  DirectVisualDiagnosticsSample diagnostics;
  diagnostics.time = pending.time;
  diagnostics.solver_valid = direct.valid;
  diagnostics.failure_reason = static_cast<int>(direct.failure_reason);
  diagnostics.features = direct.features;
  diagnostics.tracked_features = direct.tracked_features;
  diagnostics.inliers = direct.inliers;
  diagnostics.inlier_ratio = direct.inlier_ratio;
  diagnostics.initial_rmse = direct.initial_rmse;
  diagnostics.final_rmse = direct.final_rmse;
  diagnostics.exposure_gain = direct.exposure_gain;
  diagnostics.exposure_bias = direct.exposure_bias;
  diagnostics.predicted_baseline_m = initial_relative.translation().norm();
  if (!direct.valid) {
    direct_visual_diagnostics.push_back(diagnostics);
    return pending;
  }
  ++direct_visual_solver_valid_count;
  const Eigen::Matrix3d rotation_difference =
      direct.current_T_previous_camera.rotationMatrix() * initial_relative.rotationMatrix().transpose();
  const double rotation_cosine = std::clamp((rotation_difference.trace() - 1.0) * 0.5, -1.0, 1.0);
  const double rotation_error_deg = std::acos(rotation_cosine) * 180.0 / std::numbers::pi;
  const Eigen::Vector3d predicted_translation = initial_relative.translation();
  const Eigen::Vector3d measured_translation = direct.current_T_previous_camera.translation();
  const double predicted_norm = predicted_translation.norm();
  const double measured_norm = measured_translation.norm();
  const double translation_cosine = predicted_norm > 1.0e-9 && measured_norm > 1.0e-9
                                        ? predicted_translation.dot(measured_translation) /
                                              (predicted_norm * measured_norm)
                                        : -1.0;
  diagnostics.rotation_error_deg = rotation_error_deg;
  diagnostics.translation_cosine = translation_cosine;
  diagnostics.baseline_m = measured_norm;
  diagnostics.predicted_baseline_m = predicted_norm;
  core::VisualConstraintConfidence confidence;
  confidence.tracks = static_cast<std::size_t>(direct.tracked_features);
  confidence.inliers = static_cast<std::size_t>(direct.inliers);
  confidence.rotation_error_deg = rotation_error_deg;
  confidence.translation_cosine = translation_cosine;
  confidence.baseline_m = measured_norm;
  const Eigen::Matrix6d camera_adjoint_inverse = predicted_camera.Adj().inverse();
  confidence.visual_information = camera_adjoint_inverse.transpose() * direct.pose_information * camera_adjoint_inverse;
  diagnostics.confidence_gate_passed = core::visual_constraint_passes_gate(confidence, lio->config.visual_fusion);
  direct_visual_diagnostics.push_back(diagnostics);
  if (!diagnostics.confidence_gate_passed) {
    return pending;
  }
  const Sophus::SE3d desired_image_camera = previous_camera * direct.current_T_previous_camera.inverse();
  const Sophus::SE3d desired_image_base = desired_image_camera * extrinsic_cam2base.inverse();
  const Sophus::SE3d image_T_lidar_prediction = predicted_image_base.inverse() * predicted_lidar_base;
  const Sophus::SE3d desired_lidar_base = desired_image_base * image_T_lidar_prediction;
  lio->set_visual_pose_prior({lidar_time, desired_lidar_base, confidence});
  ++direct_visual_valid_count;
  return pending;
}

void BaseNode::commit_direct_visual_keyframe(LiveVisualKeyframe frame,
                                             const core::Nsec& lidar_time,
                                             const core::Vector3dVector& deskewed_lidar_frame) {
  const double dt = core::to_seconds(frame.time - lidar_time);
  Eigen::Vector6d correction = Eigen::Vector6d::Zero();
  correction.head<3>() = lio->lidar_state.velocity * dt;
  correction.tail<3>() = lio->lidar_state.angular_velocity * dt;
  frame.world_T_base = lio->lidar_state.pose * Sophus::SE3d::exp(correction);
  const Sophus::SE3d camera_T_lidar = extrinsic_cam2base.inverse() * extrinsic_lidar2base;
  frame.depth = core::project_sparse_depth(deskewed_lidar_frame, camera_T_lidar, visual_camera);
  live_visual_keyframe = std::move(frame);
}

void BaseNode::prepare_visual_prior(const core::Nsec& lidar_time) {
  if (!lio->config.visual_fusion.enabled || visual_constraints.empty() || lio->poses_with_timestamps.empty()) {
    return;
  }
  const double tolerance = lio->config.visual_prior_max_time_offset_sec;
  while (next_visual_constraint < visual_constraints.size() &&
         core::to_seconds(visual_constraints[next_visual_constraint].second_time - lidar_time) < -tolerance) {
    ++next_visual_constraint;
  }
  if (next_visual_constraint >= visual_constraints.size()) {
    return;
  }
  const auto& constraint = visual_constraints[next_visual_constraint];
  if (std::abs(core::to_seconds(constraint.second_time - lidar_time)) > tolerance) {
    return;
  }

  const auto nearest = std::min_element(
      lio->poses_with_timestamps.begin(), lio->poses_with_timestamps.end(), [&constraint](const auto& lhs, const auto& rhs) {
        return std::abs(core::to_seconds(lhs.first - constraint.first_time)) <
               std::abs(core::to_seconds(rhs.first - constraint.first_time));
      });
  if (nearest == lio->poses_with_timestamps.end() ||
      std::abs(core::to_seconds(nearest->first - constraint.first_time)) > tolerance) {
    ++next_visual_constraint;
    return;
  }
  const double prediction_dt = core::to_seconds(constraint.second_time - lio->lidar_state.time);
  Eigen::Vector6d prediction = Eigen::Vector6d::Zero();
  prediction.head<3>() = lio->lidar_state.velocity * prediction_dt;
  prediction.tail<3>() = lio->lidar_state.angular_velocity * prediction_dt;
  const Sophus::SE3d predicted_current_pose = lio->lidar_state.pose * Sophus::SE3d::exp(prediction);
  const Sophus::SE3d desired_pose =
      desired_base_pose_from_monocular_relative(nearest->second, predicted_current_pose, extrinsic_cam2base,
                                                constraint.current_R_previous_camera,
                                                constraint.current_t_previous_direction);
  auto confidence = constraint.confidence;
  confidence.baseline_m =
      ((predicted_current_pose * extrinsic_cam2base).inverse() * (nearest->second * extrinsic_cam2base))
          .translation()
          .norm();
  lio->set_visual_pose_prior({constraint.second_time, desired_pose, confidence});
  ++next_visual_constraint;
}

void BaseNode::dump_results_to_disk(const std::filesystem::path& results_dir, const std::string& run_name) const {
  try {
    std::filesystem::create_directories(results_dir); // no error if exists
    int index = 0;
    std::filesystem::path output_dir = results_dir / (run_name + "_" + std::to_string(index));
    while (std::filesystem::exists(output_dir)) {
      ++index;
      output_dir = results_dir / (run_name + "_" + std::to_string(index));
    }
    std::filesystem::create_directory(output_dir);
    const std::filesystem::path output_file = output_dir / (run_name + "_tum_" + std::to_string(index) + ".txt");
    // dump poses
    if (std::ofstream file(output_file); file.is_open()) {
      for (const auto& [timestamp, pose] : lio->poses_with_timestamps) {
        const Eigen::Vector3d& translation = pose.translation();
        const Eigen::Quaterniond& quaternion = pose.so3().unit_quaternion();
        file << std::fixed << std::setprecision(6) << core::to_seconds(timestamp) << " " << translation.x() << " "
             << translation.y() << " " << translation.z() << " " << quaternion.x() << " " << quaternion.y() << " "
             << quaternion.z() << " " << quaternion.w() << "\n";
      }
      std::cout << "Poses written to " << std::filesystem::absolute(output_file) << "\n";
    }
    // dump config
    const nlohmann::json json_config = {{"config", lio->config}};
    const std::filesystem::path config_file = output_dir / "config.json";
    if (std::ofstream file(config_file); file.is_open()) {
      file << json_config.dump(4);
      std::cout << "Configuration written to " << config_file << "\n";
    }
    // [v0.8 Phase 1] opt-in per-scan persistence-gate diagnostics (degeneracy_persistence_gate).
    if (!lio->degeneracy_persistence_diagnostics.empty()) {
      const std::filesystem::path diagnostics_file = output_dir / "degeneracy_persistence.csv";
      if (std::ofstream file(diagnostics_file); file.is_open()) {
        file << "timestamp,candidate_available,consecutive_scans,matched_absolute_cosine,confirmed,"
                "observability_window_scans,aggregate_directional_information_ratio,"
                "multiscan_observability_confirmed,"
                "intervention_count,axis_tx,axis_ty,axis_tz,axis_rx,axis_ry,axis_rz\n";
        for (const auto& sample : lio->degeneracy_persistence_diagnostics) {
          const auto& state = sample.persistent_weak_direction;
          file << std::fixed << std::setprecision(9) << core::to_seconds(sample.time) << ","
               << static_cast<int>(state.candidate_available) << "," << state.consecutive_scans << ","
               << state.matched_absolute_cosine << "," << static_cast<int>(state.confirmed) << ","
               << state.observability_window_scans << "," << state.aggregate_directional_information_ratio << ","
               << static_cast<int>(state.multiscan_observability_confirmed) << "," << sample.intervention_count;
          for (int axis = 0; axis < 6; ++axis) {
            file << "," << state.axis(axis);
          }
          file << "\n";
        }
        std::cout << "Degeneracy persistence diagnostics written to " << diagnostics_file << "\n";
      }
    }
    // Selective visual fusion / direct visual odometry summary.
    if (lio->config.visual_fusion.enabled) {
      const nlohmann::json visual_summary = {{"prior_attempt_count", lio->visual_prior_attempt_count},
                                             {"fused_scan_count", lio->visual_fused_scan_count},
                                             {"fused_direction_count", lio->visual_fused_direction_count},
                                             {"visual_unobservable_direction_count",
                                              lio->visual_unobservable_direction_count},
                                             {"direct_attempt_count", direct_visual_attempt_count},
                                             {"direct_weak_gate_skip_count", direct_visual_weak_gate_skip_count},
                                             {"direct_solver_valid_count", direct_visual_solver_valid_count},
                                             {"direct_valid_count", direct_visual_valid_count}};
      const std::filesystem::path visual_file = output_dir / "visual_fusion_summary.json";
      if (std::ofstream file(visual_file); file.is_open()) {
        file << visual_summary.dump(4) << "\n";
        std::cout << "Visual fusion summary written to " << visual_file << "\n";
      }
    }
    // Radar ego-velocity fusion summary.
    if (lio->config.radar_velocity_fusion) {
      const nlohmann::json radar_summary = {
          {"prior_attempt_count", lio->radar_prior_attempt_count},
          {"fused_scan_count", lio->radar_fused_scan_count},
          {"disagreement_corrected_scan_count", lio->radar_disagreement_corrected_scan_count}};
      const std::filesystem::path radar_file = output_dir / "radar_velocity_fusion_summary.json";
      if (std::ofstream file(radar_file); file.is_open()) {
        file << radar_summary.dump(4) << "\n";
        std::cout << "Radar velocity fusion summary written to " << radar_file << "\n";
      }
    }
    // Intensity/reflectivity texture constraint summary.
    if (lio->config.intensity_constraint || lio->config.intensity_disagreement_gate) {
      nlohmann::json intensity_summary = {{"prior_attempt_count", lio->intensity_prior_attempt_count},
                                          {"prior_applied_count", lio->intensity_prior_applied_count}};
      if (lio->config.intensity_disagreement_gate) {
        intensity_summary["disagreement_corrected_scan_count"] = lio->intensity_disagreement_corrected_scan_count;
        intensity_summary["disagreement_attempt_count"] = lio->intensity_disagreement_attempt_count;
        intensity_summary["disagreement_valid_shift_count"] = lio->intensity_disagreement_valid_shift_count;
        intensity_summary["disagreement_exceeded_threshold_count"] =
            lio->intensity_disagreement_exceeded_threshold_count;
      }
      const std::filesystem::path intensity_file = output_dir / "intensity_constraint_summary.json";
      if (std::ofstream file(intensity_file); file.is_open()) {
        file << intensity_summary.dump(4) << "\n";
        std::cout << "Intensity constraint summary written to " << intensity_file << "\n";
      }
    }
    if (!lio->visual_observability_diagnostics.empty()) {
      const std::filesystem::path observability_file = output_dir / "visual_directional_observability.csv";
      if (std::ofstream file(observability_file); file.is_open()) {
        file << "timestamp,direction_index,directional_information_ratio\n";
        for (const auto& sample : lio->visual_observability_diagnostics) {
          for (std::size_t index = 0; index < sample.ratio_count; ++index) {
            file << std::fixed << std::setprecision(9) << core::to_seconds(sample.time) << "," << index << ","
                 << sample.directional_information_ratios[index] << "\n";
          }
        }
      }
    }
    if (!direct_visual_diagnostics.empty()) {
      const std::filesystem::path diagnostics_file = output_dir / "direct_visual_diagnostics.csv";
      if (std::ofstream file(diagnostics_file); file.is_open()) {
        file << "timestamp,solver_valid,confidence_gate_passed,failure_reason,features,tracked_features,inliers,"
                "inlier_ratio,initial_rmse,final_rmse,exposure_gain,exposure_bias,"
                "rotation_error_deg,translation_cosine,baseline_m,predicted_baseline_m\n";
        for (const auto& sample : direct_visual_diagnostics) {
          file << std::fixed << std::setprecision(9) << core::to_seconds(sample.time) << ","
               << static_cast<int>(sample.solver_valid) << "," << static_cast<int>(sample.confidence_gate_passed)
               << "," << sample.failure_reason << "," << sample.features << "," << sample.tracked_features << ","
               << sample.inliers << "," << sample.inlier_ratio << "," << sample.initial_rmse << ","
               << sample.final_rmse << "," << sample.exposure_gain << "," << sample.exposure_bias << ","
               << sample.rotation_error_deg << "," << sample.translation_cosine << "," << sample.baseline_m << ","
               << sample.predicted_baseline_m << "\n";
        }
      }
    }
  } catch (const std::filesystem::filesystem_error& ex) {
    std::cerr << "[WARNING] Cannot write files to disk, encountered filesystem error: " << ex.what() << "\n";
  }
}

} // namespace rko_lio::ros
