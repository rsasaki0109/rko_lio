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

#include "node.hpp"
#include "rko_lio/core/process_timestamps.hpp"
#include "rko_lio/core/profiler.hpp"
#include "rko_lio/ros/utils/utils.hpp"
// other
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <rclcpp/serialization.hpp>
#include <stdexcept>

namespace {
using namespace std::literals;

rko_lio::core::ImuControl imu_msg_to_imu_data(const sensor_msgs::msg::Imu& imu_msg) {
  rko_lio::core::ImuControl imu_data;
  imu_data.time = rko_lio::ros::utils::ros_time_to_seconds(imu_msg.header.stamp);
  imu_data.angular_velocity = rko_lio::ros::utils::ros_xyz_to_eigen_vector3d(imu_msg.angular_velocity);
  imu_data.acceleration = rko_lio::ros::utils::ros_xyz_to_eigen_vector3d(imu_msg.linear_acceleration);
  return imu_data;
}

// ============================================================================
// [v0.8 Phase 1, diagnostic-only] anisotropic nav_msgs/Odometry.pose.covariance
// fill, derived from LIO::State::icp_diagnostics (rko_lio/core/util.hpp).
//
// This does not change odom_msg.pose.pose or odom_msg.twist.twist in any
// way -- those are still set exactly as before (see Node::publish_odometry
// below). Only the covariance field, which this fork previously left at its
// message-default (all zeros, i.e. "populated/unused" per
// docs/roadmap/v0.8.md §2 candidate B), is filled here.
//
// Scale rationale: at a Gauss-Newton optimum, the accumulated Hessian `H`
// (already averaged over correspondences by build_icp_linear_system, see
// lio.cpp) is the standard asymptotic information matrix of the ICP solve;
// its inverse is the corresponding pose-covariance approximation -- the same
// H <-> information-matrix relationship Zhang & Singh ("On Degeneracy of
// Optimization-based State Estimation Problems", ICRA 2016) use to define
// per-direction degeneracy. `H`'s eigenvectors/eigenvalues are reused
// directly from LocalizabilitySummary (already computed once in the fork
// core, not recomputed here) to build Cov = V * diag(1/lambda) * V^T.
//
// Near-exactly-degenerate directions (eigenvalue ~ 0 -- e.g. the synthetic
// corridor fixture's exact-zero along-axis eigenvalue, see
// docs/research/hilti-degeneracy-baseline.md §4) would blow up to +inf under
// a literal inverse; each eigenvalue is floored at
// max(kMinEigenvalueFloor, kRelativeEigenvalueFloor * lambda_max) before
// inverting, so the reported covariance stays finite while still reporting a
// very large (i.e. "not informative") uncertainty along that direction,
// rather than a numerically meaningless inf/nan on the wire.
constexpr double kMinEigenvalueFloor = 1e-9;
constexpr double kRelativeEigenvalueFloor = 1e-6;
// Fallback diagonal covariance (m^2 on the translation block, rad^2 on the
// rotation block) used when no ICP solve happened for this scan (first
// frame, a dropped scan, or a kidnap-recovery/local-reset scan -- see
// LIO::State::icp_diagnostics's doc comment): "no information available",
// deliberately not "perfectly known" (which a covariance of all zeros would
// imply to a consumer such as robot_localization).
constexpr double kNoDiagnosticsFallbackVariance = 1e6;

Sophus::SE3d desired_base_pose_from_monocular_relative(
    const Sophus::SE3d& world_T_previous_base,
    const Sophus::SE3d& world_T_predicted_current_base,
    const Sophus::SE3d& base_T_camera,
    const Eigen::Matrix3d& current_R_previous_camera,
    const Eigen::Vector3d& current_t_previous_direction) {
  const Sophus::SE3d world_T_previous_camera =
      world_T_previous_base * base_T_camera;
  const Sophus::SE3d world_T_predicted_current_camera =
      world_T_predicted_current_base * base_T_camera;
  const Sophus::SE3d predicted_current_T_previous_camera =
      world_T_predicted_current_camera.inverse() * world_T_previous_camera;
  const double baseline = predicted_current_T_previous_camera.translation().norm();
  const Sophus::SE3d measured_current_T_previous_camera(
      Sophus::SO3d(current_R_previous_camera),
      baseline * current_t_previous_direction.normalized());
  return world_T_previous_camera * measured_current_T_previous_camera.inverse() *
         base_T_camera.inverse();
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
  // Sophus::SE3d's se(3) tangent order (translation rho, then rotation phi)
  // matches geometry_msgs/PoseWithCovariance's documented covariance order
  // ([x, y, z, rot x, rot y, rot z]) exactly, so no axis permutation is
  // needed between H's ordering and the wire covariance.
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
                                   piecewise_gyro_deskew,
                                   fixed_lag_multiscan,
                                   fixed_lag_window_size,
                                   fixed_lag_neighbor_scans,
                                   fixed_lag_pairwise_max_iterations,
                                   fixed_lag_scan_constraint_weight,
                                   fixed_lag_odometry_prior_weight,
                                   fixed_lag_pairwise_max_translation_m,
                                   fixed_lag_pairwise_max_rotation_deg,
                                   fixed_lag_pairwise_min_correspondences,
                                   fixed_lag_pairwise_min_inlier_ratio,
                                   fixed_lag_pairwise_min_error_reduction,
                                   fixed_lag_max_pose_correction_m,
                                   fixed_lag_max_pose_correction_deg,
                                   fixed_lag_huber_delta_m,
                                   fixed_lag_map_max_error_ratio,
                                   fixed_lag_map_min_correspondence_ratio,
                                   max_iterations,
                                   voxel_size,
                                   max_points_per_voxel,
                                   max_range,
                                   min_range,
                                   convergence_criterion,
                                   max_correspondance_distance,
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
                                   max_scan_delta_sec,
                                   enable_kidnap_relocalization,
                                   reset_on_registration_failure,
                                   recovery_min_failures,
                                   relocalize_after_scan_gap,
                                   relocalization_min_correspondences,
                                   relocalization_min_inlier_ratio,
                                   relocalization_max_mean_error,
                                   relocalization_max_correspondance_distance,
                                   relocalization_yaw_samples,
                                   relocalization_pose_stride,
                                   relocalization_min_pose_separation,
                                   relocalization_max_iterations)
} // namespace rko_lio::core

namespace rko_lio::ros {

void BenchmarkConsumerCounters::configure(
    const std::string& output_path,
    const std::string& configured_phase_mode,
    const std::size_t expected,
    const std::map<std::string, std::size_t>& topic_counts,
    const std::int64_t required_end_timestamp) {
  if (output_path.empty() || configured_phase_mode.empty()) {
    return;
  }
  enabled = true;
  ack_backpressure_enabled = configured_phase_mode == "unpaced_ack";
  evidence_path = output_path;
  phase_mode = configured_phase_mode;
  expected_messages = expected;
  expected_topic_counts = topic_counts;
  required_end_timestamp_ns = required_end_timestamp;
}

void BenchmarkConsumerCounters::record_received() {
  if (enabled) {
    received_messages.fetch_add(1, std::memory_order_relaxed);
  }
}

void BenchmarkConsumerCounters::record_processed(
    const std::int64_t timestamp_ns, const std::uint64_t latency_ns) {
  if (!enabled) {
    return;
  }
  processed_messages.fetch_add(1, std::memory_order_relaxed);
  std::int64_t first = -1;
  first_processed_timestamp_ns.compare_exchange_strong(
      first, timestamp_ns, std::memory_order_relaxed);
  last_processed_timestamp_ns.store(timestamp_ns, std::memory_order_relaxed);
  auto previous = maximum_callback_latency_ns.load(std::memory_order_relaxed);
  while (previous < latency_ns &&
         !maximum_callback_latency_ns.compare_exchange_weak(
             previous, latency_ns, std::memory_order_relaxed)) {
  }
}

void BenchmarkConsumerCounters::record_drop(const bool overflow) {
  if (!enabled) {
    return;
  }
  dropped_messages.fetch_add(1, std::memory_order_relaxed);
  if (overflow) {
    queue_overflow.fetch_add(1, std::memory_order_relaxed);
  }
}

void BenchmarkConsumerCounters::observe_registration_queue(
    const std::size_t queue_size) {
  if (!enabled) {
    return;
  }
  auto previous = maximum_registration_queue_messages.load(std::memory_order_relaxed);
  while (previous < queue_size &&
         !maximum_registration_queue_messages.compare_exchange_weak(
             previous, queue_size, std::memory_order_relaxed)) {
  }
}

void BenchmarkConsumerCounters::record_processing_failure() {
  if (enabled) {
    processing_failures.fetch_add(1, std::memory_order_relaxed);
    record_drop(false);
  }
}

void BenchmarkConsumerCounters::record_pacing_late() {
  if (enabled) {
    pacing_late_messages.fetch_add(1, std::memory_order_relaxed);
  }
}

void BenchmarkConsumerCounters::mark_eof(const bool value) {
  if (enabled) {
    eof_observed.store(value, std::memory_order_release);
  }
}

void BenchmarkConsumerCounters::mark_drained(const bool value) {
  if (enabled) {
    drain_complete.store(value, std::memory_order_release);
  }
}

std::string BenchmarkConsumerCounters::failure_reason() const {
  std::lock_guard lock(failure_mutex_);
  return failure_reason_;
}

void BenchmarkConsumerCounters::set_failure_reason(const std::string& reason) {
  if (!enabled) {
    return;
  }
  std::lock_guard lock(failure_mutex_);
  if (failure_reason_.empty()) {
    failure_reason_ = reason;
  }
}

BenchmarkDrainSnapshot Node::benchmark_drain_snapshot() {
  BenchmarkDrainSnapshot snapshot;
  {
    std::lock_guard lock(buffer_mutex);
    snapshot.lidar_buffer_size = lidar_buffer.size();
    snapshot.imu_buffer_size = imu_buffer.size();
    if (!lidar_buffer.empty()) {
      snapshot.front_lidar_min_timestamp_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              lidar_buffer.front().timestamps.min).count();
      snapshot.front_lidar_max_timestamp_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              lidar_buffer.front().timestamps.max).count();
    }
    if (!imu_buffer.empty()) {
      snapshot.last_imu_timestamp_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              imu_buffer.back().time).count();
    }
    if (snapshot.front_lidar_max_timestamp_ns >= 0 &&
        snapshot.last_imu_timestamp_ns >= 0) {
      snapshot.timestamp_gap_ns =
          snapshot.last_imu_timestamp_ns - snapshot.front_lidar_max_timestamp_ns;
    }
    snapshot.atomic_can_process = atomic_can_process.load(std::memory_order_acquire);
    snapshot.registration_active = atomic_registration_active.load(std::memory_order_acquire);
  }
  snapshot.expected_messages = benchmark_consumer.expected_messages;
  snapshot.received_messages = benchmark_consumer.received_messages.load(std::memory_order_relaxed);
  snapshot.processed_messages = benchmark_consumer.processed_messages.load(std::memory_order_relaxed);
  snapshot.dropped_messages = benchmark_consumer.dropped_messages.load(std::memory_order_relaxed);
  snapshot.queue_overflow = benchmark_consumer.queue_overflow.load(std::memory_order_relaxed);
  snapshot.processing_failures = benchmark_consumer.processing_failures.load(std::memory_order_relaxed);
  return snapshot;
}

Node::Node(const std::string& node_name, const rclcpp::NodeOptions& options) {
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
  const int publisher_queue_depth =
      node->declare_parameter<int>("publisher_queue_depth", 1);
  if (publisher_queue_depth < 1) {
    throw std::invalid_argument("publisher_queue_depth must be positive");
  }
  const rclcpp::QoS publisher_qos(
      rclcpp::SystemDefaultsQoS().keep_last(
          static_cast<std::size_t>(publisher_queue_depth)).durability_volatile());
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
  publish_fixed_lag_finalized = node->declare_parameter<bool>(
      "publish_fixed_lag_finalized", publish_fixed_lag_finalized);

  publish_local_map = node->declare_parameter<bool>("publish_local_map", publish_local_map);
  if (publish_local_map) {
    map_topic = node->declare_parameter<std::string>("map_topic", map_topic);
    publish_map_after = core::Secondsd(node->declare_parameter<double>("publish_map_after", publish_map_after.count()));
    map_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(map_topic, publisher_qos);
    map_publish_thead = std::jthread([this]() { publish_map_loop(); });
  }

  // lio params
  core::LIO::Config lio_config{};
  lio_config.deskew = node->declare_parameter<bool>("deskew", lio_config.deskew);
  lio_config.piecewise_gyro_deskew =
      node->declare_parameter<bool>("piecewise_gyro_deskew", lio_config.piecewise_gyro_deskew);
  lio_config.fixed_lag_multiscan =
      node->declare_parameter<bool>("fixed_lag_multiscan", lio_config.fixed_lag_multiscan);
  lio_config.fixed_lag_window_size = static_cast<size_t>(std::max<int64_t>(
      2, node->declare_parameter<int>(
             "fixed_lag_window_size",
             static_cast<int>(lio_config.fixed_lag_window_size))));
  lio_config.fixed_lag_neighbor_scans = static_cast<size_t>(std::max<int64_t>(
      1, node->declare_parameter<int>(
             "fixed_lag_neighbor_scans",
             static_cast<int>(lio_config.fixed_lag_neighbor_scans))));
  lio_config.fixed_lag_pairwise_max_iterations = static_cast<size_t>(std::max<int64_t>(
      1, node->declare_parameter<int>(
             "fixed_lag_pairwise_max_iterations",
             static_cast<int>(lio_config.fixed_lag_pairwise_max_iterations))));
  lio_config.fixed_lag_scan_constraint_weight = node->declare_parameter<double>(
      "fixed_lag_scan_constraint_weight",
      lio_config.fixed_lag_scan_constraint_weight);
  lio_config.fixed_lag_odometry_prior_weight = node->declare_parameter<double>(
      "fixed_lag_odometry_prior_weight",
      lio_config.fixed_lag_odometry_prior_weight);
  lio_config.fixed_lag_pairwise_max_translation_m = node->declare_parameter<double>(
      "fixed_lag_pairwise_max_translation_m",
      lio_config.fixed_lag_pairwise_max_translation_m);
  lio_config.fixed_lag_pairwise_max_rotation_deg = node->declare_parameter<double>(
      "fixed_lag_pairwise_max_rotation_deg",
      lio_config.fixed_lag_pairwise_max_rotation_deg);
  lio_config.fixed_lag_pairwise_min_correspondences = static_cast<size_t>(std::max<int64_t>(
      1, node->declare_parameter<int>(
             "fixed_lag_pairwise_min_correspondences",
             static_cast<int>(lio_config.fixed_lag_pairwise_min_correspondences))));
  lio_config.fixed_lag_pairwise_min_inlier_ratio = node->declare_parameter<double>(
      "fixed_lag_pairwise_min_inlier_ratio",
      lio_config.fixed_lag_pairwise_min_inlier_ratio);
  lio_config.fixed_lag_pairwise_min_error_reduction = node->declare_parameter<double>(
      "fixed_lag_pairwise_min_error_reduction",
      lio_config.fixed_lag_pairwise_min_error_reduction);
  lio_config.fixed_lag_max_pose_correction_m = node->declare_parameter<double>(
      "fixed_lag_max_pose_correction_m",
      lio_config.fixed_lag_max_pose_correction_m);
  lio_config.fixed_lag_max_pose_correction_deg = node->declare_parameter<double>(
      "fixed_lag_max_pose_correction_deg",
      lio_config.fixed_lag_max_pose_correction_deg);
  lio_config.fixed_lag_max_latest_pose_correction_m = node->declare_parameter<double>(
      "fixed_lag_max_latest_pose_correction_m",
      lio_config.fixed_lag_max_latest_pose_correction_m);
  lio_config.fixed_lag_max_latest_pose_correction_deg = node->declare_parameter<double>(
      "fixed_lag_max_latest_pose_correction_deg",
      lio_config.fixed_lag_max_latest_pose_correction_deg);
  lio_config.fixed_lag_fix_latest_pose = node->declare_parameter<bool>(
      "fixed_lag_fix_latest_pose", lio_config.fixed_lag_fix_latest_pose);
  lio_config.fixed_lag_huber_delta_m = node->declare_parameter<double>(
      "fixed_lag_huber_delta_m", lio_config.fixed_lag_huber_delta_m);
  lio_config.fixed_lag_map_max_error_ratio = node->declare_parameter<double>(
      "fixed_lag_map_max_error_ratio", lio_config.fixed_lag_map_max_error_ratio);
  lio_config.fixed_lag_map_min_correspondence_ratio = node->declare_parameter<double>(
      "fixed_lag_map_min_correspondence_ratio",
      lio_config.fixed_lag_map_min_correspondence_ratio);
  lio_config.max_iterations =
      static_cast<size_t>(node->declare_parameter<int>("max_iterations", static_cast<int>(lio_config.max_iterations)));
  lio_config.voxel_size = node->declare_parameter<double>("voxel_size", lio_config.voxel_size);
  lio_config.max_points_per_voxel =
      static_cast<int>(node->declare_parameter<int>("max_points_per_voxel", lio_config.max_points_per_voxel));
  lio_config.max_range = node->declare_parameter<double>("max_range", lio_config.max_range);
  lio_config.min_range = node->declare_parameter<double>("min_range", lio_config.min_range);
  lio_config.convergence_criterion =
      node->declare_parameter<double>("convergence_criterion", lio_config.convergence_criterion);
  lio_config.max_correspondance_distance =
      node->declare_parameter<double>("max_correspondance_distance", lio_config.max_correspondance_distance);
  lio_config.max_num_threads =
      static_cast<int>(node->declare_parameter<int>("max_num_threads", lio_config.max_num_threads));
  lio_config.initialization_phase =
      node->declare_parameter<bool>("initialization_phase", lio_config.initialization_phase);
  lio_config.max_expected_jerk = node->declare_parameter<double>("max_expected_jerk", lio_config.max_expected_jerk);
  lio_config.double_downsample = node->declare_parameter<bool>("double_downsample", lio_config.double_downsample);
  lio_config.min_beta = node->declare_parameter<double>("min_beta", lio_config.min_beta);
  lio_config.icp_keypoint_voxel_multiplier = node->declare_parameter<double>(
      "icp_keypoint_voxel_multiplier", lio_config.icp_keypoint_voxel_multiplier);
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
  lio_config.degeneracy_persistence_min_scans = static_cast<size_t>(
      degeneracy_persistence_min_scans > 0 ? degeneracy_persistence_min_scans : 1);
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
  lio_config.degeneracy_adaptive_max_iterations = static_cast<size_t>(
      degeneracy_adaptive_max_iterations > 0 ? degeneracy_adaptive_max_iterations : 1);
  lio_config.degeneracy_adaptive_iteration_ratio = node->declare_parameter<double>(
      "degeneracy_adaptive_iteration_ratio", lio_config.degeneracy_adaptive_iteration_ratio);
  const auto degeneracy_adaptive_hold_scans = node->declare_parameter<int>(
      "degeneracy_adaptive_hold_scans", static_cast<int>(lio_config.degeneracy_adaptive_hold_scans));
  lio_config.degeneracy_adaptive_hold_scans = static_cast<size_t>(
      degeneracy_adaptive_hold_scans > 0 ? degeneracy_adaptive_hold_scans : 1);
  lio_config.degeneracy_multiscan_observability_gate = node->declare_parameter<bool>(
      "degeneracy_multiscan_observability_gate", lio_config.degeneracy_multiscan_observability_gate);
  const auto degeneracy_observability_window_scans = node->declare_parameter<int>(
      "degeneracy_observability_window_scans", static_cast<int>(lio_config.degeneracy_observability_window_scans));
  lio_config.degeneracy_observability_window_scans = static_cast<size_t>(
      degeneracy_observability_window_scans > 0 ? degeneracy_observability_window_scans : 1);
  const auto degeneracy_observability_min_scans = node->declare_parameter<int>(
      "degeneracy_observability_min_scans", static_cast<int>(lio_config.degeneracy_observability_min_scans));
  lio_config.degeneracy_observability_min_scans = static_cast<size_t>(
      degeneracy_observability_min_scans > 0 ? degeneracy_observability_min_scans : 1);
  lio_config.degeneracy_observability_max_directional_ratio = node->declare_parameter<double>(
      "degeneracy_observability_max_directional_ratio",
      lio_config.degeneracy_observability_max_directional_ratio);
  lio_config.visual_fusion.enabled = node->declare_parameter<bool>(
      "selective_visual_fusion", lio_config.visual_fusion.enabled);
  const auto visual_min_tracks = node->declare_parameter<int>(
      "visual_min_tracks", static_cast<int>(lio_config.visual_fusion.min_tracks));
  lio_config.visual_fusion.min_tracks = static_cast<std::size_t>(
      visual_min_tracks > 0 ? visual_min_tracks : 1);
  const auto visual_min_inliers = node->declare_parameter<int>(
      "visual_min_inliers", static_cast<int>(lio_config.visual_fusion.min_inliers));
  lio_config.visual_fusion.min_inliers = static_cast<std::size_t>(
      visual_min_inliers > 0 ? visual_min_inliers : 1);
  lio_config.visual_fusion.min_inlier_ratio = node->declare_parameter<double>(
      "visual_min_inlier_ratio", lio_config.visual_fusion.min_inlier_ratio);
  lio_config.visual_fusion.max_rotation_error_deg = node->declare_parameter<double>(
      "visual_max_rotation_error_deg", lio_config.visual_fusion.max_rotation_error_deg);
  lio_config.visual_fusion.min_translation_cosine = node->declare_parameter<double>(
      "visual_min_translation_cosine", lio_config.visual_fusion.min_translation_cosine);
  lio_config.visual_fusion.min_baseline_m = node->declare_parameter<double>(
      "visual_min_baseline_m", lio_config.visual_fusion.min_baseline_m);
  lio_config.visual_fusion.max_baseline_m = node->declare_parameter<double>(
      "visual_max_baseline_m", lio_config.visual_fusion.max_baseline_m);
  lio_config.visual_fusion.weak_information_ratio = node->declare_parameter<double>(
      "visual_weak_information_ratio", lio_config.visual_fusion.weak_information_ratio);
  lio_config.visual_fusion.relative_information_weight = node->declare_parameter<double>(
      "visual_relative_information_weight", lio_config.visual_fusion.relative_information_weight);
  lio_config.visual_fusion.min_visual_directional_information_ratio =
      node->declare_parameter<double>(
          "visual_min_directional_information_ratio",
          lio_config.visual_fusion.min_visual_directional_information_ratio);
  const auto visual_max_weak_directions = node->declare_parameter<int>(
      "visual_max_weak_directions",
      static_cast<int>(lio_config.visual_fusion.max_weak_directions));
  lio_config.visual_fusion.max_weak_directions = static_cast<std::size_t>(
      visual_max_weak_directions > 0 ? visual_max_weak_directions : 0);
  lio_config.visual_fusion.max_translation_update_m = node->declare_parameter<double>(
      "visual_max_translation_update_m", lio_config.visual_fusion.max_translation_update_m);
  lio_config.visual_fusion.max_rotation_update_rad = node->declare_parameter<double>(
      "visual_max_rotation_update_rad", lio_config.visual_fusion.max_rotation_update_rad);
  lio_config.visual_prior_max_time_offset_sec = node->declare_parameter<double>(
      "visual_prior_max_time_offset_sec", lio_config.visual_prior_max_time_offset_sec);
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
  lio_config.relocalization_min_correspondences =
      node->declare_parameter<int>("relocalization_min_correspondences", lio_config.relocalization_min_correspondences);
  lio_config.relocalization_min_inlier_ratio =
      node->declare_parameter<double>("relocalization_min_inlier_ratio", lio_config.relocalization_min_inlier_ratio);
  lio_config.relocalization_max_mean_error =
      node->declare_parameter<double>("relocalization_max_mean_error", lio_config.relocalization_max_mean_error);
  lio_config.relocalization_max_correspondance_distance = node->declare_parameter<double>(
      "relocalization_max_correspondance_distance", lio_config.relocalization_max_correspondance_distance);
  lio_config.relocalization_yaw_samples =
      node->declare_parameter<int>("relocalization_yaw_samples", lio_config.relocalization_yaw_samples);
  lio_config.relocalization_pose_stride =
      node->declare_parameter<int>("relocalization_pose_stride", lio_config.relocalization_pose_stride);
  lio_config.relocalization_min_pose_separation =
      node->declare_parameter<int>("relocalization_min_pose_separation", lio_config.relocalization_min_pose_separation);
  lio_config.relocalization_max_iterations =
      node->declare_parameter<int>("relocalization_max_iterations", lio_config.relocalization_max_iterations);
  lio = std::make_unique<core::LIO>(lio_config);
  if (publish_fixed_lag_finalized && !lio_config.fixed_lag_multiscan) {
    throw std::invalid_argument(
        "publish_fixed_lag_finalized requires fixed_lag_multiscan");
  }
  direct_visual_frontend = node->declare_parameter<bool>(
      "direct_visual_frontend", direct_visual_frontend);
  direct_visual_require_previous_weak_direction =
      node->declare_parameter<bool>(
          "direct_visual_require_previous_weak_direction",
          direct_visual_require_previous_weak_direction);

  // Timestamp processing params - lts for lidar time stamps, without having 100 char param names
  timestamp_proc_config.multiplier_to_seconds =
      node->declare_parameter<double>("lts_multiplier_to_seconds", timestamp_proc_config.multiplier_to_seconds);
  timestamp_proc_config.force_absolute =
      node->declare_parameter<bool>("lts_force_absolute", timestamp_proc_config.force_absolute);
  timestamp_proc_config.force_relative =
      node->declare_parameter<bool>("lts_force_relative", timestamp_proc_config.force_relative);

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

  registration_thread = std::jthread([this]() { registration_loop(); });

  RCLCPP_INFO(node->get_logger(), "RKO LIO Node is up!");
}

void Node::parse_cli_extrinsics() {
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
  extrinsics_set = imu_ok && lidar_ok;
}

void Node::load_visual_constraints() {
  const std::string path = node->declare_parameter<std::string>("visual_constraints_path", "");
  if (path.empty()) {
    throw std::runtime_error(
        "selective_visual_fusion requires visual_constraints_path during artifact-driven development");
  }
  if (!visual_extrinsic_set) {
    throw std::runtime_error(
        "selective_visual_fusion requires extrinsic_cam2base_quat_xyzw_xyz");
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
    constraint.first_time = core::Secondsd(record.at("first_stamp").get<double>());
    constraint.second_time = core::Secondsd(record.at("second_stamp").get<double>());
    const auto rotation = record.at("rotation").get<std::vector<std::vector<double>>>();
    const auto direction = record.at("translation_direction").get<std::vector<double>>();
    if (rotation.size() != 3 || rotation[0].size() != 3 ||
        rotation[1].size() != 3 || rotation[2].size() != 3 ||
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
            [](const auto& lhs, const auto& rhs) {
              return lhs.second_time < rhs.second_time;
            });
  RCLCPP_INFO_STREAM(node->get_logger(), "Loaded " << visual_constraints.size()
                     << " accepted visual constraints from " << path);
}

void Node::configure_direct_visual_frontend() {
  visual_image_topic = node->declare_parameter<std::string>(
      "visual_image_topic", visual_image_topic);
  visual_camera_time_offset_sec = node->declare_parameter<double>(
      "visual_camera_time_offset_sec", visual_camera_time_offset_sec);
  visual_max_image_lidar_delta_sec = node->declare_parameter<double>(
      "visual_max_image_lidar_delta_sec", visual_max_image_lidar_delta_sec);
  visual_keyframe_interval_sec = node->declare_parameter<double>(
      "visual_keyframe_interval_sec", visual_keyframe_interval_sec);
  visual_camera.width = node->declare_parameter<int>("visual_camera_width", 0);
  visual_camera.height = node->declare_parameter<int>("visual_camera_height", 0);
  visual_camera.fx = node->declare_parameter<double>("visual_camera_fx", 0.0);
  visual_camera.fy = node->declare_parameter<double>("visual_camera_fy", 0.0);
  visual_camera.cx = node->declare_parameter<double>("visual_camera_cx", 0.0);
  visual_camera.cy = node->declare_parameter<double>("visual_camera_cy", 0.0);
  const auto distortion = node->declare_parameter<std::vector<double>>(
      "visual_camera_distortion", std::vector<double>{});
  if (distortion.size() != 4 || !visual_camera.valid()) {
    throw std::runtime_error(
        "direct_visual_frontend requires valid fisheye intrinsics and four distortion coefficients");
  }
  std::copy(distortion.begin(), distortion.end(), visual_camera.distortion.begin());
  direct_visual_config.max_features = node->declare_parameter<int>(
      "direct_visual_max_features", direct_visual_config.max_features);
  direct_visual_config.grid_cell_size = node->declare_parameter<int>(
      "direct_visual_grid_cell_size", direct_visual_config.grid_cell_size);
  direct_visual_config.patch_radius = node->declare_parameter<int>(
      "direct_visual_patch_radius", direct_visual_config.patch_radius);
  direct_visual_config.max_iterations = node->declare_parameter<int>(
      "direct_visual_max_iterations", direct_visual_config.max_iterations);
  direct_visual_config.min_gradient = node->declare_parameter<double>(
      "direct_visual_min_gradient", direct_visual_config.min_gradient);
  direct_visual_config.huber_delta = node->declare_parameter<double>(
      "direct_visual_huber_delta", direct_visual_config.huber_delta);
  direct_visual_config.occlusion_tolerance_m = node->declare_parameter<double>(
      "direct_visual_occlusion_tolerance_m",
      direct_visual_config.occlusion_tolerance_m);
  direct_visual_config.min_residuals = node->declare_parameter<int>(
      "direct_visual_min_residuals", direct_visual_config.min_residuals);
  direct_visual_config.min_inlier_ratio = node->declare_parameter<double>(
      "direct_visual_min_inlier_ratio", direct_visual_config.min_inlier_ratio);
  direct_visual_config.max_rmse = node->declare_parameter<double>(
      "direct_visual_max_rmse", direct_visual_config.max_rmse);
  direct_visual_config.max_initial_rmse = node->declare_parameter<double>(
      "direct_visual_max_initial_rmse", direct_visual_config.max_initial_rmse);
  direct_visual_config.max_translation_correction_m = node->declare_parameter<double>(
      "direct_visual_max_translation_correction_m",
      direct_visual_config.max_translation_correction_m);
  direct_visual_config.max_rotation_correction_rad = node->declare_parameter<double>(
      "direct_visual_max_rotation_correction_rad",
      direct_visual_config.max_rotation_correction_rad);
  RCLCPP_INFO_STREAM(node->get_logger(),
                     "Live direct visual frontend: " << visual_image_topic
                     << " " << visual_camera.width << "x" << visual_camera.height
                     << " keyframe interval " << visual_keyframe_interval_sec << " s");
}

void Node::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg) {
  if (!direct_visual_frontend) {
    return;
  }
  if (image_msg->encoding != "mono8" ||
      static_cast<int>(image_msg->width) != visual_camera.width ||
      static_cast<int>(image_msg->height) != visual_camera.height ||
      image_msg->step < image_msg->width) {
    RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                            "Direct visual frontend requires mono8 images matching configured dimensions");
    return;
  }
  VisualImageFrame frame;
  frame.time = utils::ros_time_to_seconds(image_msg->header.stamp) +
               core::Secondsd(visual_camera_time_offset_sec);
  frame.image.width = static_cast<int>(image_msg->width);
  frame.image.height = static_cast<int>(image_msg->height);
  frame.image.pixels.resize(static_cast<std::size_t>(frame.image.width * frame.image.height));
  for (int row = 0; row < frame.image.height; ++row) {
    std::copy_n(image_msg->data.begin() + static_cast<std::ptrdiff_t>(row * image_msg->step),
                frame.image.width,
                frame.image.pixels.begin() + static_cast<std::ptrdiff_t>(row * frame.image.width));
  }
  std::lock_guard lock(buffer_mutex);
  visual_image_buffer.push(std::move(frame));
  while (visual_image_buffer.size() > 400) {
    visual_image_buffer.pop();
  }
}

std::optional<LiveVisualKeyframe> Node::prepare_direct_visual_prior(
    const core::Vector3dVector& scan, const core::Secondsd& lidar_time) {
  if (!direct_visual_frontend) {
    return std::nullopt;
  }
  std::optional<VisualImageFrame> selected;
  {
    std::lock_guard lock(buffer_mutex);
    while (!visual_image_buffer.empty() &&
           visual_image_buffer.front().time <= lidar_time) {
      selected = std::move(visual_image_buffer.front());
      visual_image_buffer.pop();
    }
  }
  if (!selected.has_value() ||
      (lidar_time - selected->time).count() > visual_max_image_lidar_delta_sec ||
      (live_visual_keyframe.has_value() &&
       (selected->time - live_visual_keyframe->time).count() <
           visual_keyframe_interval_sec)) {
    return std::nullopt;
  }
  LiveVisualKeyframe pending;
  pending.time = selected->time;
  pending.image = std::move(selected->image);
  const Sophus::SE3d camera_T_lidar =
      extrinsic_cam2base.inverse() * extrinsic_lidar2base;
  pending.depth = core::project_sparse_depth(scan, camera_T_lidar, visual_camera);
  if (!live_visual_keyframe.has_value()) {
    return pending;
  }
  if (direct_visual_require_previous_weak_direction &&
      (!lio->lidar_state.icp_diagnostics.has_value() ||
       core::count_weak_information_directions(
           lio->lidar_state.icp_diagnostics->H,
           lio->config.visual_fusion) == 0)) {
    ++direct_visual_weak_gate_skip_count;
    return pending;
  }

  const Sophus::SE3d predicted_image_base =
      lio->predict_pose_at(pending.time);
  const Sophus::SE3d predicted_lidar_base =
      lio->predict_pose_at(lidar_time);
  const Sophus::SE3d previous_camera =
      live_visual_keyframe->world_T_base * extrinsic_cam2base;
  const Sophus::SE3d predicted_camera =
      predicted_image_base * extrinsic_cam2base;
  const Sophus::SE3d initial_relative =
      predicted_camera.inverse() * previous_camera;
  ++direct_visual_attempt_count;
  const auto direct = core::align_direct_visual(
      live_visual_keyframe->image, live_visual_keyframe->depth,
      pending.image, pending.depth, visual_camera, initial_relative,
      direct_visual_config);
  if (direct_visual_attempt_count % 50 == 0) {
    RCLCPP_INFO_STREAM(node->get_logger(),
                       "Direct visual probe " << direct_visual_attempt_count
                       << ": valid=" << direct.valid
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
      direct.current_T_previous_camera.rotationMatrix() *
      initial_relative.rotationMatrix().transpose();
  const double rotation_cosine = std::clamp(
      (rotation_difference.trace() - 1.0) * 0.5, -1.0, 1.0);
  const double rotation_error_deg =
      std::acos(rotation_cosine) * 180.0 / std::numbers::pi;
  const Eigen::Vector3d predicted_translation = initial_relative.translation();
  const Eigen::Vector3d measured_translation =
      direct.current_T_previous_camera.translation();
  const double predicted_norm = predicted_translation.norm();
  const double measured_norm = measured_translation.norm();
  const double translation_cosine =
      predicted_norm > 1.0e-9 && measured_norm > 1.0e-9
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
  const Eigen::Matrix6d camera_adjoint_inverse =
      predicted_camera.Adj().inverse();
  confidence.visual_information =
      camera_adjoint_inverse.transpose() * direct.pose_information *
      camera_adjoint_inverse;
  diagnostics.confidence_gate_passed = core::visual_constraint_passes_gate(
      confidence, lio->config.visual_fusion);
  direct_visual_diagnostics.push_back(diagnostics);
  if (!diagnostics.confidence_gate_passed) {
    return pending;
  }
  const Sophus::SE3d desired_image_camera =
      previous_camera * direct.current_T_previous_camera.inverse();
  const Sophus::SE3d desired_image_base =
      desired_image_camera * extrinsic_cam2base.inverse();
  const Sophus::SE3d image_T_lidar_prediction =
      predicted_image_base.inverse() * predicted_lidar_base;
  const Sophus::SE3d desired_lidar_base =
      desired_image_base * image_T_lidar_prediction;
  lio->set_visual_pose_prior({lidar_time, desired_lidar_base, confidence});
  ++direct_visual_valid_count;
  return pending;
}

void Node::commit_direct_visual_keyframe(
    LiveVisualKeyframe frame, const core::Secondsd& lidar_time,
    const core::Vector3dVector& deskewed_lidar_frame) {
  const double dt = (frame.time - lidar_time).count();
  Eigen::Vector6d correction = Eigen::Vector6d::Zero();
  correction.head<3>() = lio->lidar_state.velocity * dt;
  correction.tail<3>() = lio->lidar_state.angular_velocity * dt;
  frame.world_T_base = lio->lidar_state.pose * Sophus::SE3d::exp(correction);
  const Sophus::SE3d camera_T_lidar =
      extrinsic_cam2base.inverse() * extrinsic_lidar2base;
  frame.depth = core::project_sparse_depth(
      deskewed_lidar_frame, camera_T_lidar, visual_camera);
  live_visual_keyframe = std::move(frame);
}

void Node::prepare_visual_prior(const core::Secondsd& lidar_time) {
  if (!lio->config.visual_fusion.enabled || visual_constraints.empty() ||
      lio->poses_with_timestamps.empty()) {
    return;
  }
  const double tolerance = lio->config.visual_prior_max_time_offset_sec;
  while (next_visual_constraint < visual_constraints.size() &&
         visual_constraints[next_visual_constraint].second_time.count() <
             lidar_time.count() - tolerance) {
    ++next_visual_constraint;
  }
  if (next_visual_constraint >= visual_constraints.size()) {
    return;
  }
  const auto& constraint = visual_constraints[next_visual_constraint];
  if (std::abs((constraint.second_time - lidar_time).count()) > tolerance) {
    return;
  }

  const auto nearest = std::min_element(
      lio->poses_with_timestamps.begin(), lio->poses_with_timestamps.end(),
      [&constraint](const auto& lhs, const auto& rhs) {
        return std::abs((lhs.first - constraint.first_time).count()) <
               std::abs((rhs.first - constraint.first_time).count());
      });
  if (nearest == lio->poses_with_timestamps.end() ||
      std::abs((nearest->first - constraint.first_time).count()) > tolerance) {
    ++next_visual_constraint;
    return;
  }
  const double prediction_dt = (constraint.second_time - lio->lidar_state.time).count();
  Eigen::Vector6d prediction = Eigen::Vector6d::Zero();
  prediction.head<3>() = lio->lidar_state.velocity * prediction_dt;
  prediction.tail<3>() = lio->lidar_state.angular_velocity * prediction_dt;
  const Sophus::SE3d predicted_current_pose =
      lio->lidar_state.pose * Sophus::SE3d::exp(prediction);
  const Sophus::SE3d desired_pose = desired_base_pose_from_monocular_relative(
      nearest->second, predicted_current_pose, extrinsic_cam2base,
      constraint.current_R_previous_camera,
      constraint.current_t_previous_direction);
  auto confidence = constraint.confidence;
  confidence.baseline_m =
      ((predicted_current_pose * extrinsic_cam2base).inverse() *
       (nearest->second * extrinsic_cam2base)).translation().norm();
  lio->set_visual_pose_prior(
      {constraint.second_time, desired_pose, confidence});
  ++next_visual_constraint;
}

bool Node::check_and_set_extrinsics() {
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

void Node::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg) {
  if (imu_frame.empty()) {
    if (imu_msg->header.frame_id.empty() && !extrinsics_set) {
      throw std::runtime_error("IMU message header has no frame id and we need it to query TF for the extrinsics. "
                               "Either specify the frame id or the extrinsic manually.");
    }
    imu_frame = imu_msg->header.frame_id;
    RCLCPP_INFO_STREAM(node->get_logger(), "Parsed the imu frame id as: " << imu_frame);
  }
  if (!check_and_set_extrinsics()) {
    // we assume that extrinsics are static. if they change, its better to query the tf directly in the registration
    // loop for each message being processed asynchronously.
    benchmark_consumer.record_drop(false);
    return;
  }
  {
    std::lock_guard lock(buffer_mutex);
    imu_buffer.emplace(imu_msg_to_imu_data(*imu_msg));
    atomic_can_process = !lidar_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().timestamps.max;
  }
  if (atomic_can_process) {
    sync_condition_variable.notify_one();
  }
}

void Node::lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) {
  if (lidar_frame.empty()) {
    if (lidar_msg->header.frame_id.empty() && !extrinsics_set) {
      throw std::runtime_error("LiDAR message header has no frame id and we need it to query TF for the extrinsics. "
                               "Either specify the frame id or the extrinsic manually.");
    }
    lidar_frame = lidar_msg->header.frame_id;
    RCLCPP_INFO_STREAM(node->get_logger(), "Parsed the lidar frame id as: " << lidar_frame);
  }
  if (!check_and_set_extrinsics()) {
    benchmark_consumer.record_drop(false);
    return;
  }
  {
    std::lock_guard lock(buffer_mutex);
    if (lidar_buffer.size() >= max_lidar_buffer_size) {
      RCLCPP_WARN_STREAM(node->get_logger(), "Registration lidar buffer limit reached. Dropping frame.");
      benchmark_consumer.record_drop(true);
      sync_condition_variable.notify_one();
      return;
    }
  }
  try {
    const auto& [timestamps, scan] = std::invoke([&]() -> std::tuple<core::Timestamps, core::Vector3dVector> {
      const core::Secondsd& header_stamp = utils::ros_time_to_seconds(lidar_msg->header.stamp);
      if (lio->config.deskew) {
        const auto& [scan, raw_timestamps] = utils::point_cloud2_to_eigen_with_timestamps(lidar_msg);
        const core::Timestamps& timestamps =
            core::process_timestamps(raw_timestamps, header_stamp, timestamp_proc_config);
        return {timestamps, scan};
      } else {
        RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                                "Deskewing is disabled. Populating timestamps with static header time.");
        const core::Vector3dVector scan = utils::point_cloud2_to_eigen(lidar_msg);
        return {{.min = header_stamp, .max = header_stamp, .times = core::TimestampVector(scan.size(), header_stamp)},
                scan};
      }
    });

    {
      std::lock_guard lock(buffer_mutex);
      lidar_buffer.emplace(timestamps, scan);
      benchmark_consumer.observe_registration_queue(lidar_buffer.size());
      atomic_can_process = !imu_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().timestamps.max;
    }
    if (atomic_can_process) {
      sync_condition_variable.notify_one();
    }
  } catch (const std::invalid_argument& ex) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Encountered error, dropping frame: Error. " << ex.what());
    benchmark_consumer.record_drop(false);
    benchmark_consumer.set_failure_reason(ex.what());
  }
}

void Node::registration_loop() {
  while (rclcpp::ok() && atomic_node_running) {
    SCOPED_PROFILER("ROS Registration Loop");
    std::unique_lock buffer_lock(buffer_mutex);
    sync_condition_variable.wait(buffer_lock, [this]() { return !atomic_node_running || atomic_can_process; });
    if (!atomic_node_running) {
      // node could have been killed after waiting on the cv
      break;
    }
    core::LidarFrame frame = std::move(lidar_buffer.front());
    lidar_buffer.pop();
    atomic_registration_active = true;
    const auto& [timestamps, scan] = frame;
    const auto& [start_stamp, end_stamp, time_vector] = timestamps;
    for (; !imu_buffer.empty() && imu_buffer.front().time < end_stamp; imu_buffer.pop()) {
      const core::ImuControl& imu_data = imu_buffer.front();
      lio->add_imu_measurement(extrinsic_imu2base, imu_data);
    }
    // check if there are more messages buffered already
    atomic_can_process =
        !imu_buffer.empty() && !lidar_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().timestamps.max;
    buffer_lock.unlock(); // we dont touch the buffers anymore

    try {
      std::optional<LiveVisualKeyframe> direct_visual_frame;
      if (direct_visual_frontend) {
        direct_visual_frame = prepare_direct_visual_prior(scan, end_stamp);
      } else {
        prepare_visual_prior(end_stamp);
      }
      const core::Vector3dVector deskewed_frame = std::invoke([&]() {
        if (publish_local_map) {
          std::lock_guard lock(local_map_mutex); // publish_map thread might access simultaneously
          return lio->register_scan(extrinsic_lidar2base, scan, time_vector);
        } else {
          return lio->register_scan(extrinsic_lidar2base, scan, time_vector);
        }
      });

      if (!deskewed_frame.empty()) {
        publish_or_queue_registered_output(deskewed_frame, end_stamp);
        if (publish_lidar_acceleration) {
          publish_lidar_accel(lio->lidar_state.linear_acceleration, end_stamp);
        }
        if (direct_visual_frame.has_value()) {
          commit_direct_visual_keyframe(
              std::move(*direct_visual_frame), end_stamp, deskewed_frame);
        }
      }
    } catch (const std::exception& ex) {
      // Catch both std::invalid_argument (Keypoints=0 / Δt) and std::runtime_error
      // (Number of correspondences=0). Both are recoverable on kidnap-style bags.
      RCLCPP_ERROR_STREAM(node->get_logger(), "Encountered error, dropping frame. Error: " << ex.what());
      benchmark_consumer.record_processing_failure();
      benchmark_consumer.set_failure_reason(ex.what());
    }
    atomic_registration_active = false;
  }
  atomic_registration_active = false;
  atomic_node_running = false;
}

void Node::publish_or_queue_registered_output(
    const core::Vector3dVector& deskewed_frame,
    const core::Secondsd& stamp) {
  if (publish_fixed_lag_finalized) {
    pending_fixed_lag_outputs.push_back(
        {stamp, lio->lidar_state, deskewed_frame});
    publish_ready_fixed_lag_outputs();
    return;
  }
  if (publish_deskewed_scan) {
    std_msgs::msg::Header header;
    header.frame_id = lidar_frame;
    header.stamp = rclcpp::Time(
        std::chrono::duration_cast<std::chrono::nanoseconds>(stamp).count());
    frame_publisher->publish(utils::eigen_to_point_cloud2(deskewed_frame, header));
  }
  publish_odometry(lio->lidar_state, stamp);
}

void Node::publish_ready_fixed_lag_outputs() {
  constexpr double timestamp_tolerance_sec = 1.0e-6;
  for (const core::LIO::FinalizedFixedLagPose& finalized :
       lio->take_finalized_fixed_lag_poses()) {
    while (!pending_fixed_lag_outputs.empty() &&
           pending_fixed_lag_outputs.front().time.count() <
               finalized.time.count() - timestamp_tolerance_sec) {
      publish_fixed_lag_output(
          std::move(pending_fixed_lag_outputs.front()));
      pending_fixed_lag_outputs.pop_front();
    }
    if (pending_fixed_lag_outputs.empty() ||
        std::abs(pending_fixed_lag_outputs.front().time.count() -
                 finalized.time.count()) > timestamp_tolerance_sec) {
      RCLCPP_WARN_STREAM(
          node->get_logger(),
          "No buffered deskewed frame for finalized fixed-lag pose at "
              << finalized.time.count() << "s");
      continue;
    }
    PendingFixedLagOutput output =
        std::move(pending_fixed_lag_outputs.front());
    pending_fixed_lag_outputs.pop_front();
    publish_fixed_lag_output(std::move(output), finalized.pose);
  }
}

void Node::publish_fixed_lag_output(
    PendingFixedLagOutput output,
    const std::optional<Sophus::SE3d>& finalized_pose) {
  if (finalized_pose.has_value()) {
    output.state.pose = *finalized_pose;
  }
  if (publish_deskewed_scan) {
    std_msgs::msg::Header header;
    header.frame_id = lidar_frame;
    header.stamp = rclcpp::Time(
        std::chrono::duration_cast<std::chrono::nanoseconds>(output.time).count());
    frame_publisher->publish(
        utils::eigen_to_point_cloud2(output.deskewed_frame, header));
  }
  publish_odometry(output.state, output.time);
}

void Node::flush_fixed_lag_outputs() {
  if (!publish_fixed_lag_finalized) {
    return;
  }
  lio->finalize_fixed_lag_window();
  publish_ready_fixed_lag_outputs();
}

void Node::publish_odometry(const core::State& state, const core::Secondsd& stamp) const {
  const std::string_view from_frame = base_frame;
  const std::string_view to_frame = odom_frame;
  // tf message
  geometry_msgs::msg::TransformStamped transform_msg;
  transform_msg.header.stamp = rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(stamp).count());
  if (invert_odom_tf) {
    transform_msg.header.frame_id = from_frame;
    transform_msg.child_frame_id = to_frame;
    transform_msg.transform = utils::sophus_to_transform(state.pose.inverse());
  } else {
    transform_msg.header.frame_id = to_frame;
    transform_msg.child_frame_id = from_frame;
    transform_msg.transform = utils::sophus_to_transform(state.pose);
  }
  tf_broadcaster->sendTransform(transform_msg);

  // odometry msg
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(stamp).count());
  odom_msg.header.frame_id = to_frame;
  odom_msg.child_frame_id = from_frame;
  odom_msg.pose.pose = utils::sophus_to_pose(state.pose);
  // [v0.8 Phase 1, diagnostic-only] anisotropic covariance from the final
  // ICP Hessian; see pose_covariance_from_state's doc comment above. Only
  // this field is new -- pose.pose and twist.twist are set exactly as before.
  odom_msg.pose.covariance = pose_covariance_from_state(state);
  utils::eigen_vector3d_to_ros_xyz(state.velocity, odom_msg.twist.twist.linear);
  utils::eigen_vector3d_to_ros_xyz(state.angular_velocity, odom_msg.twist.twist.angular);
  odom_publisher->publish(odom_msg);
}

void Node::publish_lidar_accel(const Eigen::Vector3d& acceleration, const core::Secondsd& stamp) const {
  auto accel_msg = geometry_msgs::msg::AccelStamped();
  accel_msg.header.stamp = rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(stamp).count());
  accel_msg.header.frame_id = base_frame;
  utils::eigen_vector3d_to_ros_xyz(acceleration, accel_msg.accel.linear);
  lidar_accel_publisher->publish(accel_msg);
}

void Node::publish_map_loop() {
  while (atomic_node_running) {
    std::this_thread::sleep_for(publish_map_after);
    std::unique_lock lock(local_map_mutex);
    if (lio->local_map_empty()) {
      RCLCPP_WARN_ONCE(node->get_logger(), "Local map publish thread: Local map is empty.");
      continue;
    }
    const core::Vector3dVector map_points = lio->local_map_pointcloud();
    lock.unlock(); // we don't access the local map anymore
    std_msgs::msg::Header map_header;
    map_header.stamp = node->now();
    map_header.frame_id = odom_frame;
    map_publisher->publish(utils::eigen_to_point_cloud2(map_points, map_header));
  }
}

Node::~Node() {
  atomic_node_running = false;
  sync_condition_variable.notify_all();
}

void Node::dump_results_to_disk(const std::filesystem::path& results_dir, const std::string& run_name) const {
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
        file << std::fixed << std::setprecision(6) << timestamp.count() << " " << translation.x() << " "
             << translation.y() << " " << translation.z() << " " << quaternion.x() << " " << quaternion.y() << " "
             << quaternion.z() << " " << quaternion.w() << "\n";
      }
      std::cout << "Poses written to " << std::filesystem::absolute(output_file) << "\n";
    }
    if (!lio->tracking_poses_with_timestamps.empty()) {
      const std::filesystem::path tracking_file = output_dir /
          (run_name + "_tracking_" + std::to_string(index) + ".txt");
      if (std::ofstream file(tracking_file); file.is_open()) {
        for (const auto& [timestamp, pose] : lio->tracking_poses_with_timestamps) {
          const Eigen::Vector3d& translation = pose.translation();
          const Eigen::Quaterniond& quaternion = pose.so3().unit_quaternion();
          file << std::fixed << std::setprecision(6) << timestamp.count() << " "
               << translation.x() << " " << translation.y() << " "
               << translation.z() << " " << quaternion.x() << " "
               << quaternion.y() << " " << quaternion.z() << " "
               << quaternion.w() << "\n";
        }
      }
    }
    // dump config
    const nlohmann::json json_config = {{"config", lio->config}};
    const std::filesystem::path config_file = output_dir / "config.json";
    if (std::ofstream file(config_file); file.is_open()) {
      file << json_config.dump(4);
      std::cout << "Configuration written to " << config_file << "\n";
    }
    if (!lio->degeneracy_persistence_diagnostics.empty()) {
      const std::filesystem::path diagnostics_file = output_dir / "degeneracy_persistence.csv";
      if (std::ofstream file(diagnostics_file); file.is_open()) {
        file << "timestamp,candidate_available,consecutive_scans,matched_absolute_cosine,confirmed,"
                "observability_window_scans,aggregate_directional_information_ratio,"
                "multiscan_observability_confirmed,"
                "intervention_count,axis_tx,axis_ty,axis_tz,axis_rx,axis_ry,axis_rz\n";
        for (const auto& sample : lio->degeneracy_persistence_diagnostics) {
          const auto& state = sample.persistent_weak_direction;
          file << std::fixed << std::setprecision(9) << sample.time.count() << ","
               << static_cast<int>(state.candidate_available) << "," << state.consecutive_scans << ","
               << state.matched_absolute_cosine << "," << static_cast<int>(state.confirmed) << ","
               << state.observability_window_scans << ","
               << state.aggregate_directional_information_ratio << ","
               << static_cast<int>(state.multiscan_observability_confirmed) << ","
               << sample.intervention_count;
          for (int axis = 0; axis < 6; ++axis) {
            file << "," << state.axis(axis);
          }
          file << "\n";
        }
        std::cout << "Degeneracy persistence diagnostics written to " << diagnostics_file << "\n";
      }
    }
    if (lio->config.fixed_lag_multiscan) {
      const double applied_pose_count = static_cast<double>(
          std::max<std::size_t>(1U, lio->fixed_lag_applied_pose_count));
      const nlohmann::json fixed_lag_summary = {
          {"pairwise_attempt_count", lio->fixed_lag_pairwise_attempt_count},
          {"pairwise_accept_count", lio->fixed_lag_pairwise_accept_count},
          {"window_attempt_count", lio->fixed_lag_window_attempt_count},
          {"window_accept_count", lio->fixed_lag_window_accept_count},
          {"max_applied_translation_m", lio->fixed_lag_max_applied_translation_m},
          {"max_applied_rotation_deg", lio->fixed_lag_max_applied_rotation_deg},
          {"applied_pose_count", lio->fixed_lag_applied_pose_count},
          {"rms_applied_translation_m", std::sqrt(
              lio->fixed_lag_applied_translation_squared_sum / applied_pose_count)},
          {"rms_applied_rotation_deg", std::sqrt(
              lio->fixed_lag_applied_rotation_deg_squared_sum / applied_pose_count)},
          {"max_latest_translation_m", lio->fixed_lag_max_latest_translation_m},
          {"max_latest_rotation_deg", lio->fixed_lag_max_latest_rotation_deg}};
      const std::filesystem::path fixed_lag_file =
          output_dir / "fixed_lag_summary.json";
      if (std::ofstream file(fixed_lag_file); file.is_open()) {
        file << fixed_lag_summary.dump(4) << "\n";
        std::cout << "Fixed-lag diagnostics written to " << fixed_lag_file << "\n";
      }
    }
    if (lio->config.visual_fusion.enabled) {
      const nlohmann::json visual_summary = {
          {"prior_attempt_count", lio->visual_prior_attempt_count},
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
    if (!lio->visual_observability_diagnostics.empty()) {
      const std::filesystem::path observability_file =
          output_dir / "visual_directional_observability.csv";
      if (std::ofstream file(observability_file); file.is_open()) {
        file << "timestamp,direction_index,directional_information_ratio\n";
        for (const auto& sample : lio->visual_observability_diagnostics) {
          for (std::size_t index = 0; index < sample.ratio_count; ++index) {
            file << std::fixed << std::setprecision(9)
                 << sample.time.count() << "," << index << ","
                 << sample.directional_information_ratios[index] << "\n";
          }
        }
      }
    }
    if (!direct_visual_diagnostics.empty()) {
      const std::filesystem::path diagnostics_file =
          output_dir / "direct_visual_diagnostics.csv";
      if (std::ofstream file(diagnostics_file); file.is_open()) {
        file << "timestamp,solver_valid,confidence_gate_passed,failure_reason,features,tracked_features,inliers,"
                "inlier_ratio,initial_rmse,final_rmse,exposure_gain,exposure_bias,"
                "rotation_error_deg,translation_cosine,baseline_m,predicted_baseline_m\n";
        for (const auto& sample : direct_visual_diagnostics) {
          file << std::fixed << std::setprecision(9) << sample.time.count() << ","
               << static_cast<int>(sample.solver_valid) << ","
               << static_cast<int>(sample.confidence_gate_passed) << ","
               << sample.failure_reason << ","
               << sample.features << "," << sample.tracked_features << ","
               << sample.inliers << "," << sample.inlier_ratio << ","
               << sample.initial_rmse << "," << sample.final_rmse << ","
               << sample.exposure_gain << "," << sample.exposure_bias << ","
               << sample.rotation_error_deg << "," << sample.translation_cosine << ","
               << sample.baseline_m << "," << sample.predicted_baseline_m << "\n";
        }
      }
    }
  } catch (const std::filesystem::filesystem_error& ex) {
    std::cerr << "[WARNING] Cannot write files to disk, encountered filesystem error: " << ex.what() << "\n";
  }
}

} // namespace rko_lio::ros
