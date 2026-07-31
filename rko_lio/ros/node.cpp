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
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LIO::Config,
                                   deskew,
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
  const int publisher_queue_depth = node->declare_parameter<int>("publisher_queue_depth", 1);
  if (publisher_queue_depth < 1) {
    throw std::invalid_argument("publisher_queue_depth must be positive");
  }
  const rclcpp::QoS publisher_qos(
      rclcpp::SystemDefaultsQoS().keep_last(static_cast<std::size_t>(publisher_queue_depth)).durability_volatile());
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
    publish_map_after = core::Secondsd(node->declare_parameter<double>("publish_map_after", publish_map_after.count()));
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
  lio_config.max_correspondance_distance =
      node->declare_parameter<double>("max_correspondance_distance", lio_config.max_correspondance_distance);
  lio_config.max_num_threads =
      static_cast<int>(node->declare_parameter<int>("max_num_threads", lio_config.max_num_threads));
  lio_config.initialization_phase =
      node->declare_parameter<bool>("initialization_phase", lio_config.initialization_phase);
  lio_config.max_expected_jerk = node->declare_parameter<double>("max_expected_jerk", lio_config.max_expected_jerk);
  lio_config.double_downsample = node->declare_parameter<bool>("double_downsample", lio_config.double_downsample);
  lio_config.icp_keypoint_voxel_multiplier = node->declare_parameter<double>(
      "icp_keypoint_voxel_multiplier", lio_config.icp_keypoint_voxel_multiplier);
  lio_config.min_beta = node->declare_parameter<double>("min_beta", lio_config.min_beta);
  lio_config.degeneracy_aware_solve =
      node->declare_parameter<bool>("degeneracy_aware_solve", lio_config.degeneracy_aware_solve);
  lio_config.degeneracy_well_conditioned_ratio = node->declare_parameter<double>(
      "degeneracy_well_conditioned_ratio", lio_config.degeneracy_well_conditioned_ratio);
  lio_config.degeneracy_multiplicity_relative_gap = node->declare_parameter<double>(
      "degeneracy_multiplicity_relative_gap", lio_config.degeneracy_multiplicity_relative_gap);
  lio_config.degeneracy_prior_weight =
      node->declare_parameter<double>("degeneracy_prior_weight", lio_config.degeneracy_prior_weight);
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

  // Timestamp processing params - lts for lidar time stamps, without having 100 char param names
  timestamp_proc_config.multiplier_to_seconds =
      node->declare_parameter<double>("lts_multiplier_to_seconds", timestamp_proc_config.multiplier_to_seconds);
  timestamp_proc_config.force_absolute =
      node->declare_parameter<bool>("lts_force_absolute", timestamp_proc_config.force_absolute);
  timestamp_proc_config.force_relative =
      node->declare_parameter<bool>("lts_force_relative", timestamp_proc_config.force_relative);

  // manually, if, define extrinsics
  parse_cli_extrinsics();

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
  extrinsics_set = imu_ok && lidar_ok;
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
    return;
  }
  {
    std::lock_guard lock(buffer_mutex);
    if (lidar_buffer.size() >= max_lidar_buffer_size) {
      RCLCPP_WARN_STREAM(node->get_logger(), "Registration lidar buffer limit reached. Dropping frame.");
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
      atomic_can_process = !imu_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().timestamps.max;
    }
    if (atomic_can_process) {
      sync_condition_variable.notify_one();
    }
  } catch (const std::invalid_argument& ex) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Encountered error, dropping frame: Error. " << ex.what());
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
      const core::Vector3dVector deskewed_frame = std::invoke([&]() {
        if (publish_local_map) {
          std::lock_guard lock(local_map_mutex); // publish_map thread might access simultaneously
          return lio->register_scan(extrinsic_lidar2base, scan, time_vector);
        } else {
          return lio->register_scan(extrinsic_lidar2base, scan, time_vector);
        }
      });

      if (!deskewed_frame.empty()) {
        // TODO: first frame is skipped and an empty frame is returned. improve how we handle this
        if (publish_deskewed_scan) {
          std_msgs::msg::Header header;
          header.frame_id = lidar_frame;
          header.stamp = rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(end_stamp).count());
          frame_publisher->publish(utils::eigen_to_point_cloud2(deskewed_frame, header));
        }
        publish_odometry(lio->lidar_state, end_stamp);
        if (publish_lidar_acceleration) {
          publish_lidar_accel(lio->lidar_state.linear_acceleration, end_stamp);
        }
      }
    } catch (const std::exception& ex) {
      // Catch both std::invalid_argument (Keypoints=0 / Δt) and std::runtime_error
      // (Number of correspondences=0). Both are recoverable on kidnap-style bags.
      RCLCPP_ERROR_STREAM(node->get_logger(), "Encountered error, dropping frame. Error: " << ex.what());
    }
    atomic_registration_active = false;
  }
  atomic_registration_active = false;
  atomic_node_running = false;
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
    if (lio->map.Empty()) {
      RCLCPP_WARN_ONCE(node->get_logger(), "Local map publish thread: Local map is empty.");
      continue;
    }
    const core::Vector3dVector map_points = lio->map.Pointcloud();
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
    // dump config
    const nlohmann::json json_config = {{"config", lio->config}};
    const std::filesystem::path config_file = output_dir / "config.json";
    if (std::ofstream file(config_file); file.is_open()) {
      file << json_config.dump(4);
      std::cout << "Configuration written to " << config_file << "\n";
    }
  } catch (const std::filesystem::filesystem_error& ex) {
    std::cerr << "[WARNING] Cannot write files to disk, encountered filesystem error: " << ex.what() << "\n";
  }
}

} // namespace rko_lio::ros
