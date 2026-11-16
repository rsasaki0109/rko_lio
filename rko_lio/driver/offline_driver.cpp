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

#include "rko_lio/driver/offline_driver.hpp"
#include "rko_lio/core/imu_acceleration_unit.hpp"
#include "rko_lio/core/voxel_down_sample.hpp"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace rko_lio::driver {

namespace {
// [v0.8 Phase 1, diagnostic-only] anisotropic nav_msgs/Odometry.pose.covariance fill, derived from
// LIO::State::icp_diagnostics (rko_lio/core/util.hpp); moved here from the ROS 2 node. At a
// Gauss-Newton optimum the accumulated (correspondence-averaged) ICP Hessian H is the asymptotic
// information matrix of the solve, so Cov = V * diag(1/lambda) * V^T with H's eigen-decomposition
// (Zhang & Singh, ICRA 2016). Eigenvalues are floored at max(kMinEigenvalueFloor,
// kRelativeEigenvalueFloor * lambda_max) before inverting, so a degenerate direction reports a very
// large but finite variance. A scan without an ICP solve (first frame, dropped scan, kidnap reset)
// reports kNoDiagnosticsFallbackVariance on the diagonal: "no information", not "perfectly known".
constexpr double kMinEigenvalueFloor = 1e-9;
constexpr double kRelativeEigenvalueFloor = 1e-6;
constexpr double kNoDiagnosticsFallbackVariance = 1e6;
} // namespace

std::array<double, 36> pose_covariance_from_state(const core::State& state) {
  std::array<double, 36> covariance{}; // zero-initialized
  if (!state.icp_diagnostics.has_value()) {
    for (int i = 0; i < 6; ++i) {
      covariance[static_cast<size_t>(i * 6 + i)] = kNoDiagnosticsFallbackVariance;
    }
    return covariance;
  }
  const core::LocalizabilitySummary& localizability = state.icp_diagnostics->localizability;
  const double lambda_max = localizability.eigenvalues.maxCoeff();
  const double eigenvalue_floor = std::max(kMinEigenvalueFloor, kRelativeEigenvalueFloor * lambda_max);
  Eigen::Vector6d inv_eigenvalues;
  for (int i = 0; i < 6; ++i) {
    inv_eigenvalues(i) = 1.0 / std::max(localizability.eigenvalues(i), eigenvalue_floor);
  }
  // Sophus::SE3d's se(3) tangent order (translation rho, then rotation phi) matches
  // geometry_msgs/PoseWithCovariance's covariance order ([x, y, z, rot x, rot y, rot z]).
  const Eigen::Matrix6d cov =
      localizability.eigenvectors * inv_eigenvalues.asDiagonal() * localizability.eigenvectors.transpose();
  for (int row = 0; row < 6; ++row) {
    for (int col = 0; col < 6; ++col) {
      covariance[static_cast<size_t>(row * 6 + col)] = cov(row, col);
    }
  }
  return covariance;
}

std::tuple<core::Timestamps, core::Vector3dVector> process_lidar_frame(const PointCloudView& cloud,
                                                                       const core::Nsec header_stamp,
                                                                       const bool deskew,
                                                                       const core::TimestampProcessingConfig& config) {
  if (deskew) {
    const auto& [scan, raw_timestamps] = point_cloud_to_eigen_with_timestamps(cloud);
    const core::Timestamps& timestamps = core::process_timestamps(raw_timestamps, header_stamp, config);
    return {timestamps, scan};
  }
  const core::Vector3dVector scan = point_cloud_to_eigen(cloud);
  const core::Nsec corrected_stamp =
      header_stamp +
      std::chrono::duration_cast<core::Nsec>(std::chrono::duration<double>(config.offset_seconds));
  return {{.min = corrected_stamp, .max = corrected_stamp, .times = core::TimestampVector(scan.size(), corrected_stamp)},
          scan};
}

std::vector<float> lidar_frame_intensities(const PointCloudView& cloud,
                                           const core::LIO::Config& config,
                                           const std::string& photometric_channel,
                                           bool& field_missing) {
  field_missing = false;
  // Only parsed when an intensity-based feature is enabled; the photometric images take precedence
  // over the reflectivity-profile features.
  if (config.photometric) {
    std::vector<float> values = point_cloud_field_as_float(cloud, photometric_channel);
    field_missing = values.empty();
    return values;
  }
  if (!(config.intensity_constraint || config.intensity_disagreement_gate)) {
    return {};
  }
  std::vector<float> intensities = point_cloud_to_intensity(cloud);
  field_missing = intensities.empty();
  return intensities;
}

OfflineDriver::OfflineDriver(DriverConfig config,
                             OutputCallback on_output,
                             Log info,
                             Log warn,
                             const std::size_t max_lidar_buffer_size)
    : config_(std::move(config)),
      on_output_(std::move(on_output)),
      info_(std::move(info)),
      warn_(std::move(warn)),
      max_lidar_buffer_size_(max_lidar_buffer_size) {
  if (max_lidar_buffer_size_ < 1) {
    throw std::invalid_argument("max_lidar_buffer_size must be positive");
  }
  if (!config_.extrinsic_imu2base || !config_.extrinsic_lidar2base) {
    throw std::invalid_argument("extrinsic_imu2base_quat_xyzw_xyz and extrinsic_lidar2base_quat_xyzw_xyz must be set "
                                "(there is no TF tree to look them up without ROS)");
  }
  if (config_.lio.visual_fusion.enabled || config_.direct_visual_frontend) {
    throw std::invalid_argument("visual fusion is not available in the ROS-free driver");
  }
  if (!config_.radar_topic.empty() || config_.lio.radar_velocity_fusion ||
      config_.lio.radar_velocity_continuous_fusion) {
    throw std::invalid_argument("radar fusion is not available in the ROS-free driver");
  }
  extrinsic_imu2base_ = *config_.extrinsic_imu2base;
  extrinsic_lidar2base_ = *config_.extrinsic_lidar2base;
  lio_ = std::make_unique<core::LIO>(config_.lio);
}

void OfflineDriver::add_imu(const core::Nsec time,
                            const Eigen::Vector3d& linear_acceleration,
                            const Eigen::Vector3d& angular_velocity) {
  core::ImuControl imu_data;
  imu_data.time = time;
  imu_data.angular_velocity = angular_velocity;
  imu_data.acceleration = linear_acceleration;
  // BaseNode::imu_data_in_mps2: the unit scale is fixed from the first sample.
  if (!(imu_acceleration_scale_ > 0.0)) {
    imu_acceleration_scale_ = core::acceleration_scale(config_.imu_acceleration_unit, imu_data.acceleration.norm());
    char message[256];
    std::snprintf(message, sizeof(message), "IMU acceleration unit %s: first sample |a| = %.3f, scaling by %.5f",
                  config_.imu_acceleration_unit.c_str(), imu_data.acceleration.norm(), imu_acceleration_scale_);
    info_(message);
  }
  imu_data.acceleration *= imu_acceleration_scale_;
  imu_buffer_.push(imu_data);
  process_ready_frames();
}

void OfflineDriver::add_lidar(const PointCloudView& cloud, const core::Nsec header_stamp) {
  try {
    if (!config_.lio.deskew && !warned_deskew_off_) {
      warn_("Deskewing is disabled. Populating timestamps with static header time.");
      warned_deskew_off_ = true;
    }
    auto [timestamps, scan] = process_lidar_frame(cloud, header_stamp, config_.lio.deskew, config_.timestamp_proc_config);
    bool field_missing = false;
    std::vector<float> intensities =
        lidar_frame_intensities(cloud, config_.lio, config_.photometric_channel, field_missing);
    if (field_missing && !warned_intensity_missing_) {
      warn_(config_.lio.photometric
                ? "photometric is enabled but the point cloud has no '" + config_.photometric_channel +
                      "' field; photometric terms will not be used."
                : std::string("intensity_constraint/intensity_disagreement_gate is enabled but the point cloud has no "
                              "usable 'reflectivity' (uint16) or 'intensity' (float32) field; the intensity "
                              "feature(s) will not fire."));
      warned_intensity_missing_ = true;
    }
    if (lidar_buffer_.size() >= max_lidar_buffer_size_) {
      lidar_buffer_.pop();
      warn_("Registration lidar buffer limit reached. Dropping oldest pending frame.");
    }
    lidar_buffer_.push(LidarFrame{std::move(timestamps), std::move(scan), std::move(intensities)});
  } catch (const std::invalid_argument& ex) {
    warn_(std::string("Encountered error, dropping frame: Error. ") + ex.what());
    return;
  }
  process_ready_frames();
}

bool OfflineDriver::can_process() const {
  return !imu_buffer_.empty() && !lidar_buffer_.empty() && imu_buffer_.back().time > lidar_buffer_.front().timestamps.max;
}

void OfflineDriver::process_ready_frames() {
  // ThreadedNode::registration_loop, run inline: the ROS 2 offline node registers a frame once an
  // IMU sample newer than its last point is buffered, after every older buffered sample.
  while (can_process()) {
    LidarFrame frame = std::move(lidar_buffer_.front());
    lidar_buffer_.pop();
    const auto& [start_stamp, end_stamp, time_vector] = frame.timestamps;
    const std::vector<float>* intensities_ptr = frame.intensities.empty() ? nullptr : &frame.intensities;
    for (; !imu_buffer_.empty() && imu_buffer_.front().time < end_stamp; imu_buffer_.pop()) {
      lio_->add_imu_measurement(extrinsic_imu2base_, imu_buffer_.front());
    }
    try {
      const core::Vector3dVector deskewed_frame =
          lio_->register_scan(extrinsic_lidar2base_, frame.scan, time_vector, intensities_ptr);
      if (deskewed_frame.empty()) {
        continue; // the first frame only initializes
      }
      OdometryOutput output;
      core::State state = lio_->lidar_state;
      if (!lio_->output_poses_with_timestamps.empty()) {
        // output_filter_position_sigma: the filtered position with the registered attitude
        state.pose = lio_->output_poses_with_timestamps.back().second;
      }
      output.time = state.time;
      output.pose = state.pose;
      output.velocity = state.velocity;
      output.angular_velocity = state.angular_velocity;
      output.pose_covariance = pose_covariance_from_state(state);
      if (config_.publish_deskewed_scan) {
        output.deskewed_scan = config_.deskewed_scan_voxel_size > 0.0
                                   ? core::voxel_down_sample(deskewed_frame, config_.deskewed_scan_voxel_size)
                                   : deskewed_frame;
      }
      if (on_output_) {
        on_output_(output);
      }
    } catch (const std::exception& ex) {
      // std::invalid_argument (Keypoints=0 / dt) and std::runtime_error (no correspondences) are
      // recoverable, as in the ROS 2 node.
      warn_(std::string("Encountered error, dropping frame. Error: ") + ex.what());
    }
  }
}

void OfflineDriver::write_tum(const std::string& path) const {
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot write " + path);
  }
  const auto& poses =
      lio_->output_poses_with_timestamps.empty() ? lio_->poses_with_timestamps : lio_->output_poses_with_timestamps;
  for (const auto& [timestamp, pose] : poses) {
    const Eigen::Vector3d& translation = pose.translation();
    const Eigen::Quaterniond& quaternion = pose.so3().unit_quaternion();
    file << std::fixed << std::setprecision(6) << core::to_seconds(timestamp) << " " << translation.x() << " "
         << translation.y() << " " << translation.z() << " " << quaternion.x() << " " << quaternion.y() << " "
         << quaternion.z() << " " << quaternion.w() << "\n";
  }
}

} // namespace rko_lio::driver
