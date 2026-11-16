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

// The RKO-LIO offline pipeline without ROS: per-frame processing shared with the ROS 2 node
// (LiDAR timestamps, intensities, IMU units, odometry covariance) and OfflineDriver, which feeds
// a recorded log through the LIO in the same order as the ROS 2 offline node (ThreadedNode): a
// LiDAR frame is registered once an IMU sample newer than its last point has arrived, after every
// buffered IMU sample older than that point.

#pragma once
#include "rko_lio/core/lio.hpp"
#include "rko_lio/core/process_timestamps.hpp"
#include "rko_lio/driver/driver_config.hpp"
#include "rko_lio/driver/point_cloud.hpp"
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

namespace rko_lio::driver {

// ---- per-frame processing shared with the ROS 2 node ----

// Points and per-point times of one LiDAR frame (BaseNode::process_lidar_msg). With deskew off,
// every point gets the header stamp plus the configured offset.
std::tuple<core::Timestamps, core::Vector3dVector> process_lidar_frame(const PointCloudView& cloud,
                                                                       core::Nsec header_stamp,
                                                                       bool deskew,
                                                                       const core::TimestampProcessingConfig& config);

// Per-point values for the intensity-based features, empty when none is enabled
// (BaseNode::process_lidar_intensity). `field_missing` is set when a feature is on but the cloud
// has no usable field.
std::vector<float> lidar_frame_intensities(const PointCloudView& cloud,
                                           const core::LIO::Config& config,
                                           const std::string& photometric_channel,
                                           bool& field_missing);

// nav_msgs/Odometry pose covariance ([x, y, z, rot x, rot y, rot z], row-major) from the final ICP
// Hessian of the state, or a large diagonal when the scan had no ICP solve.
std::array<double, 36> pose_covariance_from_state(const core::State& state);

// ---- offline driver ----

struct OdometryOutput {
  core::Nsec time{0};
  Sophus::SE3d pose; // the output-filtered position when output_filter_position_sigma is set
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
  std::array<double, 36> pose_covariance{};
  // Deskewed scan in the LiDAR frame (voxelized by deskewed_scan_voxel_size); empty unless
  // publish_deskewed_scan is set.
  core::Vector3dVector deskewed_scan;
};

class OfflineDriver {
public:
  using Log = std::function<void(const std::string&)>;
  using OutputCallback = std::function<void(const OdometryOutput&)>;

  // Needs extrinsic_imu2base and extrinsic_lidar2base in the config (there is no TF without ROS).
  // Visual and radar fusion are not available here; enabling them throws std::invalid_argument.
  OfflineDriver(DriverConfig config, OutputCallback on_output, Log info, Log warn, std::size_t max_lidar_buffer_size = 10);

  void add_imu(core::Nsec time, const Eigen::Vector3d& linear_acceleration, const Eigen::Vector3d& angular_velocity);
  void add_lidar(const PointCloudView& cloud, core::Nsec header_stamp);

  // Frames still waiting for IMU data once the log ends; the ROS 2 offline node drops them too.
  std::size_t pending_lidar_frames() const { return lidar_buffer_.size(); }

  const DriverConfig& config() const { return config_; }
  const core::LIO& lio() const { return *lio_; }

  // The trajectory file of dump_results: one TUM line per registered scan, 6 decimals.
  void write_tum(const std::string& path) const;

private:
  struct LidarFrame {
    core::Timestamps timestamps;
    core::Vector3dVector scan;
    std::vector<float> intensities;
  };

  bool can_process() const;
  void process_ready_frames();

  DriverConfig config_;
  OutputCallback on_output_;
  Log info_;
  Log warn_;
  std::size_t max_lidar_buffer_size_;
  std::unique_ptr<core::LIO> lio_;
  Sophus::SE3d extrinsic_imu2base_;
  Sophus::SE3d extrinsic_lidar2base_;
  double imu_acceleration_scale_ = 0.0;
  bool warned_deskew_off_ = false;
  bool warned_intensity_missing_ = false;
  std::queue<core::ImuControl> imu_buffer_;
  std::queue<LidarFrame> lidar_buffer_;
};

} // namespace rko_lio::driver
