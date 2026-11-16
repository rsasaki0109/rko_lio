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

// Everything the RKO-LIO node reads from its parameters that does not create a ROS object: the
// LIO configuration, sensor and frame names, output switches, timestamp handling and extrinsics.
// The ROS 2 node and the ROS-free driver both build it with declare_driver_config(), so the two
// read the same names with the same defaults.

#pragma once
#include "rko_lio/core/lio.hpp"
#include "rko_lio/core/process_timestamps.hpp"
#include "rko_lio/core/radar_ego_velocity.hpp"
#include "rko_lio/driver/parameters.hpp"
#include <chrono>
#include <string>

namespace rko_lio::driver {

// An extrinsic that may be unset. (Not std::optional<Sophus::SE3d>: GCC 13 rejects the implicitly
// generated move constructor of that optional inside DriverConfig in some builds.)
struct Extrinsic {
  bool set = false;
  Sophus::SE3d pose;
  explicit operator bool() const { return set; }
  const Sophus::SE3d& operator*() const { return pose; }
};

struct DriverConfig {
  // Sensors and frames.
  std::string imu_topic;
  std::string lidar_topic;
  std::string base_frame;
  std::string imu_frame = "";   // default: from the first IMU message
  std::string lidar_frame = ""; // default: from the first LiDAR message
  std::string odom_frame = "odom";
  std::string odom_topic = "rko_lio/odom";
  bool invert_odom_tf = false;

  // Outputs.
  bool publish_lidar_acceleration = false;
  bool publish_deskewed_scan = false;
  std::string deskewed_scan_topic = "rko_lio/frame";
  double deskewed_scan_voxel_size = 0.0;
  bool publish_local_map = false;
  std::string map_topic = "rko_lio/local_map";
  core::Nsec publish_map_after = std::chrono::seconds(1);
  bool dump_results = false;
  std::string results_dir = "results";
  std::string run_name = "rko_lio_run";

  // Inputs.
  std::string imu_acceleration_unit = "mps2";
  core::TimestampProcessingConfig timestamp_proc_config;
  /** Point field used for the photometric intensity images. */
  std::string photometric_channel = "intensity";

  // Odometry.
  core::LIO::Config lio;

  // Optional sensors (fork additions, default off).
  bool direct_visual_frontend = false;
  bool direct_visual_require_previous_weak_direction = false;
  std::string radar_topic = "";
  core::RadarEgoVelocityConfig radar_ego_velocity_config;
  double radar_velocity_scale = 1.0;
  double radar_sigma_forward_mps = 0.15;
  double radar_sigma_lateral_mps = 0.5;
  double radar_sigma_vertical_mps = 0.5;

  // Extrinsics given as parameters (extrinsic_<name>2base_quat_xyzw_xyz). Unset ones come from
  // TF in the ROS 2 node; the ROS-free driver requires the IMU and LiDAR ones.
  Extrinsic extrinsic_imu2base;
  Extrinsic extrinsic_lidar2base;
  Extrinsic extrinsic_cam2base;   // read only with visual fusion on
  Extrinsic extrinsic_radar2base; // read only with radar fusion on
};

// Reads every parameter above, in the ROS 2 node's names, types and defaults. Throws
// std::invalid_argument / std::runtime_error on invalid values, as the node does.
DriverConfig declare_driver_config(Parameters& params);

} // namespace rko_lio::driver
