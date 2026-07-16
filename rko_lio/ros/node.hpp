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

#pragma once
#include "rko_lio/core/direct_visual_odometry.hpp"
#include "rko_lio/core/lio.hpp"
#include "rko_lio/core/process_timestamps.hpp"
#include "rko_lio/core/radar_ego_velocity.hpp"
// stl
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
// ros
#include <geometry_msgs/msg/accel_stamped.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

namespace rko_lio::ros {
struct VisualRelativeConstraint {
  core::Secondsd first_time{0.0};
  core::Secondsd second_time{0.0};
  Eigen::Matrix3d current_R_previous_camera = Eigen::Matrix3d::Identity();
  Eigen::Vector3d current_t_previous_direction = Eigen::Vector3d::Zero();
  core::VisualConstraintConfidence confidence;
};

struct VisualImageFrame {
  core::Secondsd time{0.0};
  core::GrayImage image;
};

struct LiveVisualKeyframe {
  core::Secondsd time{0.0};
  core::GrayImage image;
  core::SparseDepthImage depth;
  Sophus::SE3d world_T_base;
};

struct DirectVisualDiagnosticsSample {
  core::Secondsd time{0.0};
  bool solver_valid = false;
  bool confidence_gate_passed = false;
  int failure_reason = 0;
  int features = 0;
  int tracked_features = 0;
  int inliers = 0;
  double inlier_ratio = 0.0;
  double initial_rmse = std::numeric_limits<double>::infinity();
  double final_rmse = std::numeric_limits<double>::infinity();
  double exposure_gain = 1.0;
  double exposure_bias = 0.0;
  double rotation_error_deg = std::numeric_limits<double>::infinity();
  double translation_cosine = -1.0;
  double baseline_m = 0.0;
  double predicted_baseline_m = 0.0;
};

class Node {
public:
  rclcpp::Node::SharedPtr node;
  std::unique_ptr<core::LIO> lio;
  core::TimestampProcessingConfig timestamp_proc_config;

  std::string imu_topic;
  std::string imu_frame = ""; // default: get from the first imu message
  std::string lidar_topic;
  std::string lidar_frame = ""; // default: get from the first lidar message
  std::string base_frame;
  std::string odom_frame = "odom";
  std::string odom_topic = "rko_lio/odometry";
  std::string map_topic = "rko_lio/local_map";
  std::string deskewed_scan_topic = "rko_lio/frame";

  bool dump_results = false;
  std::string results_dir = "results";
  std::string run_name = "rko_lio_run";

  bool invert_odom_tf = false;
  bool publish_lidar_acceleration = false;
  bool publish_deskewed_scan = false;
  bool publish_local_map = false;

  Sophus::SE3d extrinsic_imu2base;
  Sophus::SE3d extrinsic_lidar2base;
  Sophus::SE3d extrinsic_cam2base;
  Sophus::SE3d extrinsic_radar2base;
  bool extrinsics_set = false;
  bool visual_extrinsic_set = false;
  bool radar_extrinsic_set = false;

  std::string radar_topic = ""; // default: radar velocity fusion disabled
  core::RadarEgoVelocityConfig radar_ego_velocity_config;
  // Estimates buffered by time: the offline reader runs far ahead of the
  // registration thread, so a latest-wins one-shot prior would always be
  // stale-gated. The registration loop picks the nearest estimate per scan.
  std::deque<core::RadarVelocityPrior> radar_prior_queue;
  std::mutex radar_prior_mutex;
  void prepare_radar_prior(const core::Secondsd& lidar_time);

  bool direct_visual_frontend = false;
  bool direct_visual_require_previous_weak_direction = false;
  std::string visual_image_topic = "/alphasense/cam0/image_raw";
  double visual_camera_time_offset_sec = 0.0;
  double visual_max_image_lidar_delta_sec = 0.04;
  double visual_keyframe_interval_sec = 0.25;
  core::FisheyeCameraModel visual_camera;
  core::DirectVisualOdometryConfig direct_visual_config;
  std::queue<VisualImageFrame> visual_image_buffer;
  std::optional<LiveVisualKeyframe> live_visual_keyframe;
  std::size_t direct_visual_attempt_count = 0;
  std::size_t direct_visual_weak_gate_skip_count = 0;
  std::size_t direct_visual_solver_valid_count = 0;
  std::size_t direct_visual_valid_count = 0;
  std::vector<DirectVisualDiagnosticsSample> direct_visual_diagnostics;

  std::vector<VisualRelativeConstraint> visual_constraints;
  std::size_t next_visual_constraint = 0;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frame_publisher;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher;
  rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr lidar_accel_publisher;

  // multithreading
  std::jthread map_publish_thead;
  core::Secondsd publish_map_after = std::chrono::seconds(1);
  std::mutex local_map_mutex;

  std::jthread registration_thread;
  std::mutex buffer_mutex;
  std::condition_variable sync_condition_variable;
  std::atomic<bool> atomic_node_running = true;
  std::atomic<bool> atomic_can_process = false;
  std::atomic<bool> atomic_registration_active = false;
  std::queue<core::ImuControl> imu_buffer;
  std::queue<core::LidarFrame> lidar_buffer;
  size_t max_lidar_buffer_size = 50;

  Node() = delete;
  Node(const std::string& node_name, const rclcpp::NodeOptions& options);

  void parse_cli_extrinsics();
  void load_visual_constraints();
  void configure_direct_visual_frontend();
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg);
  std::optional<LiveVisualKeyframe> prepare_direct_visual_prior(
      const core::Vector3dVector& scan, const core::Secondsd& lidar_time);
  void commit_direct_visual_keyframe(
      LiveVisualKeyframe frame, const core::Secondsd& lidar_time,
      const core::Vector3dVector& deskewed_lidar_frame);
  void prepare_visual_prior(const core::Secondsd& lidar_time);
  bool check_and_set_extrinsics();
  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg);
  void lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg);
  void radar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& radar_msg);
  void registration_loop();
  void publish_odometry(const core::State& state, const core::Secondsd& stamp) const;
  void publish_lidar_accel(const Eigen::Vector3d& acceleration, const core::Secondsd& stamp) const;
  void publish_map_loop();
  void dump_results_to_disk(const std::filesystem::path& results_dir, const std::string& run_name) const;

  ~Node();
  Node(const Node&) = delete;
  Node(Node&&) = delete;
  Node& operator=(const Node&) = delete;
  Node& operator=(Node&&) = delete;
};
} // namespace rko_lio::ros
