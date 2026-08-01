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

#include "threaded_node.hpp"
#include "rko_lio/core/process_timestamps.hpp"
#include "rko_lio/core/profiler.hpp"
#include "rko_lio/ros/utils/utils.hpp"
// other
#include <stdexcept>

namespace {
using namespace std::literals;
} // namespace

namespace rko_lio::ros {

ThreadedNode::ThreadedNode(const std::string& node_name, const rclcpp::NodeOptions& options) : BaseNode(node_name, options) {
  max_lidar_buffer_size = static_cast<size_t>(node->declare_parameter<int>(
      "async.max_lidar_buffer_size", static_cast<int>(max_lidar_buffer_size)));
  const int output_publish_delay_ms = node->declare_parameter<int>("async.output_publish_delay_ms", 0);
  if (output_publish_delay_ms < 0) {
    throw std::invalid_argument("async.output_publish_delay_ms must be non-negative");
  }
  output_publish_delay = std::chrono::milliseconds(output_publish_delay_ms);
  registration_thread = std::jthread([this]() { registration_loop(); });
}

void ThreadedNode::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg) {
  if (!ensure_frame_and_extrinsics(imu_frame, imu_msg->header.frame_id, "IMU")) {
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

void ThreadedNode::lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) {
  if (!ensure_frame_and_extrinsics(lidar_frame, lidar_msg->header.frame_id, "LiDAR")) {
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
    const auto [timestamps, scan] = process_lidar_msg(lidar_msg);
    // Fork addition: per-point reflectivity/intensity, only parsed when an intensity-based
    // feature is enabled (see BaseNode::process_lidar_intensity).
    const std::vector<float> intensities = process_lidar_intensity(lidar_msg);
    {
      std::lock_guard lock(buffer_mutex);
      lidar_buffer.emplace(timestamps, scan, intensities);
      atomic_can_process = !imu_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().timestamps.max;
    }
    if (atomic_can_process) {
      sync_condition_variable.notify_one();
    }
  } catch (const std::invalid_argument& ex) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Encountered error, dropping frame: Error. " << ex.what());
  }
}

void ThreadedNode::registration_loop() {
  while (rclcpp::ok() && atomic_node_running) {
    SCOPED_PROFILER("ROS Registration Loop");
    std::unique_lock buffer_lock(buffer_mutex);
    sync_condition_variable.wait(buffer_lock, [this]() { return !atomic_node_running || atomic_can_process; });
    if (!atomic_node_running) {
      // node could have been killed after waiting on the cv
      break;
    }
    LidarFrame frame = std::move(lidar_buffer.front());
    lidar_buffer.pop();
    registration_busy = true;
    const auto& [timestamps, scan, intensities] = frame;
    const auto& [start_stamp, end_stamp, time_vector] = timestamps;
    const std::vector<float>* intensities_ptr = intensities.empty() ? nullptr : &intensities;
    for (; !imu_buffer.empty() && imu_buffer.front().time < end_stamp; imu_buffer.pop()) {
      const core::ImuControl& imu_data = imu_buffer.front();
      lio->add_imu_measurement(extrinsic_imu2base, imu_data);
    }
    // check if there are more messages buffered already
    atomic_can_process =
        !imu_buffer.empty() && !lidar_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().timestamps.max;
    buffer_lock.unlock(); // we dont touch the buffers anymore

    bool published_outputs = false;
    try {
      // Fork additions: prepare any pending visual/radar priors for this scan before
      // register_scan consumes them (one-shot; see LIO::register_scan). Default-off unless
      // the corresponding config flags are set.
      std::optional<LiveVisualKeyframe> direct_visual_frame;
      if (direct_visual_frontend) {
        direct_visual_frame = prepare_direct_visual_prior(scan, end_stamp);
      } else {
        prepare_visual_prior(end_stamp);
      }
      prepare_radar_prior(end_stamp);
      const core::Vector3dVector deskewed_frame = register_scan_locked(scan, time_vector, intensities_ptr);
      if (!deskewed_frame.empty()) {
        // TODO: first frame is skipped and an empty frame is returned. improve how we handle this
        publish_lidar_outputs(deskewed_frame);
        publish_tf(lio->lidar_state);
        published_outputs = true;
        if (direct_visual_frame.has_value()) {
          commit_direct_visual_keyframe(std::move(*direct_visual_frame), end_stamp, deskewed_frame);
        }
      }
    } catch (const std::exception& ex) {
      // Catch both std::invalid_argument (Keypoints=0 / Δt) and std::runtime_error (Number of
      // correspondences=0). Both are recoverable on kidnap-style bags (fork addition).
      RCLCPP_ERROR_STREAM(node->get_logger(), "Encountered error, dropping frame. Error: " << ex.what());
    }
    // Offline frontends can otherwise publish much faster than a synchronous
    // graph backend can consume loop-search inputs. The default remains zero
    // for online operation; dataset profiles may opt into bounded pacing.
    if (published_outputs && output_publish_delay.count() > 0) {
      std::this_thread::sleep_for(output_publish_delay);
    }
    registration_busy = false;
  }
  atomic_node_running = false;
}

ThreadedNode::~ThreadedNode() {
  atomic_node_running = false;
  sync_condition_variable.notify_all();
}

} // namespace rko_lio::ros
