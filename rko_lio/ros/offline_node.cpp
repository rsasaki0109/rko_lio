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
#include "rko_lio/core/profiler.hpp"
#include "rko_lio/ros/utils/rosbag.hpp"
// other
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

namespace {
template <typename T>
std::shared_ptr<T> deserialize_next_msg(const rclcpp::SerializedMessage& serialized_msg) {
  const auto msg = std::make_shared<T>();
  const rclcpp::Serialization<T> serializer;
  serializer.deserialize_message(&serialized_msg, msg.get());
  return msg;
}

using BagProgressPublisher = rclcpp::Publisher<std_msgs::msg::Float32MultiArray>;
} // namespace

namespace rko_lio::ros {
class OfflineNode : public ThreadedNode {
public:
  std::unique_ptr<utils::BufferableBag> bag;

  BagProgressPublisher::SharedPtr bag_progress_publisher;

  float total_bag_msgs = 0;
  float processed_bag_msgs = 0;
  std::chrono::steady_clock::time_point bag_start_time;

  explicit OfflineNode(const rclcpp::NodeOptions& options)
      : ThreadedNode("rko_lio_offline_node", options), bag_start_time(std::chrono::steady_clock::now()) {
    // bag reading
    const tf2::Duration skip_to_time = tf2::durationFromSec(node->declare_parameter<double>("skip_to_time", 0.0));
    // Fork addition: dispatch the optional radar and direct-visual-frontend image topics from
    // the same bag when those (default-off) features are enabled.
    std::vector<std::string> topics{imu_topic, lidar_topic};
    if (direct_visual_frontend) {
      topics.push_back(visual_image_topic);
    }
    if (!radar_topic.empty()) {
      topics.push_back(radar_topic);
    }
    bag = std::make_unique<utils::BufferableBag>(node->declare_parameter<std::string>("bag_path"),
                                                 std::make_shared<utils::BufferableBag::TFBridge>(*node),
                                                 topics, skip_to_time);
    total_bag_msgs = bag->message_count();
    bag_progress_publisher = node->create_publisher<std_msgs::msg::Float32MultiArray>("rko_lio/bag_progress", 10);
  }

  void publish_bag_progress() const {
    const auto now = std::chrono::steady_clock::now();
    const float elapsed_seconds = std::chrono::duration<float>(now - bag_start_time).count();

    const float percent_complete = 100.0F * processed_bag_msgs / total_bag_msgs;
    const float avg_time_per_msg = (processed_bag_msgs > 0) ? elapsed_seconds / processed_bag_msgs : 0.0F;
    const float seconds_remaining = avg_time_per_msg * (total_bag_msgs - processed_bag_msgs);

    std_msgs::msg::Float32MultiArray progress_msg;
    progress_msg.layout.dim.resize(1);
    progress_msg.layout.dim[0].label = "percent_complete,seconds_remaining";
    progress_msg.layout.dim[0].size = 2;
    progress_msg.layout.dim[0].stride = 2;
    progress_msg.data = {percent_complete, seconds_remaining};

    bag_progress_publisher->publish(progress_msg);
  }

  void run() {
    while (rclcpp::ok() && !bag->finished()) {
      {
        if (lidar_buffer.size() >= 0.9 * max_lidar_buffer_size) {
          RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                                  "Lidar buffer size: " << lidar_buffer.size()
                                                        << ", max_lidar_buffer_size: " << max_lidar_buffer_size
                                                        << ", throttling the bag reading thread as it's too fast.\n");
          // this is a hack. can be improved
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          continue;
        }
      }
      const rosbag2_storage::SerializedBagMessage serialized_bag_msg = bag->PopNextMessage();
      const auto& topic_name = serialized_bag_msg.topic_name;
      const rclcpp::SerializedMessage serialized_msg(*serialized_bag_msg.serialized_data);

      // check the topic and call the appropriate callback
      if (topic_name == imu_topic) {
        const auto& imu_msg = deserialize_next_msg<sensor_msgs::msg::Imu>(serialized_msg);
        imu_callback(imu_msg);
      } else if (topic_name == lidar_topic) {
        const auto& lidar_msg = deserialize_next_msg<sensor_msgs::msg::PointCloud2>(serialized_msg);
        lidar_callback(lidar_msg);
      } else if (direct_visual_frontend && topic_name == visual_image_topic) {
        // Fork addition: direct visual odometry frontend image dispatch.
        const auto& image_msg = deserialize_next_msg<sensor_msgs::msg::Image>(serialized_msg);
        image_callback(image_msg);
      } else if (!radar_topic.empty() && topic_name == radar_topic) {
        // Fork addition: radar ego-velocity fusion dispatch.
        const auto& radar_msg = deserialize_next_msg<sensor_msgs::msg::PointCloud2>(serialized_msg);
        radar_callback(radar_msg);
      }

      processed_bag_msgs++;
      publish_bag_progress();
    }
    while (rclcpp::ok()) {
      {
        // Even if the bag finishes, wait until every queued LiDAR frame has either been
        // registered or dropped. Trailing IMU messages are expected in many bags and should
        // not keep the offline run alive forever.
        std::lock_guard<std::mutex> lock(buffer_mutex);
        if (lidar_buffer.empty() && !registration_busy.load()) {
          break;
        }
        // Fork addition (bug fix): a trailing frame whose sweep extends past the last IMU
        // record can never satisfy the processing predicate once the bag is exhausted -- no
        // producer will ever set atomic_can_process again, so the pipeline is permanently
        // idle. Drop the leftovers instead of hanging forever (previously required an
        // external `timeout -s INT` to terminate offline_node at end-of-bag).
        if (!atomic_can_process && !registration_busy.load()) {
          RCLCPP_INFO_STREAM(node->get_logger(),
                             "Bag exhausted with " << lidar_buffer.size()
                                                   << " trailing LiDAR frame(s) that can no longer be matched with "
                                                      "IMU data. Dropping them and finishing the run.");
          break;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
};
} // namespace rko_lio::ros

int main(int argc, char** argv) {
  const rko_lio::core::Timer timer("RKO LIO Offline Node");
  rclcpp::init(argc, argv);
  auto node = rko_lio::ros::OfflineNode(rclcpp::NodeOptions());
  node.run();
  rclcpp::shutdown();
  return 0;
}
