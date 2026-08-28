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
#include "rko_lio/core/profiler.hpp"
#include "rko_lio/ros/utils/rosbag.hpp"
// other
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <thread>

namespace {
template <typename T>
std::shared_ptr<T> deserialize_next_msg(const rclcpp::SerializedMessage& serialized_msg) {
  const auto msg = std::make_shared<T>();
  const rclcpp::Serialization<T> serializer;
  serializer.deserialize_message(&serialized_msg, msg.get());
  return msg;
}

using BagProgressPublisher = rclcpp::Publisher<std_msgs::msg::Float32MultiArray>;
void publish_bag_progress(const BagProgressPublisher::SharedPtr& publisher,
                          const size_t processed_bag_msgs,
                          const size_t total_bag_msgs) {
  if (total_bag_msgs == 0) {
    return;
  }
  static const auto start_time = std::chrono::steady_clock::now();
  const auto now = std::chrono::steady_clock::now();
  const float elapsed_seconds = std::chrono::duration<float>(now - start_time).count();

  const float percent_complete = 100.0F * processed_bag_msgs / total_bag_msgs;
  const float avg_time_per_msg = (processed_bag_msgs > 0) ? elapsed_seconds / processed_bag_msgs : 0.0F;
  const float seconds_remaining = avg_time_per_msg * (total_bag_msgs - processed_bag_msgs);

  std_msgs::msg::Float32MultiArray progress_msg;
  progress_msg.data.resize(2);
  progress_msg.data[0] = percent_complete;
  progress_msg.data[1] = seconds_remaining;

  publisher->publish(progress_msg);
}

bool env_equals(const char* name, const char* expected) {
  const char* value = std::getenv(name);
  return value != nullptr && std::string(value) == expected;
}

std::string env_value(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string(value);
}

double env_double_or_default(const char* name, const double fallback) {
  const std::string value = env_value(name);
  if (value.empty()) {
    return fallback;
  }
  try {
    std::size_t consumed = 0;
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
      throw std::invalid_argument("not a finite decimal");
    }
    return parsed;
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(name) + " must be a finite number");
  }
}
} // namespace

namespace rko_lio::ros {
class OfflineNode : public Node {
public:
  std::unique_ptr<utils::BufferableBag> bag;

  BagProgressPublisher::SharedPtr bag_progress_publisher;

  float total_bag_msgs = 0;
  float processed_bag_msgs = 0;
  double replay_start_delay_sec = 0.0;
  bool single_message_buffer_enabled = false;
  // Evidence is committed as soon as reader EOF and processing drain are
  // known.  The final call at the end of run() is an idempotent fallback for
  // ordinary shutdown paths; it must not wait for graph/map post-processing.
  bool benchmark_evidence_written = false;
  double benchmark_drain_timeout_sec = 30.0;
  std::string benchmark_drain_diagnostic_path;
  bool benchmark_eof_diagnostic_written = false;
  bool benchmark_timeout_diagnostic_written = false;

  explicit OfflineNode(const rclcpp::NodeOptions& options) : Node("rko_lio_offline_node", options) {
    // increase the lidar buffer limit because we're offline
    max_lidar_buffer_size = 100;
    // bag reading
    const tf2::Duration skip_to_time = tf2::durationFromSec(node->declare_parameter<double>("skip_to_time", 0.0));
    std::vector<std::string> topics{imu_topic, lidar_topic};
    if (direct_visual_frontend) {
      topics.push_back(visual_image_topic);
    }
    // ros2 launch normally propagates the benchmark environment, but the
    // explicit parameters are the authoritative bridge for launch/process
    // implementations that sanitize non-ROS environment variables.  The
    // wrapper writes these only for the additive v2 contract, so default and
    // ordinary offline runs remain inert.
    const std::string phase_contract = node->declare_parameter<std::string>(
        "m6a10_phase_contract_version", env_value("M6A10_PHASE_CONTRACT_VERSION"));
    const std::string configured_phase_mode = node->declare_parameter<std::string>(
        "m6a10_phase_mode", env_value("M6A10_PHASE_MODE"));
    const std::string configured_evidence_path = node->declare_parameter<std::string>(
        "m6a10_consumer_evidence_path", env_value("M6A10_CONSUMER_EVIDENCE"));
    const bool configured_single_message_buffer = node->declare_parameter<bool>(
        "m6a10_single_message_buffer", false);
    const bool v2_contract = phase_contract == "m6a10-online-compute-v2";
    // The callback boundary is synchronous even when BufferableBag performs
    // bounded disk/decompression prefetch.  A one-message buffer remains an
    // explicit diagnostic option, but is not required for the v2 throughput
    // contract and defaults off to avoid making capacity=1 a gate.
    single_message_buffer_enabled =
        v2_contract && configured_phase_mode == "unpaced_ack" &&
        configured_single_message_buffer;
    bag = std::make_unique<utils::BufferableBag>(node->declare_parameter<std::string>("bag_path"),
                                                 std::make_shared<utils::BufferableBag::TFBridge>(node),
                                                 topics, skip_to_time,
                                                 std::chrono::seconds(1), single_message_buffer_enabled);
    replay_start_delay_sec =
        node->declare_parameter<double>("offline_replay_start_delay_sec", 0.0);
    if (replay_start_delay_sec < 0.0) {
      throw std::invalid_argument("offline_replay_start_delay_sec must be non-negative");
    }
    total_bag_msgs = bag->message_count();
    if (v2_contract && !configured_evidence_path.empty() && !configured_phase_mode.empty()) {
        benchmark_consumer.configure(
            configured_evidence_path, configured_phase_mode, bag->message_count(),
            bag->topic_message_counts(), bag->required_end_timestamp_ns());
        benchmark_drain_timeout_sec = node->declare_parameter<double>(
            "m6a10_drain_timeout_seconds",
            env_double_or_default("M6A10_DRAIN_TIMEOUT_SECONDS", 30.0));
        if (!std::isfinite(benchmark_drain_timeout_sec) || benchmark_drain_timeout_sec < 0.0) {
          throw std::invalid_argument("m6a10_drain_timeout_seconds must be finite and non-negative");
        }
        benchmark_drain_diagnostic_path = node->declare_parameter<std::string>(
            "m6a10_drain_diagnostic_path",
            env_value("M6A10_DRAIN_DIAGNOSTIC"));
        if (benchmark_drain_diagnostic_path.empty()) {
          const std::filesystem::path evidence_path(configured_evidence_path);
          benchmark_drain_diagnostic_path =
              (evidence_path.parent_path() / "m6a10_drain_diagnostic.json").string();
        }
    }
    bag_progress_publisher = node->create_publisher<std_msgs::msg::Float32MultiArray>("rko_lio/bag_progress", 10);
  }

  void pace_input(const std::int64_t timestamp_ns) {
    if (!benchmark_consumer.enabled || benchmark_consumer.phase_mode != "paced_1x") {
      return;
    }
    if (!pacing_origin_set) {
      pacing_origin_set = true;
      pacing_origin_ns = timestamp_ns;
      pacing_start = std::chrono::steady_clock::now();
      return;
    }
    const auto target = pacing_start + std::chrono::nanoseconds(timestamp_ns - pacing_origin_ns);
    const auto now = std::chrono::steady_clock::now();
    if (now < target) {
      std::this_thread::sleep_until(target);
    } else if (now - target > std::chrono::milliseconds(250)) {
      benchmark_consumer.record_pacing_late();
    }
  }

  bool write_benchmark_evidence(const int exit_status) {
    if (!benchmark_consumer.enabled) {
      return true;
    }
    if (benchmark_evidence_written) {
      return true;
    }
    const std::filesystem::path output_path(benchmark_consumer.evidence_path);
    const std::filesystem::path part_path = output_path.string() + ".part";
    try {
      if (output_path.empty() || output_path.filename().empty() ||
          std::filesystem::exists(output_path) || std::filesystem::exists(part_path)) {
        throw std::runtime_error("consumer evidence output already exists or is invalid");
      }
      if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
      }
      const auto timestamp_seconds = [](const std::int64_t timestamp_ns) -> nlohmann::json {
        if (timestamp_ns < 0) {
          return nullptr;
        }
        return static_cast<double>(timestamp_ns) / 1.0e9;
      };
      nlohmann::json expected_topic_counts = nlohmann::json::object();
      for (const auto& [topic, count] : benchmark_consumer.expected_topic_counts) {
        expected_topic_counts[topic] = count;
      }
      const auto received = benchmark_consumer.received_messages.load(std::memory_order_relaxed);
      const auto processed = benchmark_consumer.processed_messages.load(std::memory_order_relaxed);
      const auto dropped = benchmark_consumer.dropped_messages.load(std::memory_order_relaxed);
      const auto overflow = benchmark_consumer.queue_overflow.load(std::memory_order_relaxed);
      const auto late = benchmark_consumer.pacing_late_messages.load(std::memory_order_relaxed);
      const bool eof = benchmark_consumer.eof_observed.load(std::memory_order_acquire);
      const bool drained = benchmark_consumer.drain_complete.load(std::memory_order_acquire);
      const bool counts_ok = benchmark_consumer.expected_messages == received &&
                             received == processed && dropped == 0 && overflow == 0;
      const bool status_ok = exit_status == 0 && eof && drained && counts_ok;
      const bool paced = benchmark_consumer.phase_mode == "paced_1x";
      const bool unpaced = benchmark_consumer.phase_mode == "unpaced_ack";
      const auto last_timestamp = benchmark_consumer.last_processed_timestamp_ns.load(
          std::memory_order_relaxed);
      nlohmann::json evidence = {
          {"schema_version", 2},
          {"contract_version", "m6a10-online-compute-v2"},
          {"system", "ours"},
          {"phase_mode", benchmark_consumer.phase_mode},
          {"benchmark_only", true},
          {"ack_source_kind", unpaced ? "synchronous_processing" : "consumer_callback"},
          {"ack_source", "rko_lio::ros::OfflineNode::run callback dispatch"},
          {"ack_semantics", unpaced
                                 ? "next callback dispatch begins after callback return; bounded disk prefetch is lossless and outside the throughput gate"
                                 : "callback return acknowledgement"},
          {"prefetch_policy", single_message_buffer_enabled
                                  ? "single_message_buffer"
                                  : "bounded_time_window"},
          {"publisher_count_used", false},
          {"eof_observed", eof},
          {"eof_source", "BufferableBag::finished() after reader exhaustion and buffer drain"},
          {"expected_messages", benchmark_consumer.expected_messages},
          {"expected_topic_counts", expected_topic_counts},
          {"received_messages", received},
          {"processed_messages", processed},
          {"dropped_messages", dropped},
          {"queue_overflow", overflow},
          {"backlog_at_drain", drained ? 0 : 1},
          // The reader calls one callback synchronously at a time.  The
          // registration queue is reported separately and is never silently
          // promoted to a consumer backlog or publisher acknowledgement.
          {"maximum_backlog_messages", 0},
          {"maximum_registration_queue_messages",
           benchmark_consumer.maximum_registration_queue_messages.load(std::memory_order_relaxed)},
          {"first_processed_timestamp_seconds", timestamp_seconds(
              benchmark_consumer.first_processed_timestamp_ns.load(std::memory_order_relaxed))},
          {"last_processed_timestamp_seconds", timestamp_seconds(last_timestamp)},
          {"required_end_timestamp_seconds", timestamp_seconds(
              benchmark_consumer.required_end_timestamp_ns)},
          {"maximum_callback_latency_seconds",
           static_cast<double>(benchmark_consumer.maximum_callback_latency_ns.load(
               std::memory_order_relaxed)) / 1.0e9},
          {"paced_input_rate", paced ? nlohmann::json(1.0) : nlohmann::json(nullptr)},
          {"paced_input_rate_verified", paced && late == 0 && counts_ok},
          {"ack_backpressure_verified", unpaced && benchmark_consumer.ack_backpressure_enabled && counts_ok},
          {"single_message_buffer_verified", single_message_buffer_enabled},
          {"pacing_late_messages", late},
          {"processing_failures", benchmark_consumer.processing_failures.load(std::memory_order_relaxed)},
          {"drain_complete", drained},
          {"drain_source", "lidar_buffer empty and registration inactive; reader EOF and processing drain completed before evidence"},
          {"counter_evidence_path", benchmark_consumer.evidence_path},
          {"status", status_ok ? "pass" : "invalid"},
          {"failure_reason", benchmark_consumer.failure_reason().empty()
                                 ? nlohmann::json(nullptr)
                                 : nlohmann::json(benchmark_consumer.failure_reason())}};
      std::ofstream file(part_path, std::ios::binary | std::ios::trunc);
      if (!file.is_open()) {
        throw std::runtime_error("cannot open consumer evidence staging file");
      }
      file << evidence.dump() << '\n';
      file.flush();
      if (!file.good()) {
        throw std::runtime_error("cannot flush consumer evidence staging file");
      }
      file.close();
      std::filesystem::rename(part_path, output_path);
      benchmark_evidence_written = true;
      return true;
    } catch (const std::exception& ex) {
      RCLCPP_ERROR_STREAM(node->get_logger(), "Could not write consumer evidence: " << ex.what());
      return false;
    }
  }

  bool write_benchmark_drain_diagnostic(const std::string& event,
                                        const std::string& reason) {
    if (!benchmark_consumer.enabled || benchmark_drain_diagnostic_path.empty()) {
      return true;
    }
    bool& already_written = event == "eof"
                                ? benchmark_eof_diagnostic_written
                                : benchmark_timeout_diagnostic_written;
    if (already_written) {
      return true;
    }
    const auto snapshot = benchmark_drain_snapshot();
    try {
      std::filesystem::path output_path(benchmark_drain_diagnostic_path);
      const auto filename = output_path.filename().string();
      const auto extension_pos = filename.rfind('.');
      const auto stem = extension_pos == std::string::npos
                            ? filename
                            : filename.substr(0, extension_pos);
      const auto extension = extension_pos == std::string::npos
                                 ? std::string{}
                                 : filename.substr(extension_pos);
      output_path = output_path.parent_path() /
                    (stem + "." + event + extension);
      const std::filesystem::path part_path = output_path.string() + ".part";
      if (output_path.empty() || output_path.filename().empty() ||
          std::filesystem::exists(output_path) || std::filesystem::exists(part_path)) {
        throw std::runtime_error("drain diagnostic output already exists or is invalid");
      }
      if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
      }
      nlohmann::json diagnostic = {
          {"schema_version", 1},
          {"kind", "m6a10_drain_diagnostic"},
          {"status", event == "timeout" ? "timeout" : "eof_observed"},
          {"event", event},
          {"reason", reason},
          {"benchmark_only", true},
          {"contract_version", "m6a10-online-compute-v2"},
          {"phase_mode", benchmark_consumer.phase_mode},
          {"drain_timeout_seconds", benchmark_drain_timeout_sec},
          {"snapshot",
           {{"lidar_buffer_size", snapshot.lidar_buffer_size},
            {"imu_buffer_size", snapshot.imu_buffer_size},
            {"front_lidar_min_timestamp_ns", snapshot.front_lidar_min_timestamp_ns},
            {"front_lidar_max_timestamp_ns", snapshot.front_lidar_max_timestamp_ns},
            {"last_imu_timestamp_ns", snapshot.last_imu_timestamp_ns},
            {"timestamp_gap_ns", snapshot.timestamp_gap_ns},
            {"timestamp_gap_seconds", snapshot.timestamp_gap_ns < 0
                                           ? nlohmann::json(nullptr)
                                           : nlohmann::json(static_cast<double>(snapshot.timestamp_gap_ns) / 1.0e9)},
            {"atomic_can_process", snapshot.atomic_can_process},
            {"registration_active", snapshot.registration_active}}},
          {"counters",
           {{"expected_messages", snapshot.expected_messages},
            {"received_messages", snapshot.received_messages},
            {"processed_messages", snapshot.processed_messages},
            {"dropped_messages", snapshot.dropped_messages},
            {"queue_overflow", snapshot.queue_overflow},
            {"processing_failures", snapshot.processing_failures}}},
          {"eof_observed", benchmark_consumer.eof_observed.load(std::memory_order_acquire)},
          {"drain_complete", benchmark_consumer.drain_complete.load(std::memory_order_acquire)},
          {"evidence_path", benchmark_consumer.evidence_path}};
      std::ofstream file(part_path, std::ios::binary | std::ios::trunc);
      if (!file.is_open()) {
        throw std::runtime_error("cannot open drain diagnostic staging file");
      }
      file << diagnostic.dump() << '\n';
      file.flush();
      if (!file.good()) {
        throw std::runtime_error("cannot flush drain diagnostic staging file");
      }
      file.close();
      std::filesystem::rename(part_path, output_path);
      already_written = true;
      return true;
    } catch (const std::exception& ex) {
      RCLCPP_ERROR_STREAM(node->get_logger(), "Could not write drain diagnostic: " << ex.what());
      benchmark_consumer.set_failure_reason(ex.what());
      return false;
    }
  }

  int run() {
    int exit_status = 0;
    bool eof_observed = false;
    bool drain_complete = false;
    if (replay_start_delay_sec > 0.0) {
      RCLCPP_INFO(node->get_logger(),
                  "Waiting %.3f s before offline replay so output subscribers can connect.",
                  replay_start_delay_sec);
      std::this_thread::sleep_for(std::chrono::duration<double>(replay_start_delay_sec));
    }
    try {
      while (rclcpp::ok() && !bag->finished()) {
      bool throttle_bag_reading = false;
      size_t current_lidar_buffer_size = 0;
      {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        if (lidar_buffer.size() >= 0.9 * max_lidar_buffer_size) {
          throttle_bag_reading = true;
          current_lidar_buffer_size = lidar_buffer.size();
        }
      }
      if (throttle_bag_reading) {
        RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                                "Lidar buffer size: " << current_lidar_buffer_size
                                                      << ", max_lidar_buffer_size: " << max_lidar_buffer_size
                                                      << ", throttling the bag reading thread as it's too fast.\n");
        // this is a hack. can be improved
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }
        const rosbag2_storage::SerializedBagMessage serialized_bag_msg = bag->PopNextMessage();
        const auto& topic_name = serialized_bag_msg.topic_name;
        const auto message_timestamp_ns = bag->last_message_timestamp_ns();
        pace_input(message_timestamp_ns);
        const rclcpp::SerializedMessage serialized_msg(*serialized_bag_msg.serialized_data);

        benchmark_consumer.record_received();
        const auto callback_start = std::chrono::steady_clock::now();
        try {
          // Check the topic and call the appropriate callback.  This dispatch
          // is the benchmark consumer boundary; no publisher count is used.
          if (topic_name == imu_topic) {
            const auto& imu_msg = deserialize_next_msg<sensor_msgs::msg::Imu>(serialized_msg);
            imu_callback(imu_msg);
          } else if (topic_name == lidar_topic) {
            const auto& lidar_msg = deserialize_next_msg<sensor_msgs::msg::PointCloud2>(serialized_msg);
            lidar_callback(lidar_msg);
          } else if (direct_visual_frontend && topic_name == visual_image_topic) {
            const auto& image_msg = deserialize_next_msg<sensor_msgs::msg::Image>(serialized_msg);
            image_callback(image_msg);
          }
        } catch (const std::exception& ex) {
          if (!benchmark_consumer.enabled) {
            throw;
          }
          benchmark_consumer.record_processing_failure();
          benchmark_consumer.set_failure_reason(ex.what());
          exit_status = 1;
          break;
        }
        const auto callback_latency = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - callback_start).count();
        benchmark_consumer.record_processed(message_timestamp_ns, static_cast<std::uint64_t>(callback_latency));

        processed_bag_msgs++;
        publish_bag_progress(bag_progress_publisher, processed_bag_msgs, total_bag_msgs);
      }
    } catch (const std::exception& ex) {
      if (!benchmark_consumer.enabled) {
        throw;
      }
      benchmark_consumer.record_processing_failure();
      benchmark_consumer.set_failure_reason(ex.what());
      exit_status = 1;
    }
    eof_observed = bag->finished();
    benchmark_consumer.mark_eof(eof_observed);
    if (eof_observed && benchmark_consumer.enabled &&
        !write_benchmark_drain_diagnostic("eof", "reader_eof_before_processing_drain")) {
      exit_status = 1;
    }
    if (exit_status == 0 && rclcpp::ok() && eof_observed) {
      const auto drain_deadline = std::chrono::steady_clock::now() +
                                   std::chrono::duration<double>(benchmark_drain_timeout_sec);
      while (rclcpp::ok()) {
        {
          // Even if the bag finishes, wait until every queued LiDAR frame has
          // either been registered or dropped. Trailing IMU messages are expected
          // in many bags and should not keep the offline run alive forever.
          std::lock_guard<std::mutex> lock(buffer_mutex);
          if (lidar_buffer.empty() && !atomic_registration_active) {
            break;
          }
        }
        if (benchmark_consumer.enabled && std::chrono::steady_clock::now() >= drain_deadline) {
          const bool diagnostic_written = write_benchmark_drain_diagnostic(
              "timeout", "benchmark drain timeout before lidar buffer became empty");
          benchmark_consumer.set_failure_reason(
              "benchmark drain timeout before lidar buffer became empty");
          exit_status = diagnostic_written ? 124 : 1;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    {
      std::lock_guard<std::mutex> lock(buffer_mutex);
      drain_complete = lidar_buffer.empty() && !atomic_registration_active;
    }
    benchmark_consumer.mark_drained(drain_complete);
    // Commit the consumer proof before fixed-lag/map post-processing.  That
    // post-processing can wait on graph subscribers even after the bag has
    // reached EOF, so making evidence depend on it creates a deadlock in the
    // wrapper's fail-closed completion barrier.
    if (exit_status == 0 && eof_observed && drain_complete &&
        !write_benchmark_evidence(exit_status)) {
      exit_status = 1;
    }
    if (exit_status == 0 && eof_observed && drain_complete) {
      flush_fixed_lag_outputs();
    }
    if (exit_status == 0 && publish_fixed_lag_finalized && eof_observed && drain_complete) {
      // Allow reliable DDS writers to hand the final window to recorders and
      // graph subscribers before main() shuts down the ROS context.
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (!eof_observed && exit_status == 0) {
      exit_status = 130;
      benchmark_consumer.set_failure_reason("offline replay stopped before BufferableBag EOF");
    }
    if (!write_benchmark_evidence(exit_status)) {
      exit_status = 1;
    }
    return exit_status;
  }

private:
  bool pacing_origin_set = false;
  std::int64_t pacing_origin_ns = 0;
  std::chrono::steady_clock::time_point pacing_start{};
};
} // namespace rko_lio::ros

int main(int argc, char** argv) {
  const rko_lio::core::Timer timer("RKO LIO Offline Node");
  rclcpp::init(argc, argv);
  auto offline_node = rko_lio::ros::OfflineNode(rclcpp::NodeOptions());
  const int exit_status = offline_node.run();
  rclcpp::shutdown();
  return exit_status;
}
