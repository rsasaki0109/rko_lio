#include <gtest/gtest.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include "rko_lio/ros/threaded_node.hpp"

using rko_lio::ros::ThreadedNode;
using namespace std::chrono_literals;
namespace {
rclcpp::NodeOptions options(int capacity = 1) {
  return rclcpp::NodeOptions().parameter_overrides({
      {"imu_topic", "imu"}, {"lidar_topic", "points"}, {"base_frame", "base"},
      {"imu_frame", "base"}, {"lidar_frame", "base"}, {"deskew", false},
      {"extrinsic_imu2base_quat_xyzw_xyz", std::vector<double>{0, 0, 0, 1, 0, 0, 0}},
      {"extrinsic_lidar2base_quat_xyzw_xyz", std::vector<double>{0, 0, 0, 1, 0, 0, 0}},
      {"max_num_threads", 1}, {"dump_results", false},
      {"async.max_lidar_buffer_size", capacity}});
}
auto cloud(int seconds) {
  auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  msg->header.frame_id = "base";
  msg->header.stamp.sec = seconds;
  sensor_msgs::PointCloud2Modifier modifier(*msg);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(1);
  sensor_msgs::PointCloud2Iterator<float> x(*msg, "x");
  x[0] = 2.0F; x[1] = 1.0F; x[2] = 0.0F;
  return msg;
}
void stop(ThreadedNode& node) {
  node.atomic_node_running = false;
  node.sync_condition_variable.notify_all();
  node.registration_thread.join();
}
class LatestPending : public ::testing::Test {
  void SetUp() override { if (!rclcpp::ok()) rclcpp::init(0, nullptr); }
  void TearDown() override { rclcpp::shutdown(); }
};
TEST_F(LatestPending, ReplacesOldestPendingAndRetainsCapacity) {
  ThreadedNode node("queue_test", options(2));
  stop(node);
  node.lidar_callback(cloud(1)); node.lidar_callback(cloud(2)); node.lidar_callback(cloud(3));
  ASSERT_EQ(node.lidar_buffer.size(), 2U);
  EXPECT_EQ(node.lidar_buffer.front().timestamps.max, 2s);
  node.lidar_buffer.pop();
  EXPECT_EQ(node.lidar_buffer.front().timestamps.max, 3s);
  EXPECT_FALSE(node.atomic_can_process);
}
TEST_F(LatestPending, FailedConversionPreservesPendingFrame) {
  ThreadedNode node("queue_test", options());
  stop(node);
  node.lidar_callback(cloud(1));
  node.lio->config.deskew = true; // Valid XYZ but no required per-point time.
  node.lidar_callback(cloud(2));
  ASSERT_EQ(node.lidar_buffer.size(), 1U);
  EXPECT_EQ(node.lidar_buffer.front().timestamps.max, 1s);
}
TEST_F(LatestPending, ReplacementRecomputesImuReadiness) {
  ThreadedNode node("queue_test", options());
  stop(node);
  auto imu = std::make_shared<sensor_msgs::msg::Imu>();
  imu->header.frame_id = "base"; imu->header.stamp.sec = 2;
  node.imu_callback(imu);
  node.lidar_callback(cloud(1));
  ASSERT_TRUE(node.atomic_can_process);
  node.lidar_callback(cloud(3));
  EXPECT_FALSE(node.atomic_can_process);
  EXPECT_EQ(node.imu_buffer.size(), 1U);
  imu->header.stamp.sec = 4; node.imu_callback(imu);
  EXPECT_TRUE(node.atomic_can_process);
  EXPECT_EQ(node.lidar_buffer.front().timestamps.max, 3s);
}
TEST_F(LatestPending, WorkerWaitsForNewFrameImuAndDrainsQueue) {
  ThreadedNode node("queue_test", options());
  node.lidar_callback(cloud(1)); node.lidar_callback(cloud(3));
  auto imu = std::make_shared<sensor_msgs::msg::Imu>();
  imu->header.frame_id = "base"; imu->header.stamp.sec = 2;
  node.imu_callback(imu);
  std::this_thread::sleep_for(20ms);
  { std::lock_guard lock(node.buffer_mutex);
    EXPECT_EQ(node.lidar_buffer.size(), 1U);
    EXPECT_FALSE(node.atomic_can_process); }
  imu->header.stamp.sec = 4; node.imu_callback(imu);
  bool drained = false;
  for (int i = 0; i < 200 && !drained; ++i) {
    { std::lock_guard lock(node.buffer_mutex); drained = node.lidar_buffer.empty(); }
    if (!drained) std::this_thread::sleep_for(5ms);
  }
  EXPECT_TRUE(drained);
  stop(node);
  EXPECT_FALSE(node.atomic_node_running);
}
TEST_F(LatestPending, RejectsNonpositiveCapacity) {
  const auto context = rclcpp::contexts::get_global_default_context();
  const auto callbacks_before = context->get_on_shutdown_callbacks().size();
  EXPECT_THROW(ThreadedNode("zero", options(0)), std::invalid_argument);
  EXPECT_THROW(ThreadedNode("negative", options(-1)), std::invalid_argument);
  EXPECT_EQ(context->get_on_shutdown_callbacks().size(), callbacks_before);
}
TEST_F(LatestPending, DestructionUnregistersShutdownCallback) {
  const auto context = rclcpp::contexts::get_global_default_context();
  const auto callbacks_before = context->get_on_shutdown_callbacks().size();
  {
    ThreadedNode node("queue_test", options());
    stop(node);
    EXPECT_GT(context->get_on_shutdown_callbacks().size(), callbacks_before);
  }
  EXPECT_EQ(context->get_on_shutdown_callbacks().size(), callbacks_before);
}
TEST_F(LatestPending, ShutdownCallbackUsesNodeContext) {
  const auto context = std::make_shared<rclcpp::Context>();
  context->init(0, nullptr);
  const auto global = rclcpp::contexts::get_global_default_context();
  const auto global_callbacks_before = global->get_on_shutdown_callbacks().size();
  const auto callbacks_before = context->get_on_shutdown_callbacks().size();
  {
    ThreadedNode node("queue_test", options().context(context));
    stop(node);
    EXPECT_GT(context->get_on_shutdown_callbacks().size(), callbacks_before);
    EXPECT_EQ(global->get_on_shutdown_callbacks().size(), global_callbacks_before);
    context->shutdown("test shutdown while node is alive");
  }
  EXPECT_EQ(context->get_on_shutdown_callbacks().size(), callbacks_before);
}
} // namespace
