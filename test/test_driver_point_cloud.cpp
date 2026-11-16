// MIT License
//
// Copyright (c) 2025 Meher V.R. Malladi.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// The ROS-free point cloud readers (rko_lio/driver/point_cloud.hpp) read exactly what
// sensor_msgs::PointCloud2ConstIterator reads, including padded points and missing fields.

#include "rko_lio/driver/point_cloud.hpp"
#include "rko_lio/ros/utils/point_cloud_read.hpp"
#include <gtest/gtest.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace {
sensor_msgs::msg::PointCloud2 make_cloud(const std::string& time_name, uint8_t time_type) {
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.height = 2;
  cloud.width = 7;
  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2Fields(6, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
                                sensor_msgs::msg::PointField::FLOAT32, "z", 1, sensor_msgs::msg::PointField::FLOAT32,
                                "intensity", 1, sensor_msgs::msg::PointField::FLOAT32, "reflectivity", 1,
                                sensor_msgs::msg::PointField::UINT16, time_name.c_str(), 1, time_type);
  // Padding after the last field, as Ouster and Hesai drivers produce.
  cloud.point_step += 6;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.data.assign(static_cast<size_t>(cloud.row_step) * cloud.height, 0xAB);
  const size_t n = static_cast<size_t>(cloud.height) * cloud.width;
  sensor_msgs::PointCloud2Iterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z"), intensity(cloud, "intensity");
  sensor_msgs::PointCloud2Iterator<uint16_t> reflectivity(cloud, "reflectivity");
  for (size_t i = 0; i < n; ++i, ++x, ++y, ++z, ++intensity, ++reflectivity) {
    *x = 0.5F * static_cast<float>(i);
    *y = -1.25F * static_cast<float>(i);
    *z = 3.0F + static_cast<float>(i) / 7.0F;
    *intensity = 10.0F + static_cast<float>(i);
    *reflectivity = static_cast<uint16_t>(100 + 3 * i);
  }
  if (time_type == sensor_msgs::msg::PointField::FLOAT64) {
    sensor_msgs::PointCloud2Iterator<double> t(cloud, time_name);
    for (size_t i = 0; i < n; ++i, ++t) *t = 0.001 * static_cast<double>(i);
  } else if (time_type == sensor_msgs::msg::PointField::FLOAT32) {
    sensor_msgs::PointCloud2Iterator<float> t(cloud, time_name);
    for (size_t i = 0; i < n; ++i, ++t) *t = 0.001F * static_cast<float>(i);
  } else {
    sensor_msgs::PointCloud2Iterator<uint32_t> t(cloud, time_name);
    for (size_t i = 0; i < n; ++i, ++t) *t = static_cast<uint32_t>(1000 * i);
  }
  return cloud;
}
} // namespace

TEST(DriverPointCloud, MatchesTheRosIterator) {
  for (const auto& [name, type] : std::vector<std::pair<std::string, uint8_t>>{
           {"t", sensor_msgs::msg::PointField::UINT32},
           {"time", sensor_msgs::msg::PointField::FLOAT32},
           {"timestamp", sensor_msgs::msg::PointField::FLOAT64},
           {"offset_time", sensor_msgs::msg::PointField::UINT32}}) {
    const auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>(make_cloud(name, type));
    const auto view = rko_lio::ros::utils::point_cloud_view(*msg);
    const auto [points, times] = rko_lio::driver::point_cloud_to_eigen_with_timestamps(view);
    sensor_msgs::PointCloud2ConstIterator<float> x(*msg, "x"), y(*msg, "y"), z(*msg, "z");
    ASSERT_EQ(points.size(), 14U);
    for (size_t i = 0; i < points.size(); ++i, ++x, ++y, ++z) {
      EXPECT_EQ(points[i].x(), static_cast<double>(*x));
      EXPECT_EQ(points[i].y(), static_cast<double>(*y));
      EXPECT_EQ(points[i].z(), static_cast<double>(*z));
    }
    EXPECT_EQ(rko_lio::driver::point_cloud_to_eigen(view), points);
    ASSERT_EQ(times.size(), 14U);
    EXPECT_EQ(times[3], type == sensor_msgs::msg::PointField::UINT32 ? 3000.0
                        : type == sensor_msgs::msg::PointField::FLOAT32 ? static_cast<double>(0.003F)
                                                                          : 0.003);
    // reflectivity (uint16) wins over intensity (float32)
    const auto intensity = rko_lio::driver::point_cloud_to_intensity(view);
    ASSERT_EQ(intensity.size(), 14U);
    EXPECT_EQ(intensity[5], 115.0F);
    const auto field = rko_lio::driver::point_cloud_field_as_float(view, "intensity");
    ASSERT_EQ(field.size(), 14U);
    EXPECT_EQ(field[5], 15.0F);
    EXPECT_TRUE(rko_lio::driver::point_cloud_field_as_float(view, "ring").empty());
  }
}

TEST(DriverPointCloud, MissingFieldsThrowLikeTheRosIterator) {
  sensor_msgs::msg::PointCloud2 msg = make_cloud("t", sensor_msgs::msg::PointField::UINT32);
  msg.fields.pop_back(); // no time field
  const auto view = rko_lio::ros::utils::point_cloud_view(msg);
  EXPECT_THROW(rko_lio::driver::point_cloud_to_eigen_with_timestamps(view), std::invalid_argument);
  msg.fields.erase(msg.fields.begin()); // no x
  EXPECT_THROW(rko_lio::driver::point_cloud_to_eigen(rko_lio::ros::utils::point_cloud_view(msg)), std::runtime_error);
}
