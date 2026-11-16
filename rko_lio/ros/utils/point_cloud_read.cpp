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

#include "point_cloud_read.hpp"

namespace rko_lio::ros::utils {
using Vector3dVector = std::vector<Eigen::Vector3d>;
using PointCloud2 = sensor_msgs::msg::PointCloud2;

driver::PointCloudView point_cloud_view(const PointCloud2& msg) {
  driver::PointCloudView view;
  view.height = msg.height;
  view.width = msg.width;
  view.point_step = msg.point_step;
  view.fields.reserve(msg.fields.size());
  for (const auto& field : msg.fields) {
    view.fields.push_back({field.name, field.offset, field.datatype, field.count});
  }
  view.data = msg.data.data();
  view.data_size = msg.data.size();
  return view;
}

Vector3dVector point_cloud2_to_eigen(const PointCloud2::ConstSharedPtr& msg) {
  return driver::point_cloud_to_eigen(point_cloud_view(*msg));
}

std::tuple<Vector3dVector, std::vector<double>> point_cloud2_to_eigen_with_timestamps(
    const PointCloud2::ConstSharedPtr& msg) {
  return driver::point_cloud_to_eigen_with_timestamps(point_cloud_view(*msg));
}

std::vector<float> point_cloud2_to_intensity(const PointCloud2::ConstSharedPtr& msg) {
  return driver::point_cloud_to_intensity(point_cloud_view(*msg));
}

std::vector<float> point_cloud2_field_as_float(const PointCloud2::ConstSharedPtr& msg, const std::string& field_name) {
  return driver::point_cloud_field_as_float(point_cloud_view(*msg), field_name);
}
} // namespace rko_lio::ros::utils
