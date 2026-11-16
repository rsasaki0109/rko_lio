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

// ROS-free view of a sensor_msgs/PointCloud2 and the readers the RKO-LIO node uses. Reading follows
// sensor_msgs::PointCloud2ConstIterator exactly: point i of a field is at
// data + i * point_step + field.offset for i < height * width (row_step is not used), and a
// missing field throws std::runtime_error.

#pragma once
#include "rko_lio/core/util.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace rko_lio::driver {

struct PointField {
  // sensor_msgs/PointField datatype codes.
  static constexpr uint8_t INT8 = 1;
  static constexpr uint8_t UINT8 = 2;
  static constexpr uint8_t INT16 = 3;
  static constexpr uint8_t UINT16 = 4;
  static constexpr uint8_t INT32 = 5;
  static constexpr uint8_t UINT32 = 6;
  static constexpr uint8_t FLOAT32 = 7;
  static constexpr uint8_t FLOAT64 = 8;

  std::string name;
  uint32_t offset = 0;
  uint8_t datatype = 0;
  uint32_t count = 0;
};

// Borrows the point buffer; the caller keeps it alive while the view is used.
struct PointCloudView {
  uint32_t height = 0;
  uint32_t width = 0;
  uint32_t point_step = 0;
  std::vector<PointField> fields;
  const uint8_t* data = nullptr;
  std::size_t data_size = 0;
};

core::Vector3dVector point_cloud_to_eigen(const PointCloudView& cloud);

// Points and their raw per-point times from the first of the fields t, timestamp, time, stamps,
// offset_time (uint32, float32 or float64). Throws std::invalid_argument when there is none.
std::tuple<core::Vector3dVector, std::vector<double>> point_cloud_to_eigen_with_timestamps(
    const PointCloudView& cloud);

// uint16 "reflectivity", else float32 "intensity", else empty.
std::vector<float> point_cloud_to_intensity(const PointCloudView& cloud);

// One field of any scalar type as float; empty when the field does not exist.
std::vector<float> point_cloud_field_as_float(const PointCloudView& cloud, const std::string& field_name);

} // namespace rko_lio::driver
