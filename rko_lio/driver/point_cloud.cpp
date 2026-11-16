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

#include "rko_lio/driver/point_cloud.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace rko_lio::driver {

namespace {
const PointField& find_field(const PointCloudView& cloud, const std::string& name) {
  const auto field =
      std::find_if(cloud.fields.cbegin(), cloud.fields.cend(), [&](const PointField& f) { return f.name == name; });
  if (field == cloud.fields.cend()) {
    throw std::runtime_error("Field " + name + " does not exist");
  }
  return *field;
}

// Mirrors sensor_msgs::PointCloud2ConstIterator<T>: a pointer stepping point_step bytes.
template <typename T>
class FieldReader {
public:
  FieldReader(const PointCloudView& cloud, const std::string& name)
      : pointer_(cloud.data + find_field(cloud, name).offset), step_(cloud.point_step) {}
  T operator*() const {
    T value;
    std::memcpy(&value, pointer_, sizeof(T));
    return value;
  }
  FieldReader& operator++() {
    pointer_ += step_;
    return *this;
  }

private:
  const uint8_t* pointer_;
  std::size_t step_;
};

std::size_t point_count(const PointCloudView& cloud) {
  return static_cast<std::size_t>(cloud.height) * cloud.width;
}
} // namespace

core::Vector3dVector point_cloud_to_eigen(const PointCloudView& cloud) {
  const std::size_t count = point_count(cloud);
  core::Vector3dVector points;
  points.reserve(count);
  FieldReader<float> x(cloud, "x");
  FieldReader<float> y(cloud, "y");
  FieldReader<float> z(cloud, "z");
  for (std::size_t i = 0; i < count; ++i, ++x, ++y, ++z) {
    points.emplace_back(*x, *y, *z);
  }
  return points;
}

std::tuple<core::Vector3dVector, std::vector<double>> point_cloud_to_eigen_with_timestamps(
    const PointCloudView& cloud) {
  const std::size_t count = point_count(cloud);
  core::Vector3dVector points;
  points.reserve(count);
  FieldReader<float> x(cloud, "x");
  FieldReader<float> y(cloud, "y");
  FieldReader<float> z(cloud, "z");

  std::vector<double> raw_timestamps;
  raw_timestamps.reserve(count);

  const PointField* timestamp_field = nullptr;
  for (const PointField& field : cloud.fields) {
    // offset_time: Livox ROS driver (uint32 nanoseconds from the scan header stamp)
    if (field.name == "t" || field.name == "timestamp" || field.name == "time" || field.name == "stamps" ||
        field.name == "offset_time") {
      if (field.count != 0U) {
        timestamp_field = &field;
        break;
      }
    }
  }
  if (timestamp_field == nullptr) {
    throw std::invalid_argument("Point cloud needs timestamps for deskewing");
  }

  auto extract_points_and_timestamps = [&](auto&& time_reader) {
    for (std::size_t i = 0; i < count; ++i, ++x, ++y, ++z, ++time_reader) {
      points.emplace_back(*x, *y, *z);
      raw_timestamps.emplace_back(static_cast<double>(*time_reader));
    }
  };

  switch (timestamp_field->datatype) {
  case PointField::UINT32:
    extract_points_and_timestamps(FieldReader<uint32_t>(cloud, timestamp_field->name));
    break;
  case PointField::FLOAT32:
    extract_points_and_timestamps(FieldReader<float>(cloud, timestamp_field->name));
    break;
  case PointField::FLOAT64:
    extract_points_and_timestamps(FieldReader<double>(cloud, timestamp_field->name));
    break;
  default:
    throw std::invalid_argument("Unsupported timestamp field datatype. Please open an issue.");
  }
  return {points, raw_timestamps};
}

std::vector<float> point_cloud_to_intensity(const PointCloudView& cloud) {
  const std::size_t count = point_count(cloud);
  const PointField* reflectivity_field = nullptr;
  const PointField* intensity_field = nullptr;
  for (const PointField& field : cloud.fields) {
    if (field.name == "reflectivity") {
      reflectivity_field = &field;
    } else if (field.name == "intensity") {
      intensity_field = &field;
    }
  }
  std::vector<float> intensities;
  if (reflectivity_field != nullptr && reflectivity_field->datatype == PointField::UINT16) {
    intensities.reserve(count);
    FieldReader<uint16_t> reflectivity(cloud, "reflectivity");
    for (std::size_t i = 0; i < count; ++i, ++reflectivity) {
      intensities.push_back(static_cast<float>(*reflectivity));
    }
    return intensities;
  }
  if (intensity_field != nullptr && intensity_field->datatype == PointField::FLOAT32) {
    intensities.reserve(count);
    FieldReader<float> intensity(cloud, "intensity");
    for (std::size_t i = 0; i < count; ++i, ++intensity) {
      intensities.push_back(*intensity);
    }
    return intensities;
  }
  // Neither a uint16 "reflectivity" nor a float32 "intensity" field: intensity unavailable.
  return {};
}

std::vector<float> point_cloud_field_as_float(const PointCloudView& cloud, const std::string& field_name) {
  const std::size_t count = point_count(cloud);
  const auto field = std::find_if(cloud.fields.cbegin(), cloud.fields.cend(),
                                  [&](const PointField& f) { return f.name == field_name; });
  if (field == cloud.fields.cend()) {
    return {};
  }
  std::vector<float> values;
  values.reserve(count);
  auto read = [&](auto tag) {
    FieldReader<decltype(tag)> it(cloud, field_name);
    for (std::size_t i = 0; i < count; ++i, ++it) {
      values.push_back(static_cast<float>(*it));
    }
  };
  switch (field->datatype) {
  case PointField::FLOAT32: read(float{}); break;
  case PointField::FLOAT64: read(double{}); break;
  case PointField::UINT8: read(uint8_t{}); break;
  case PointField::UINT16: read(uint16_t{}); break;
  case PointField::UINT32: read(uint32_t{}); break;
  case PointField::INT8: read(int8_t{}); break;
  case PointField::INT16: read(int16_t{}); break;
  case PointField::INT32: read(int32_t{}); break;
  default: return {};
  }
  return values;
}

} // namespace rko_lio::driver
