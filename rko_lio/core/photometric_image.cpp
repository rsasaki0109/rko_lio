/*
 * MIT License
 *
 * Copyright (c) 2026 rsasaki0109.
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

#include "photometric_image.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace rko_lio::core {

namespace {

constexpr double kTwoPi = 2.0 * M_PI;

// OpenCV's default border (BORDER_REFLECT_101): ... 2 1 | 0 1 2 ... n-2 n-1 | n-2 ...
int reflect101(int i, int n) {
  if (n == 1) {
    return 0;
  }
  while (i < 0 || i >= n) {
    i = i < 0 ? -i : 2 * n - 2 - i;
  }
  return i;
}

// Correlation with a vertical kernel (taps along rows), anchor at the centre tap.
template <typename RowFunction>
void for_each_row(int rows, const RowFunction& row_function) {
  tbb::parallel_for(tbb::blocked_range<int>(0, rows), [&](const tbb::blocked_range<int>& range) {
    for (int r = range.begin(); r != range.end(); ++r) {
      row_function(r);
    }
  });
}

FloatImage filter_vertical(const FloatImage& image, const std::vector<double>& kernel) {
  FloatImage out(image.rows, image.cols);
  const int anchor = static_cast<int>(kernel.size()) / 2;
  for_each_row(image.rows, [&](int r) {
    for (int c = 0; c < image.cols; ++c) {
      double sum = 0.0;
      for (int k = 0; k < static_cast<int>(kernel.size()); ++k) {
        sum += kernel[k] * image.at(reflect101(r + k - anchor, image.rows), c);
      }
      out.at(r, c) = static_cast<float>(sum);
    }
  });
  return out;
}

// Correlation with a horizontal kernel (taps along columns), anchor at the centre tap.
FloatImage filter_horizontal(const FloatImage& image, const std::vector<double>& kernel) {
  FloatImage out(image.rows, image.cols);
  const int anchor = static_cast<int>(kernel.size()) / 2;
  for_each_row(image.rows, [&](int r) {
    for (int c = 0; c < image.cols; ++c) {
      double sum = 0.0;
      for (int k = 0; k < static_cast<int>(kernel.size()); ++k) {
        sum += kernel[k] * image.at(r, reflect101(c + k - anchor, image.cols));
      }
      out.at(r, c) = static_cast<float>(sum);
    }
  });
  return out;
}

FloatImage box_mean(const FloatImage& image, int window_cols, int window_rows) {
  const std::vector<double> horizontal(window_cols, 1.0 / window_cols);
  const std::vector<double> vertical(window_rows, 1.0 / window_rows);
  return filter_vertical(filter_horizontal(image, horizontal), vertical);
}

} // namespace

bool LidarImageModel::valid() const {
  return rows > 1 && cols > 0 && static_cast<int>(altitudes_rad.size()) == rows &&
         static_cast<int>(pixel_shift_by_row.size()) == rows && altitudes_rad.front() > altitudes_rad.back();
}

Eigen::Vector3d LidarImageModel::to_lidar_origin(const Eigen::Vector3d& cloud_point) const {
  return cloud_point - Eigen::Vector3d(0.0, 0.0, cloud_to_lidar_z_m);
}

bool LidarImageModel::project(const Eigen::Vector3d& p, Eigen::Vector2d& uv) const {
  const double horizontal = std::hypot(p.x(), p.y()) - beam_offset_m;
  const double distance = std::hypot(horizontal, p.z());
  if (!(distance > 0.0)) {
    return false;
  }
  const double azimuth = std::atan2(p.y(), p.x());
  const double elevation = std::asin(p.z() / distance);
  uv.x() = -static_cast<double>(cols) / kTwoPi * azimuth + static_cast<double>(cols) / 2.0;
  if (elevation > altitudes_rad.front() || elevation < altitudes_rad.back()) {
    return false;
  }
  // Rows are ordered by decreasing elevation: find the beam just above, then interpolate.
  // Last row whose altitude is still >= elevation (altitudes decrease with the row).
  const auto first_below = std::upper_bound(altitudes_rad.begin(), altitudes_rad.end(), elevation,
                                            [](double value, double altitude) { return value > altitude; });
  const int above = static_cast<int>(first_below - altitudes_rad.begin()) - 1;
  if (above + 1 >= rows) {
    uv.y() = rows - 1;
  } else {
    uv.y() = above + (altitudes_rad[above] - elevation) / (altitudes_rad[above] - altitudes_rad[above + 1]);
  }
  return uv.x() >= 0.0 && uv.x() <= cols - 1 && uv.y() >= 0.0 && uv.y() <= rows - 1;
}

Eigen::Matrix<double, 2, 3> LidarImageModel::projection_jacobian(const Eigen::Vector3d& p) const {
  const double fx = -static_cast<double>(cols) / kTwoPi;
  const double fy = -static_cast<double>(rows) / std::abs(altitudes_rad.front() - altitudes_rad.back());
  const double rxy = std::max(std::hypot(p.x(), p.y()), 1e-9);
  const double horizontal = rxy - beam_offset_m;
  const double distance2 = horizontal * horizontal + p.z() * p.z();
  const double fx_irxy2 = fx / (rxy * rxy);
  Eigen::Matrix<double, 2, 3> du_dp;
  du_dp << -fx_irxy2 * p.y(), fx_irxy2 * p.x(), 0.0,
      -fy * p.x() * p.z() / (horizontal * distance2), -fy * p.y() * p.z() / (horizontal * distance2),
      fy * horizontal / distance2;
  return du_dp;
}

int LidarImageModel::cloud_index(int row, int col) const {
  if (row < 0 || row >= rows || col < 0 || col >= cols) {
    return -1;
  }
  const int destaggered = ((col + u_shift) % cols + cols) % cols;
  const int raw_col = ((destaggered - pixel_shift_by_row[row]) % cols + cols) % cols;
  return row * cols + raw_col;
}

std::array<int, 2> LidarImageModel::pixel_of_cloud_index(int index) const {
  const int row = index / cols;
  const int raw_col = index % cols;
  const int destaggered = (raw_col + pixel_shift_by_row[row]) % cols;
  return {row, ((destaggered - u_shift) % cols + cols) % cols};
}

double FloatImage::bilinear(double u, double v) const {
  u = std::clamp(u, 0.0, static_cast<double>(cols - 1));
  v = std::clamp(v, 0.0, static_cast<double>(rows - 1));
  const int u0 = static_cast<int>(u);
  const int v0 = static_cast<int>(v);
  const int u1 = std::min(u0 + 1, cols - 1);
  const int v1 = std::min(v0 + 1, rows - 1);
  const double fu = u - u0;
  const double fv = v - v0;
  return (1 - fu) * (1 - fv) * at(v0, u0) + fu * (1 - fv) * at(v0, u1) + (1 - fu) * fv * at(v1, u0) +
         fu * fv * at(v1, u1);
}

void remove_line_artifacts(FloatImage& image, const std::vector<double>& highpass, const std::vector<double>& lowpass) {
  if (highpass.empty() || lowpass.empty()) {
    return;
  }
  const FloatImage lines = filter_horizontal(filter_vertical(image, highpass), lowpass);
  for (std::size_t i = 0; i < image.data.size(); ++i) {
    image.data[i] = std::max(0.0F, image.data[i] - lines.data[i]);
  }
}

void normalize_brightness(FloatImage& image, int window_cols, int window_rows) {
  const FloatImage brightness = box_mean(image, window_cols, window_rows);
  for (std::size_t i = 0; i < image.data.size(); ++i) {
    image.data[i] = 140.0F * image.data[i] / (brightness.data[i] + 1.0F);
  }
}

PhotometricFrame build_photometric_frame(const LidarImageModel& model,
                                         const IntensityImageConfig& config,
                                         const std::vector<Eigen::Vector3d>& cloud_points,
                                         const std::vector<float>& intensities,
                                         const std::vector<int>& capture_slot,
                                         const std::vector<Sophus::SE3d>& end_from_capture_slots) {
  PhotometricFrame frame;
  const std::size_t n_pixels = static_cast<std::size_t>(model.rows) * model.cols;
  if (!model.valid() || cloud_points.size() != n_pixels || intensities.size() != n_pixels ||
      capture_slot.size() != n_pixels) {
    return frame;
  }
  frame.rows = model.rows;
  frame.cols = model.cols;
  frame.intensity = FloatImage(model.rows, model.cols);
  frame.range = FloatImage(model.rows, model.cols);
  frame.pixel_point.assign(n_pixels, -1);
  frame.points_end.assign(n_pixels, Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()));
  frame.capture_slot = capture_slot;
  frame.capture_from_end.reserve(end_from_capture_slots.size());
  for (const auto& end_from_capture : end_from_capture_slots) {
    frame.capture_from_end.push_back(end_from_capture.inverse());
  }
  frame.buckets.assign(n_pixels, {});
  frame.bucket_count.assign(n_pixels, 0U);

  const float scale = static_cast<float>(config.intensity_scale);
  // Pixel each point was captured at, and the pixel its deskewed point projects to
  // (computed in parallel; the buckets are filled in index order afterwards).
  std::vector<int> bucket_pixel(n_pixels, -1);
  tbb::parallel_for(tbb::blocked_range<int>(0, static_cast<int>(n_pixels)), [&](const tbb::blocked_range<int>& range) {
    for (int index = range.begin(); index != range.end(); ++index) {
      const Eigen::Vector3d& point = cloud_points[index];
      const double range_m = point.norm();
      if (!point.allFinite() || range_m < 1e-3) {
        continue;
      }
      const auto [row, col] = model.pixel_of_cloud_index(index);
      frame.intensity.at(row, col) = intensities[index] * scale;
      frame.range.at(row, col) = static_cast<float>(range_m);
      frame.pixel_point[static_cast<std::size_t>(row) * model.cols + col] = index;
      const int slot = capture_slot[index];
      if (slot < 0 || slot >= static_cast<int>(end_from_capture_slots.size())) {
        continue;
      }
      const Eigen::Vector3d point_end = end_from_capture_slots[slot] * point;
      frame.points_end[index] = point_end;
      Eigen::Vector2d uv;
      if (!model.project(model.to_lidar_origin(point_end), uv)) {
        continue;
      }
      const int bucket_row = static_cast<int>(std::lround(uv.y()));
      const int bucket_col = static_cast<int>(std::lround(uv.x()));
      if (bucket_row >= 0 && bucket_row < model.rows && bucket_col >= 0 && bucket_col < model.cols) {
        bucket_pixel[index] = bucket_row * model.cols + bucket_col;
      }
    }
  });
  for (int index = 0; index < static_cast<int>(n_pixels); ++index) {
    const int pixel = bucket_pixel[index];
    if (pixel >= 0 && frame.bucket_count[pixel] < PhotometricFrame::kBucket) {
      frame.buckets[pixel][frame.bucket_count[pixel]++] = index;
    }
  }

  if (config.line_removal) {
    remove_line_artifacts(frame.intensity, config.line_highpass, config.line_lowpass);
  }
  if (config.brightness_filter) {
    normalize_brightness(frame.intensity, config.brightness_window_cols, config.brightness_window_rows);
  }
  if (config.blur) {
    const std::vector<double> gauss{0.25, 0.5, 0.25};
    frame.intensity = filter_vertical(filter_horizontal(frame.intensity, gauss), gauss);
  }
  for (float& value : frame.intensity.data) {
    value = std::min(value, 255.0F);
  }
  frame.dx = filter_horizontal(frame.intensity, {-0.5, 0.0, 0.5});
  frame.dy = filter_vertical(frame.intensity, {-0.5, 0.0, 0.5});

  // Valid pixels: in range and outside the static masks, eroded so a whole patch fits.
  std::vector<std::uint8_t> mask(n_pixels, 1U);
  for (const auto& [x, y, width, height] : config.masks) {
    for (int r = std::max(0, y); r < std::min(model.rows, y + height); ++r) {
      for (int c = std::max(0, x); c < std::min(model.cols, x + width); ++c) {
        mask[static_cast<std::size_t>(r) * model.cols + c] = 0U;
      }
    }
  }
  for (std::size_t i = 0; i < n_pixels; ++i) {
    const float range = frame.range.data[i];
    if (range < config.min_range_m || range > config.max_range_m) {
      mask[i] = 0U;
    }
  }
  // Square erosion as a row pass then a column pass; outside the image does not
  // erode (OpenCV's default morphology border).
  const int kernel = config.patch_size + config.erosion_margin;
  const int before = kernel / 2;
  const int after = kernel - before - 1;
  std::vector<std::uint8_t> eroded_rows(n_pixels, 1U);
  for_each_row(model.rows, [&](int r) {
    for (int c = 0; c < model.cols; ++c) {
      std::uint8_t keep = 1U;
      for (int cc = std::max(0, c - before); cc <= std::min(model.cols - 1, c + after) && keep != 0U; ++cc) {
        keep = mask[static_cast<std::size_t>(r) * model.cols + cc];
      }
      eroded_rows[static_cast<std::size_t>(r) * model.cols + c] = keep;
    }
  });
  frame.mask.assign(n_pixels, 1U);
  for_each_row(model.rows, [&](int r) {
    for (int c = 0; c < model.cols; ++c) {
      std::uint8_t keep = 1U;
      for (int rr = std::max(0, r - before); rr <= std::min(model.rows - 1, r + after) && keep != 0U; ++rr) {
        keep = eroded_rows[static_cast<std::size_t>(rr) * model.cols + c];
      }
      frame.mask[static_cast<std::size_t>(r) * model.cols + c] = keep;
    }
  });
  return frame;
}

bool project_captured(const PhotometricFrame& frame,
                      const LidarImageModel& model,
                      const Eigen::Vector3d& point_end,
                      const bool round_to_pixel,
                      CapturedProjection& result) {
  Eigen::Vector2d uv_end;
  if (!model.project(model.to_lidar_origin(point_end), uv_end)) {
    return false;
  }
  if (round_to_pixel) {
    uv_end = uv_end.array().round();
  }
  int row = static_cast<int>(uv_end.y());
  const int col = static_cast<int>(uv_end.x());
  if (row < 0 || row >= frame.rows || col < 0 || col >= frame.cols) {
    return false;
  }
  auto bucket_at = [&](int r) { return static_cast<std::size_t>(r) * frame.cols + col; };
  if (frame.bucket_count[bucket_at(row)] == 0U) {
    // As COIN-LIO: fall back to the first populated pixel of the column.
    row = 0;
    while (row < frame.rows && frame.bucket_count[bucket_at(row)] == 0U) {
      ++row;
    }
    if (row >= frame.rows) {
      return false;
    }
  }
  const std::size_t pixel = bucket_at(row);
  int nearest = -1;
  double nearest_distance = std::numeric_limits<double>::max();
  for (int i = 0; i < frame.bucket_count[pixel]; ++i) {
    const int index = frame.buckets[pixel][i];
    const double distance = (point_end - frame.points_end[index]).norm();
    if (distance < nearest_distance) {
      nearest_distance = distance;
      nearest = index;
    }
  }
  if (nearest < 0) {
    return false;
  }
  result.capture_slot = frame.capture_slot[nearest];
  if (result.capture_slot < 0 || result.capture_slot >= static_cast<int>(frame.capture_from_end.size())) {
    return false;
  }
  result.point_captured = frame.capture_from_end[result.capture_slot] * point_end;
  return model.project(model.to_lidar_origin(result.point_captured), result.uv);
}

} // namespace rko_lio::core
