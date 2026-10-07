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

// Intensity images of an organized spinning LiDAR (Ouster layout) for
// photometric registration.
//
// Follows COIN-LIO (Pfreundschuh et al., "COIN-LIO: Complementary
// Intensity-Augmented LiDAR Inertial Odometry", ICRA 2024; reference code
// https://github.com/ethz-asl/COIN-LIO, BSD-3-Clause, Copyright (c) 2024
// Patrick Pfreundschuh): the beam projection model, the line-artifact and
// brightness filters, and the lookup that maps a deskewed point back to the
// pixel it was captured at. Pure functions and plain buffers only, no OpenCV.

#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace rko_lio::core {

/** Beam geometry of an organized spinning LiDAR, from the sensor metadata. */
struct LidarImageModel {
  int rows = 0;
  int cols = 0;
  /** Beam elevation per row in radians, row 0 the highest beam (decreasing). */
  std::vector<double> altitudes_rad;
  /** Ouster `pixel_shift_by_row`: destaggers the organized cloud columns. */
  std::vector<int> pixel_shift_by_row;
  /** Ouster `lidar_origin_to_beam_origin_mm`, in metres. */
  double beam_offset_m = 0.0;
  /** Column shift applied to the destaggered image. */
  int u_shift = 0;
  /** Height of the lidar origin above the point cloud frame origin (Ouster os_sensor: 0.03617 m). */
  double cloud_to_lidar_z_m = 0.0;

  bool valid() const;
  /** Lidar-origin frame point of a cloud-frame point. */
  Eigen::Vector3d to_lidar_origin(const Eigen::Vector3d& cloud_point) const;
  /** Sub-pixel (u = column, v = row) of a lidar-origin point; false outside the image. */
  bool project(const Eigen::Vector3d& lidar_point, Eigen::Vector2d& uv) const;
  /** d(u, v) / d(point) of project(), with the linearized vertical scale. */
  Eigen::Matrix<double, 2, 3> projection_jacobian(const Eigen::Vector3d& lidar_point) const;
  /** Index of the organized cloud point shown at pixel (row, col), or -1. */
  int cloud_index(int row, int col) const;
  /** Pixel (row, col) at which an organized cloud index is shown. */
  std::array<int, 2> pixel_of_cloud_index(int index) const;
};

/** Image filtering, as tuned for COIN-LIO's Ouster OS0-128 configuration. */
struct IntensityImageConfig {
  /** Multiplier applied to the raw intensity before filtering. */
  double intensity_scale = 0.25;
  /** Vertical high-pass and horizontal low-pass FIR kernels of the line filter. */
  bool line_removal = true;
  std::vector<double> line_highpass;
  std::vector<double> line_lowpass;
  /** Normalize by the local mean brightness over a window (cols x rows). */
  bool brightness_filter = true;
  int brightness_window_cols = 41;
  int brightness_window_rows = 7;
  bool blur = true;
  /** Pixels whose range lies outside [min, max] are masked out. */
  double min_range_m = 0.7;
  double max_range_m = 30.0;
  /** Masked area grows by (patch_size + erosion_margin) to keep whole patches valid. */
  int patch_size = 5;
  int erosion_margin = 2;
  /** Static masks (x, y, width, height), e.g. the handle and the connector. */
  std::vector<std::array<int, 4>> masks;
};

/** Plain row-major float image. */
struct FloatImage {
  int rows = 0;
  int cols = 0;
  std::vector<float> data;

  FloatImage() = default;
  FloatImage(int rows_, int cols_, float value = 0.0F)
      : rows(rows_), cols(cols_), data(static_cast<std::size_t>(rows_) * cols_, value) {}
  float& at(int row, int col) { return data[static_cast<std::size_t>(row) * cols + col]; }
  float at(int row, int col) const { return data[static_cast<std::size_t>(row) * cols + col]; }
  /** Bilinear value at (u = column, v = row), clamped to the image. */
  double bilinear(double u, double v) const;
};

/** One scan rendered as a filtered intensity image plus the lookups photometric registration needs. */
struct PhotometricFrame {
  int rows = 0;
  int cols = 0;
  FloatImage intensity;
  FloatImage dx;
  FloatImage dy;
  FloatImage range;
  std::vector<std::uint8_t> mask;
  /** Organized cloud index shown at each pixel, -1 where empty. */
  std::vector<int> pixel_point;
  /** Deskewed cloud-frame point per cloud index (scan end frame); NaN when invalid. */
  std::vector<Eigen::Vector3d> points_end;
  /** Cloud-frame transform from the scan end frame to each point's capture frame, by slot. */
  std::vector<int> capture_slot;
  std::vector<Sophus::SE3d> capture_from_end;
  /** Up to kBucket cloud indices whose deskewed point projects to each pixel. */
  static constexpr int kBucket = 10;
  std::vector<std::array<int, kBucket>> buckets;
  std::vector<std::uint8_t> bucket_count;

  bool valid() const { return rows > 0 && cols > 0; }
  bool masked(int row, int col) const { return mask[static_cast<std::size_t>(row) * cols + col] == 0U; }
};

/** Where the sensor actually saw a deskewed point: its capture-frame position and pixel. */
struct CapturedProjection {
  /** Cloud-frame point in the frame it was captured in. */
  Eigen::Vector3d point_captured;
  Eigen::Vector2d uv;
  int capture_slot = -1;
};

/** Vertical high-pass then horizontal low-pass, subtracted from the image (COIN-LIO line removal). */
void remove_line_artifacts(FloatImage& image, const std::vector<double>& highpass, const std::vector<double>& lowpass);
/** image <- 140 * image / (1 + local mean over a cols x rows window). */
void normalize_brightness(FloatImage& image, int window_cols, int window_rows);

/**
 * Build the photometric frame of one scan.
 *
 * @param cloud_points   organized cloud in the cloud frame (raw, not deskewed), rows*cols points
 * @param intensities    raw intensity per cloud point
 * @param capture_slot   per cloud point, index into `end_from_capture_slots` (its capture time)
 * @param end_from_capture_slots cloud-frame transforms from a capture frame to the scan end frame
 */
PhotometricFrame build_photometric_frame(const LidarImageModel& model,
                                         const IntensityImageConfig& config,
                                         const std::vector<Eigen::Vector3d>& cloud_points,
                                         const std::vector<float>& intensities,
                                         const std::vector<int>& capture_slot,
                                         const std::vector<Sophus::SE3d>& end_from_capture_slots);

/**
 * Where a cloud-frame point given in the scan end frame was seen in the captured image.
 * The nearest deskewed point projecting to the same pixel supplies the capture time.
 */
bool project_captured(const PhotometricFrame& frame,
                      const LidarImageModel& model,
                      const Eigen::Vector3d& point_end,
                      bool round_to_pixel,
                      CapturedProjection& result);

} // namespace rko_lio::core
