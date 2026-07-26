/*
 * MIT License
 *
 * Copyright (c) 2026 Sasaki
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

// Sliding-window absolute gravity alignment.
//
// The per-scan orientation regularization (build_orientation_linear_system in
// lio.cpp) references the body-acceleration Kalman filter, whose measurement
// is constructed with the *current* rotation estimate. When the orientation
// drifts slowly in pitch/roll, that reference drifts with it, so the
// regularization cannot observe -- let alone correct -- the drift (measured on
// the NTNU tunnel sequence: a 0.5 -> 6.1 deg monotonic world-frame gravity
// tilt that survives even a 40x stronger regularization weight).
//
// This helper instead treats the *long-window* mean of raw (gravity-inclusive)
// accelerometer samples, rotated into the world frame, as an absolute
// reference: over tens of seconds the body-acceleration content of a
// quasi-steady traversal averages out and the mean must point world-up. The
// residual tilt between that mean and +z is fed back as a small, capped,
// roll/pitch-only correction. Yaw is untouched (gravity says nothing about
// it), and the magnitude gate rejects windows contaminated by sustained
// accelerations.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

#include <algorithm>
#include <cmath>

namespace rko_lio::core {

struct GravityAlignmentCorrection {
  bool valid = false;
  double tilt_rad = 0.0;
  // World-frame left-multiplicative correction, identity unless valid.
  Sophus::SO3d correction = Sophus::SO3d();
};

/// Compute a capped roll/pitch correction that rotates `mean_world_accel`
/// (the window mean of raw accelerometer samples expressed in the world
/// frame, expected to point world-up with magnitude ~g) toward +z.
///
/// - `gravity_magnitude`: expected |g| (m/s^2).
/// - `max_magnitude_deviation`: reject the window when
///   | |mean| - g | > max_magnitude_deviation * g (sustained body
///   acceleration or bad data).
/// - `gain`: fraction of the measured tilt corrected per call.
/// - `max_correction_rad`: hard cap on the correction angle per call.
/// - `max_plausible_tilt_rad`: reject the window entirely when the measured
///   tilt exceeds this. The permissive default (45 deg) only rejects clearly
///   nonsensical windows: on sequences with fast genuine attitude error
///   (e.g. fog clutter-lock) large measured tilts still carry real signal,
///   and the per-call cap is the primary safety mechanism. Tighten on rigs
///   where large window tilts are known-impossible.
inline GravityAlignmentCorrection compute_gravity_alignment_correction(const Eigen::Vector3d& mean_world_accel,
                                                                       const double gravity_magnitude,
                                                                       const double max_magnitude_deviation,
                                                                       const double gain,
                                                                       const double max_correction_rad,
                                                                       const double max_plausible_tilt_rad = 0.785) {
  GravityAlignmentCorrection result;
  const double magnitude = mean_world_accel.norm();
  if (!(magnitude > 0.0) || std::abs(magnitude - gravity_magnitude) > max_magnitude_deviation * gravity_magnitude) {
    return result;
  }
  const Eigen::Vector3d up = mean_world_accel / magnitude;
  // axis = up x z: rotating about it by the tilt angle takes `up` onto +z.
  const Eigen::Vector3d axis = up.cross(Eigen::Vector3d::UnitZ());
  const double sin_tilt = axis.norm();
  const double tilt = std::atan2(sin_tilt, up.z());
  result.tilt_rad = tilt;
  if (tilt > max_plausible_tilt_rad) {
    return result;
  }
  if (!(sin_tilt > 0.0) || tilt <= 0.0) {
    result.valid = true; // aligned already; identity correction
    return result;
  }
  const double correction_angle = std::min(std::max(gain, 0.0) * tilt, max_correction_rad);
  result.correction = Sophus::SO3d::exp((axis / sin_tilt) * correction_angle);
  result.valid = true;
  return result;
}

} // namespace rko_lio::core
