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
#include <optional>
#include <vector>

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

/** One IMU interval between two registered scans: midpoint time (s), length (s), and the mean raw
 *  (gravity-inclusive) accelerometer sample rotated into the world frame. */
struct WorldSpecificForceInterval {
  double mid_time = 0.0;
  double duration = 0.0;
  Eigen::Vector3d specific_force = Eigen::Vector3d::Zero();
};

/** World-frame up vector times |g| (m/s^2) from the specific force measured while moving between
 *  `begin` and `end` (s), with the motion's velocity change removed. The velocity at each end is the
 *  mean over end +- half_span from the registered positions (`position_at(t)`); the specific force is
 *  weighted by the matching ramp (rising over begin +- half_span, falling over end +- half_span), so
 *     sum(w f dt) = up |g| (end - begin) + V(end) - V(begin)
 *  holds for any motion, up to the discretization of the intervals. A start while walking leaves
 *  the 0.1-s initialization average far from gravity; this needs no standstill. Returns nullopt
 *  when the intervals do not cover [begin - half_span, end + half_span]. */
template <typename PositionAt>
std::optional<Eigen::Vector3d> estimate_world_up_while_moving(const std::vector<WorldSpecificForceInterval>& intervals,
                                                              const PositionAt& position_at,
                                                              const double begin,
                                                              const double end,
                                                              const double half_span) {
  if (!(end - begin >= 2.0 * half_span) || !(half_span > 0.0) || intervals.empty()) return std::nullopt;
  const double first = intervals.front().mid_time - 0.5 * intervals.front().duration;
  const double last = intervals.back().mid_time + 0.5 * intervals.back().duration;
  if (first > begin - half_span || last < end + half_span) return std::nullopt;
  Eigen::Vector3d weighted_force = Eigen::Vector3d::Zero();
  double weight_sum = 0.0;
  for (const auto& interval : intervals) {
    const double t = interval.mid_time;
    const double w = std::clamp(std::min(t - (begin - half_span), (end + half_span) - t) / (2.0 * half_span), 0.0, 1.0);
    weighted_force += w * interval.duration * interval.specific_force;
    weight_sum += w * interval.duration;
  }
  if (!(weight_sum > 0.0)) return std::nullopt;
  const Eigen::Vector3d velocity_begin =
      (position_at(begin + half_span) - position_at(begin - half_span)) / (2.0 * half_span);
  const Eigen::Vector3d velocity_end = (position_at(end + half_span) - position_at(end - half_span)) / (2.0 * half_span);
  return Eigen::Vector3d((weighted_force - (velocity_end - velocity_begin)) / weight_sum);
}

/** Registered attitude with its roll/pitch correction of the predicted attitude limited to
 *  cap_rad. The correction is taken in the world frame; its vertical (yaw) part is kept.
 *  Sets *limited when the cap applied. */
inline Sophus::SO3d limit_tilt_correction(const Sophus::SO3d& predicted, const Sophus::SO3d& registered,
                                          const double cap_rad, bool* limited = nullptr) {
  const Eigen::Vector3d correction = (registered * predicted.inverse()).log();
  Eigen::Vector3d tilt(correction.x(), correction.y(), 0.0);
  if (limited) *limited = false;
  if (cap_rad <= 0.0 || tilt.norm() <= cap_rad) return registered;
  tilt *= cap_rad / tilt.norm();
  if (limited) *limited = true;
  return Sophus::SO3d::exp(Eigen::Vector3d(tilt.x(), tilt.y(), correction.z())) * predicted;
}

} // namespace rko_lio::core
