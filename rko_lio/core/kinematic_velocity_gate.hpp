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

// Accelerometer-consistency velocity gate for geometric zero-motion lock-in.
//
// In a self-similar tunnel the point-to-point ICP has a hard zero-motion
// attractor: once the estimate slips behind, nearest-neighbor associations
// confirm "no motion" and the velocity estimate collapses within a few scans
// (measured on the NTNU tunnel: 160 s frozen at reach 42 m while the platform
// travelled 340 m). The implied deceleration of such a collapse (~17 m/s^2 at
// walking pace over one 0.1 s scan) is physically impossible and directly
// contradicted by the accelerometer, which sees no braking. This gate clamps
// the per-scan velocity change along the previous motion direction to what the
// measured body acceleration (plus a margin) allows -- a symmetric rate
// limiter, so runaway forward jumps are clamped the same way. Genuine stops
// pass through because braking shows up in the accelerometer and raises the
// allowed change. Pure functions only, unit-testable in isolation like
// gravity_alignment.hpp / localizability_weighting.hpp.

#pragma once

#include <Eigen/Core>

#include <cmath>

namespace rko_lio::core {

struct KinematicVelocityClamp {
  /** True if the ICP translation violated the accelerometer bound and a correction applies. */
  bool corrected = false;
  /** World-frame translation to ADD to the ICP pose so the implied velocity respects the bound. */
  Eigen::Vector3d correction = Eigen::Vector3d::Zero();
  /** |dv|/dt the ICP result implied along the previous motion direction (diagnostic). */
  double implied_accel = 0.0;
  /** The accelerometer-derived bound it was checked against (diagnostic). */
  double allowed_accel = 0.0;
};

/**
 * Clamp the per-scan velocity change along the previous motion direction to the
 * accelerometer-consistent bound.
 *
 * `previous_velocity_world`: the world-frame velocity estimate before this scan.
 * `translation_step_world`: optimized_pose.translation() - previous_pose.translation().
 * `dt`: elapsed time between the scans (s).
 * `measured_accel_world`: gravity-compensated body acceleration measured over the
 *   interval, rotated into the world frame (m/s^2). Only its projection onto the
 *   motion direction bounds the along-axis speed change: walking/driving vibration
 *   is mostly perpendicular to travel and must not loosen the bound, while genuine
 *   braking projects strongly onto the axis and opens it.
 * `accel_margin`: slack added to the projected magnitude (m/s^2) covering bias and
 *   accelerometer noise.
 * `min_speed`: below this previous speed (m/s) the motion direction is untrustworthy
 *   and the gate stays out of the way (also lets the rig start from rest).
 * `weight`: fraction of the computed clamp correction actually applied (1 = full).
 */
inline KinematicVelocityClamp clamp_velocity_change_to_accel(const Eigen::Vector3d& previous_velocity_world,
                                                             const Eigen::Vector3d& translation_step_world,
                                                             const double dt,
                                                             const Eigen::Vector3d& measured_accel_world,
                                                             const double accel_margin,
                                                             const double min_speed,
                                                             const double weight) {
  KinematicVelocityClamp result;
  const double previous_speed = previous_velocity_world.norm();
  if (!(dt > 0.0) || previous_speed < min_speed || !(weight > 0.0)) {
    return result;
  }
  const Eigen::Vector3d axis = previous_velocity_world / previous_speed;
  const double icp_speed_along_axis = axis.dot(translation_step_world) / dt;
  const double speed_change = icp_speed_along_axis - previous_speed;
  result.implied_accel = std::abs(speed_change) / dt;
  result.allowed_accel = std::abs(axis.dot(measured_accel_world)) + std::max(0.0, accel_margin);
  const double allowed_change = result.allowed_accel * dt;
  if (std::abs(speed_change) <= allowed_change) {
    return result;
  }
  const double clamped_speed =
      previous_speed + (speed_change > 0.0 ? allowed_change : -allowed_change);
  result.corrected = true;
  result.correction = weight * (clamped_speed - icp_speed_along_axis) * dt * axis;
  return result;
}

} // namespace rko_lio::core
