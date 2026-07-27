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

// Range-distribution guard for the straight-tunnel inertial velocity bridge.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace rko_lio::core {

struct KinematicSceneRangeSupport {
  bool valid = false;
  bool trusted = false;
  std::size_t valid_point_count = 0;
  double near_fraction = 0.0;
  double far_fraction = 0.0;
};

struct KinematicPersistentSpeedGate {
  std::size_t streak = 0;
  bool rejected = false;
};

struct KinematicSceneReenableGate {
  double last_rejected_time_sec = -1.0;
  bool rejected = false;
  bool cooldown_active = false;
};

inline KinematicPersistentSpeedGate update_kinematic_persistent_speed_gate(
    const double speed_mps,
    const double max_activation_speed_mps,
    const std::size_t min_consecutive_scans,
    const std::size_t previous_streak) {
  if (!(max_activation_speed_mps > 0.0)) {
    return {};
  }
  const bool exceeded =
      !std::isfinite(speed_mps) || speed_mps > max_activation_speed_mps;
  const std::size_t streak =
      exceeded && previous_streak < std::numeric_limits<std::size_t>::max()
          ? previous_streak + 1
          : (exceeded ? previous_streak : 0U);
  return {
      .streak = streak,
      .rejected = streak >= std::max<std::size_t>(1, min_consecutive_scans),
  };
}

inline KinematicSceneReenableGate update_kinematic_scene_reenable_gate(
    const bool raw_scene_trusted,
    const double current_time_sec,
    const double reenable_delay_sec,
    const double previous_rejected_time_sec) {
  if (!std::isfinite(current_time_sec)) {
    return {
        .last_rejected_time_sec = current_time_sec,
        .rejected = true,
        .cooldown_active = false,
    };
  }
  if (!raw_scene_trusted) {
    return {
        .last_rejected_time_sec = current_time_sec,
        .rejected = true,
        .cooldown_active = false,
    };
  }
  const bool cooldown_active =
      reenable_delay_sec > 0.0 && previous_rejected_time_sec >= 0.0 &&
      current_time_sec - previous_rejected_time_sec < reenable_delay_sec;
  return {
      .last_rejected_time_sec = previous_rejected_time_sec,
      .rejected = cooldown_active,
      .cooldown_active = cooldown_active,
  };
}

inline KinematicSceneReenableGate update_kinematic_speed_reenable_gate(
    const bool persistent_speed_within_envelope,
    const double current_time_sec,
    const double reenable_delay_sec,
    const double previous_rejected_time_sec) {
  return update_kinematic_scene_reenable_gate(
      persistent_speed_within_envelope, current_time_sec, reenable_delay_sec,
      previous_rejected_time_sec);
}

/**
 * Decide whether a scan contains both near structure and persistent long-range
 * support, as expected in the validated walking tunnel.
 *
 * Dense fog/clutter lock is dominated by returns close to the sensor and loses
 * almost all long-range structure. A close-range handheld corridor has the
 * same conservative outcome. This is only an applicability guard: it does not
 * claim that a passing scan is geometrically localizable.
 */
template <typename PointRange>
KinematicSceneRangeSupport evaluate_kinematic_scene_range_support(
    const PointRange& points,
    const double min_valid_range_m,
    const double max_valid_range_m,
    const double near_range_m,
    const double max_near_fraction,
    const double far_range_m,
    const double min_far_fraction,
    const std::size_t min_valid_points) {
  KinematicSceneRangeSupport result;
  if (!(min_valid_range_m >= 0.0) || !(max_valid_range_m > min_valid_range_m) ||
      !(near_range_m > min_valid_range_m) || !(far_range_m > near_range_m) ||
      !(far_range_m < max_valid_range_m) || !std::isfinite(max_near_fraction) ||
      !std::isfinite(min_far_fraction)) {
    return result;
  }

  std::size_t near_count = 0;
  std::size_t far_count = 0;
  for (const auto& point : points) {
    if (!point.allFinite()) {
      continue;
    }
    const double range = point.norm();
    if (!(range > min_valid_range_m) || !(range < max_valid_range_m)) {
      continue;
    }
    ++result.valid_point_count;
    near_count += range < near_range_m ? 1U : 0U;
    far_count += range > far_range_m ? 1U : 0U;
  }
  if (result.valid_point_count < std::max<std::size_t>(1, min_valid_points)) {
    return result;
  }

  result.valid = true;
  result.near_fraction =
      static_cast<double>(near_count) / static_cast<double>(result.valid_point_count);
  result.far_fraction =
      static_cast<double>(far_count) / static_cast<double>(result.valid_point_count);
  result.trusted =
      result.near_fraction <= std::clamp(max_near_fraction, 0.0, 1.0) &&
      result.far_fraction >= std::clamp(min_far_fraction, 0.0, 1.0);
  return result;
}

} // namespace rko_lio::core
