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

#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <sophus/so3.hpp>

#include "util.hpp"

namespace rko_lio::core {

struct TimedAngularVelocity {
  Secondsd time{0.0};
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
};

inline Sophus::SO3d integrate_piecewise_angular_velocity(
    const std::vector<TimedAngularVelocity>& samples,
    const Secondsd start_time,
    const Secondsd end_time,
    const Eigen::Vector3d& fallback_angular_velocity) {
  if (end_time < start_time) {
    throw std::invalid_argument("gyro integration end time precedes start time");
  }

  Sophus::SO3d rotation;
  Secondsd integrated_until = start_time;
  Eigen::Vector3d angular_velocity = fallback_angular_velocity;
  for (const TimedAngularVelocity& sample : samples) {
    if (sample.time <= start_time) {
      angular_velocity = sample.angular_velocity;
      continue;
    }
    if (sample.time >= end_time) {
      break;
    }
    const double dt = (sample.time - integrated_until).count();
    rotation = rotation * Sophus::SO3d::exp(angular_velocity * std::max(0.0, dt));
    integrated_until = sample.time;
    angular_velocity = sample.angular_velocity;
  }
  const double final_dt = (end_time - integrated_until).count();
  return rotation * Sophus::SO3d::exp(angular_velocity * std::max(0.0, final_dt));
}

}  // namespace rko_lio::core
