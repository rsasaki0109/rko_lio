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

/** Scan-local prefix integrator for piecewise-constant angular velocity.
 *
 * The legacy helper below walks every IMU sample for every LiDAR point.  This
 * object performs the identical ordered SO(3) products once at IMU boundaries;
 * a point lookup then needs one binary search and one final exponential.
 */
class PiecewiseAngularVelocityIntegrator {
 public:
  PiecewiseAngularVelocityIntegrator(
      const std::vector<TimedAngularVelocity>& samples,
      const Secondsd start_time,
      const Eigen::Vector3d& fallback_angular_velocity)
      : samples_(samples),
        start_time_(start_time),
        initial_angular_velocity_(fallback_angular_velocity) {
    for (const TimedAngularVelocity& sample : samples_) {
      if (sample.time <= start_time_) {
        initial_angular_velocity_ = sample.angular_velocity;
      }
    }
    monotonic_ = std::is_sorted(
        samples_.cbegin(), samples_.cend(),
        [](const TimedAngularVelocity& lhs, const TimedAngularVelocity& rhs) {
          return lhs.time < rhs.time;
        });
    if (!monotonic_) {
      return;
    }

    Sophus::SO3d rotation;
    Secondsd integrated_until = start_time_;
    Eigen::Vector3d angular_velocity = initial_angular_velocity_;
    for (const TimedAngularVelocity& sample : samples_) {
      if (sample.time <= start_time_) {
        continue;
      }
      const double dt = (sample.time - integrated_until).count();
      rotation = rotation * Sophus::SO3d::exp(
          angular_velocity * std::max(0.0, dt));
      integrated_until = sample.time;
      angular_velocity = sample.angular_velocity;
      boundary_times_.push_back(sample.time);
      rotations_after_boundary_.push_back(rotation);
      angular_velocities_after_boundary_.push_back(angular_velocity);
    }
  }

  Sophus::SO3d integrateUntil(const Secondsd end_time) const {
    if (end_time < start_time_) {
      throw std::invalid_argument("gyro integration end time precedes start time");
    }
    if (!monotonic_) {
      Sophus::SO3d rotation;
      Secondsd integrated_until = start_time_;
      Eigen::Vector3d angular_velocity = initial_angular_velocity_;
      for (const TimedAngularVelocity& sample : samples_) {
        if (sample.time <= start_time_) {
          angular_velocity = sample.angular_velocity;
          continue;
        }
        if (sample.time >= end_time) {
          break;
        }
        const double dt = (sample.time - integrated_until).count();
        rotation = rotation * Sophus::SO3d::exp(
            angular_velocity * std::max(0.0, dt));
        integrated_until = sample.time;
        angular_velocity = sample.angular_velocity;
      }
      const double final_dt = (end_time - integrated_until).count();
      return rotation * Sophus::SO3d::exp(
          angular_velocity * std::max(0.0, final_dt));
    }

    const auto boundary = std::lower_bound(
        boundary_times_.cbegin(), boundary_times_.cend(), end_time);
    const std::size_t processed = static_cast<std::size_t>(
        std::distance(boundary_times_.cbegin(), boundary));
    Sophus::SO3d rotation;
    Secondsd integrated_until = start_time_;
    Eigen::Vector3d angular_velocity = initial_angular_velocity_;
    if (processed > 0U) {
      const std::size_t last = processed - 1U;
      rotation = rotations_after_boundary_[last];
      integrated_until = boundary_times_[last];
      angular_velocity = angular_velocities_after_boundary_[last];
    }
    const double final_dt = (end_time - integrated_until).count();
    return rotation * Sophus::SO3d::exp(
        angular_velocity * std::max(0.0, final_dt));
  }

 private:
  std::vector<TimedAngularVelocity> samples_;
  bool monotonic_ = true;
  Secondsd start_time_{0.0};
  Eigen::Vector3d initial_angular_velocity_ = Eigen::Vector3d::Zero();
  std::vector<Secondsd> boundary_times_;
  std::vector<Sophus::SO3d> rotations_after_boundary_;
  std::vector<Eigen::Vector3d> angular_velocities_after_boundary_;
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
