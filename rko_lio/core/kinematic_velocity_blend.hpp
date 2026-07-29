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

// Soft, anchor-decayed fusion of ICP and IMU-propagated velocity.

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace rko_lio::core {

inline bool should_bridge_low_speed_with_inertial_activity(
    const double previous_speed_mps,
    const double min_speed_mps,
    const double accel_magnitude_variance,
    const double min_accel_magnitude_variance,
    const bool anchor_active,
    const bool propagated_velocity_available) {
  return previous_speed_mps < std::max(0.0, min_speed_mps) &&
         min_accel_magnitude_variance > 0.0 &&
         std::isfinite(accel_magnitude_variance) &&
         accel_magnitude_variance >= min_accel_magnitude_variance &&
         anchor_active && propagated_velocity_available;
}

inline Eigen::Vector3d rotate_kinematic_velocity_prior(
    const Eigen::Matrix3d& world_rotation_delta,
    const Eigen::Vector3d& velocity_world) {
  if (!world_rotation_delta.allFinite() || !velocity_world.allFinite()) {
    return Eigen::Vector3d::Zero();
  }
  return world_rotation_delta * velocity_world;
}

struct KinematicVelocityBlend {
  /** True when the inputs support either an anchor decision or a blend. */
  bool valid = false;
  /** ICP agrees with propagation closely enough to refresh the trusted anchor. */
  bool anchor_agrees = false;
  /** World-frame translation to add to the optimized ICP pose. */
  Eigen::Vector3d correction = Eigen::Vector3d::Zero();
  /** Propagation contribution along the motion axis, in [0, 1]. */
  double propagation_weight = 0.0;
  /** 3D ICP-versus-propagation velocity innovation norm. */
  double disagreement_mps = 0.0;
};

/**
 * Blend ICP velocity with a short-lived inertial propagation prior.
 *
 * The propagation speed is projected onto the previous direction of travel,
 * making the expected cross-axis velocity zero for one scan. Its information is:
 *
 *   I_pred = scale * exp(-anchor_age / decay_time) * I
 *
 * while ICP uses a fixed isotropic information scale. This deliberately avoids
 * the point-to-point Hessian's misleading N*I translation block. If ICP agrees
 * with propagation, the caller should refresh the anchor and no correction is
 * applied. During disagreement, propagation bridges the weak direction softly;
 * its authority decays until ICP takes over again.
 */
inline KinematicVelocityBlend blend_icp_with_propagated_velocity(
    const Eigen::Vector3d& motion_axis_world,
    const Eigen::Vector3d& icp_translation_step_world,
    const double dt,
    const Eigen::Vector3d& propagated_velocity_world,
    const double anchor_age_sec,
    const double icp_information_scale,
    const double propagation_information_scale,
    const double propagation_decay_time_sec,
    const double anchor_agreement_mps,
    const double max_icp_innovation_mps,
    const double min_speed) {
  KinematicVelocityBlend result;
  if (!(dt > 0.0) || !motion_axis_world.allFinite() || !icp_translation_step_world.allFinite() ||
      !propagated_velocity_world.allFinite() || !std::isfinite(anchor_age_sec) ||
      motion_axis_world.norm() < std::max(0.0, min_speed)) {
    return result;
  }

  const Eigen::Vector3d axis = motion_axis_world.normalized();
  const Eigen::Vector3d icp_velocity_world = icp_translation_step_world / dt;
  const Eigen::Vector3d propagation_prior_world =
      axis * axis.dot(propagated_velocity_world);
  const Eigen::Vector3d icp_innovation_world =
      icp_velocity_world - propagation_prior_world;
  result.valid = true;
  result.disagreement_mps = icp_innovation_world.norm();
  result.anchor_agrees = result.disagreement_mps <= std::max(0.0, anchor_agreement_mps);
  if (result.anchor_agrees) {
    return result;
  }

  if (!(icp_information_scale > 0.0) || !(propagation_information_scale > 0.0) ||
      !(propagation_decay_time_sec > 0.0) || anchor_age_sec < 0.0) {
    return result;
  }
  const double decayed_propagation_information =
      propagation_information_scale * std::exp(-anchor_age_sec / propagation_decay_time_sec);
  result.propagation_weight =
      decayed_propagation_information / (icp_information_scale + decayed_propagation_information);
  Eigen::Vector3d robust_icp_innovation_world = icp_innovation_world;
  if (max_icp_innovation_mps > 0.0 &&
      robust_icp_innovation_world.norm() > max_icp_innovation_mps) {
    robust_icp_innovation_world *=
        max_icp_innovation_mps / robust_icp_innovation_world.norm();
  }
  const Eigen::Vector3d fused_velocity_world =
      propagation_prior_world +
      (1.0 - result.propagation_weight) * robust_icp_innovation_world;
  // Correction is relative to the raw ICP velocity, not the robustified one.
  result.correction = (fused_velocity_world - icp_velocity_world) * dt;
  return result;
}

} // namespace rko_lio::core
