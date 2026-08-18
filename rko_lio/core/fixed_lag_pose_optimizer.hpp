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
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

namespace rko_lio::core {

struct FixedLagRelativeConstraint {
  std::size_t from = 0;
  std::size_t to = 0;
  Sophus::SE3d from_to;
  double weight = 1.0;
};

struct FixedLagPoseOptimizerConfig {
  std::size_t max_iterations = 8;
  double finite_difference_step = 1.0e-6;
  double convergence_step_norm = 1.0e-7;
  double diagonal_damping = 1.0e-6;
  double huber_delta_m = 0.25;
  double rotation_scale_m_per_rad = 1.0;
  bool fix_latest_pose = false;
};

struct FixedLagPoseOptimizerResult {
  std::vector<Sophus::SE3d> poses;
  double initial_cost = std::numeric_limits<double>::infinity();
  double final_cost = std::numeric_limits<double>::infinity();
  std::size_t iterations = 0;
  bool valid = false;
};

inline bool finite_pose(const Sophus::SE3d& pose) {
  return pose.matrix().allFinite();
}

inline Eigen::Matrix<double, 6, 1> fixed_lag_constraint_residual(
    const std::vector<Sophus::SE3d>& poses,
    const FixedLagRelativeConstraint& constraint,
    const double rotation_scale_m_per_rad) {
  const Sophus::SE3d predicted =
      poses[constraint.from].inverse() * poses[constraint.to];
  Eigen::Matrix<double, 6, 1> residual =
      (constraint.from_to.inverse() * predicted).log();
  residual.tail<3>() *= rotation_scale_m_per_rad;
  return residual;
}

inline double fixed_lag_huber_cost(
    const double residual_norm,
    const double delta,
    const double weight) {
  if (residual_norm <= delta) {
    return 0.5 * weight * residual_norm * residual_norm;
  }
  return weight * delta * (residual_norm - 0.5 * delta);
}

inline double fixed_lag_pose_cost(
    const std::vector<Sophus::SE3d>& poses,
    const std::vector<FixedLagRelativeConstraint>& constraints,
    const FixedLagPoseOptimizerConfig& config) {
  double cost = 0.0;
  for (const FixedLagRelativeConstraint& constraint : constraints) {
    const double norm = fixed_lag_constraint_residual(
        poses, constraint, config.rotation_scale_m_per_rad).norm();
    cost += fixed_lag_huber_cost(norm, config.huber_delta_m, constraint.weight);
  }
  return cost;
}

inline FixedLagPoseOptimizerResult optimize_fixed_lag_poses(
    const std::vector<Sophus::SE3d>& initial_poses,
    const std::vector<FixedLagRelativeConstraint>& constraints,
    const FixedLagPoseOptimizerConfig& config = {}) {
  FixedLagPoseOptimizerResult result;
  result.poses = initial_poses;
  if (initial_poses.empty() || config.max_iterations == 0U ||
      !std::isfinite(config.finite_difference_step) ||
      config.finite_difference_step <= 0.0 ||
      !std::isfinite(config.convergence_step_norm) ||
      config.convergence_step_norm < 0.0 ||
      !std::isfinite(config.diagonal_damping) ||
      config.diagonal_damping < 0.0 ||
      !std::isfinite(config.huber_delta_m) || config.huber_delta_m <= 0.0 ||
      !std::isfinite(config.rotation_scale_m_per_rad) ||
      config.rotation_scale_m_per_rad <= 0.0) {
    return result;
  }
  if (!std::all_of(initial_poses.begin(), initial_poses.end(), finite_pose)) {
    return result;
  }
  for (const FixedLagRelativeConstraint& constraint : constraints) {
    if (constraint.from >= initial_poses.size() ||
        constraint.to >= initial_poses.size() ||
        constraint.from == constraint.to || !finite_pose(constraint.from_to) ||
        !std::isfinite(constraint.weight) || constraint.weight <= 0.0) {
      return result;
    }
  }

  result.initial_cost = fixed_lag_pose_cost(result.poses, constraints, config);
  result.final_cost = result.initial_cost;
  if (initial_poses.size() == 1U || constraints.empty()) {
    result.valid = true;
    return result;
  }

  std::vector<Eigen::Index> variable_offsets(
      initial_poses.size(), Eigen::Index{-1});
  std::size_t variable_pose_count = 0U;
  for (std::size_t pose_index = 1U; pose_index < initial_poses.size(); ++pose_index) {
    if (config.fix_latest_pose && pose_index + 1U == initial_poses.size()) {
      continue;
    }
    variable_offsets[pose_index] =
        static_cast<Eigen::Index>(6U * variable_pose_count++);
  }
  if (variable_pose_count == 0U) {
    result.valid = true;
    return result;
  }
  const Eigen::Index dimension = static_cast<Eigen::Index>(6U * variable_pose_count);
  for (std::size_t iteration = 0; iteration < config.max_iterations; ++iteration) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dimension, dimension);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(dimension);

    for (const FixedLagRelativeConstraint& constraint : constraints) {
      const Eigen::Matrix<double, 6, 1> residual = fixed_lag_constraint_residual(
          result.poses, constraint, config.rotation_scale_m_per_rad);
      const double residual_norm = residual.norm();
      const double robust_weight = constraint.weight *
          std::min(1.0, config.huber_delta_m / std::max(residual_norm, 1.0e-15));

      Eigen::Matrix<double, 6, 6> J_from =
          Eigen::Matrix<double, 6, 6>::Zero();
      Eigen::Matrix<double, 6, 6> J_to =
          Eigen::Matrix<double, 6, 6>::Zero();
      for (Eigen::Index axis = 0; axis < 6; ++axis) {
        Eigen::Matrix<double, 6, 1> perturbation =
            Eigen::Matrix<double, 6, 1>::Zero();
        perturbation[axis] = config.finite_difference_step;
        if (variable_offsets[constraint.from] >= 0) {
          std::vector<Sophus::SE3d> perturbed = result.poses;
          perturbed[constraint.from] =
              Sophus::SE3d::exp(perturbation) * perturbed[constraint.from];
          J_from.col(axis) = (fixed_lag_constraint_residual(
              perturbed, constraint, config.rotation_scale_m_per_rad) - residual) /
              config.finite_difference_step;
        }
        if (variable_offsets[constraint.to] >= 0) {
          std::vector<Sophus::SE3d> perturbed = result.poses;
          perturbed[constraint.to] =
              Sophus::SE3d::exp(perturbation) * perturbed[constraint.to];
          J_to.col(axis) = (fixed_lag_constraint_residual(
              perturbed, constraint, config.rotation_scale_m_per_rad) - residual) /
              config.finite_difference_step;
        }
      }

      auto accumulate_diagonal = [&](const std::size_t pose_index,
                                     const Eigen::Matrix<double, 6, 6>& J) {
        const Eigen::Index offset = variable_offsets[pose_index];
        if (offset < 0) {
          return;
        }
        H.block<6, 6>(offset, offset).noalias() += robust_weight * J.transpose() * J;
        b.segment<6>(offset).noalias() += robust_weight * J.transpose() * residual;
      };
      accumulate_diagonal(constraint.from, J_from);
      accumulate_diagonal(constraint.to, J_to);
      if (variable_offsets[constraint.from] >= 0 &&
          variable_offsets[constraint.to] >= 0) {
        const Eigen::Index from_offset = variable_offsets[constraint.from];
        const Eigen::Index to_offset = variable_offsets[constraint.to];
        const Eigen::Matrix<double, 6, 6> cross =
            robust_weight * J_from.transpose() * J_to;
        H.block<6, 6>(from_offset, to_offset).noalias() += cross;
        H.block<6, 6>(to_offset, from_offset).noalias() += cross.transpose();
      }
    }

    H.diagonal().array() += config.diagonal_damping;
    const Eigen::VectorXd update = H.ldlt().solve(-b);
    if (!update.allFinite()) {
      return result;
    }
    if (update.norm() <= config.convergence_step_norm) {
      result.iterations = iteration;
      result.valid = true;
      return result;
    }

    bool accepted = false;
    double step_scale = 1.0;
    for (int line_search = 0; line_search < 8; ++line_search) {
      std::vector<Sophus::SE3d> candidate = result.poses;
      for (std::size_t pose_index = 1U; pose_index < candidate.size(); ++pose_index) {
        const Eigen::Index offset = variable_offsets[pose_index];
        if (offset < 0) {
          continue;
        }
        candidate[pose_index] = Sophus::SE3d::exp(
            step_scale * update.segment<6>(offset)) * candidate[pose_index];
      }
      const double candidate_cost = fixed_lag_pose_cost(candidate, constraints, config);
      if (std::isfinite(candidate_cost) && candidate_cost <= result.final_cost) {
        result.poses = std::move(candidate);
        result.final_cost = candidate_cost;
        accepted = true;
        break;
      }
      step_scale *= 0.5;
    }
    result.iterations = iteration + 1U;
    if (!accepted) {
      result.valid = true;
      return result;
    }
    if (step_scale * update.norm() <= config.convergence_step_norm) {
      result.valid = true;
      return result;
    }
  }
  result.valid = true;
  return result;
}

}  // namespace rko_lio::core
