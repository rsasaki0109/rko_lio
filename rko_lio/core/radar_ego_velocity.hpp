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

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <vector>

namespace rko_lio::core {

/** One radar point's unit line-of-sight direction (sensor frame) and reported radial Doppler speed (m/s). */
struct RadarDopplerMeasurement {
  Eigen::Vector3d direction = Eigen::Vector3d::Zero();
  double doppler_velocity = 0.0;
};

struct RadarEgoVelocityConfig {
  std::size_t ransac_iterations = 100;
  double ransac_inlier_threshold = 0.15;
  std::size_t min_points = 3;
  std::size_t min_inliers = 8;
  /** Sign convention relating radial Doppler to ego velocity: v_r_i = doppler_sign * (d_i . v_ego). */
  double doppler_sign = -1.0;
  /** Fixed so identical scans reproduce identical minimal samples (this repo enforces determinism). */
  std::uint64_t ransac_seed = 0x5A5AULL;
};

struct RadarEgoVelocityResult {
  bool valid = false;
  /** Estimated ego velocity, expressed in the same frame as the input directions. */
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  std::size_t inlier_count = 0;
  double residual_rms = std::numeric_limits<double>::infinity();
};

namespace detail {

// Closed-form weighted least squares over the given measurement subset:
// minimizes sum (doppler_sign * d_i . v - v_r_i)^2 for v.
inline std::optional<Eigen::Vector3d> solve_radar_velocity_least_squares(
    const std::vector<RadarDopplerMeasurement>& measurements,
    const std::vector<std::size_t>& indices,
    const double doppler_sign) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
  for (const std::size_t index : indices) {
    const Eigen::Vector3d& direction = measurements[index].direction;
    const double target = measurements[index].doppler_velocity / doppler_sign;
    A.noalias() += direction * direction.transpose();
    rhs.noalias() += direction * target;
  }
  const Eigen::FullPivLU<Eigen::Matrix3d> lu(A);
  if (!lu.isInvertible()) {
    return std::nullopt;
  }
  const Eigen::Vector3d velocity = lu.solve(rhs);
  if (!velocity.allFinite()) {
    return std::nullopt;
  }
  return velocity;
}

} // namespace detail

/**
 * Estimate 3D ego velocity from a single radar scan's per-point radial Doppler
 * measurements. Dynamic-object outliers are rejected with RANSAC over a
 * 3-point minimal sample (the minimum needed to solve for a 3D velocity from
 * radial-speed constraints); the final estimate is a closed-form weighted
 * least-squares fit over the largest inlier set found.
 */
inline RadarEgoVelocityResult estimate_radar_ego_velocity(
    const std::vector<RadarDopplerMeasurement>& measurements,
    const RadarEgoVelocityConfig& config = {}) {
  RadarEgoVelocityResult result;
  const std::size_t min_points = std::max<std::size_t>(3, config.min_points);
  if (measurements.size() < min_points || std::abs(config.doppler_sign) < 1.0e-9) {
    return result;
  }
  for (const auto& measurement : measurements) {
    if (!measurement.direction.allFinite() || !std::isfinite(measurement.doppler_velocity)) {
      return result;
    }
  }

  const double inlier_threshold = std::max(0.0, config.ransac_inlier_threshold);
  const std::size_t n = measurements.size();

  std::mt19937_64 rng(config.ransac_seed);
  std::uniform_int_distribution<std::size_t> pick(0, n - 1);

  std::vector<std::size_t> best_inliers;
  for (std::size_t iter = 0; iter < std::max<std::size_t>(1, config.ransac_iterations); ++iter) {
    std::array<std::size_t, 3> sample{};
    sample[0] = pick(rng);
    do {
      sample[1] = pick(rng);
    } while (sample[1] == sample[0]);
    do {
      sample[2] = pick(rng);
    } while (sample[2] == sample[0] || sample[2] == sample[1]);

    const std::vector<std::size_t> sample_indices(sample.begin(), sample.end());
    const auto candidate =
        detail::solve_radar_velocity_least_squares(measurements, sample_indices, config.doppler_sign);
    if (!candidate.has_value()) {
      continue;
    }

    std::vector<std::size_t> inliers;
    inliers.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double predicted = config.doppler_sign * measurements[i].direction.dot(*candidate);
      if (std::abs(predicted - measurements[i].doppler_velocity) <= inlier_threshold) {
        inliers.push_back(i);
      }
    }
    if (inliers.size() > best_inliers.size()) {
      best_inliers = std::move(inliers);
    }
  }

  if (best_inliers.size() < std::max<std::size_t>(3, config.min_inliers)) {
    return result;
  }

  const auto refined =
      detail::solve_radar_velocity_least_squares(measurements, best_inliers, config.doppler_sign);
  if (!refined.has_value()) {
    return result;
  }

  double residual_squared_sum = 0.0;
  for (const std::size_t index : best_inliers) {
    const double predicted = config.doppler_sign * measurements[index].direction.dot(*refined);
    const double residual = predicted - measurements[index].doppler_velocity;
    residual_squared_sum += residual * residual;
  }

  result.valid = true;
  result.velocity = *refined;
  result.inlier_count = best_inliers.size();
  result.residual_rms = std::sqrt(residual_squared_sum / static_cast<double>(best_inliers.size()));
  return result;
}

/**
 * Build a 3x3 translation information (inverse covariance) matrix from independent per-axis
 * sigmas (e.g. radar forward/lateral/vertical velocity noise), expressed in whatever frame the
 * three sigmas are defined in.
 */
inline Eigen::Matrix3d diagonal_velocity_information(const double sigma_axis0,
                                                     const double sigma_axis1,
                                                     const double sigma_axis2) {
  const Eigen::Vector3d inverse_variance(1.0 / (sigma_axis0 * sigma_axis0), 1.0 / (sigma_axis1 * sigma_axis1),
                                         1.0 / (sigma_axis2 * sigma_axis2));
  return inverse_variance.asDiagonal();
}

/** Result of blending two independent Gaussian estimates of the same 3D velocity. */
struct RadarIcpVelocityBlendResult {
  bool valid = false;
  /** Information-weighted fused velocity; only meaningful when `valid`. */
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
};

/**
 * Information-weighted (inverse-covariance-weighted) fusion of an ICP-derived velocity estimate
 * and a radar-derived velocity estimate, both expressed in the same (world) frame:
 *
 *   v_fused = (I_icp + I_radar)^-1 * (I_icp * v_icp + I_radar * v_radar)
 *
 * This is the closed-form Bayesian fusion of two independent Gaussian estimates of the same
 * quantity, and reduces to a simple weighted average when the informations are isotropic and
 * commute. Per-axis (anisotropic) informations let one sensor dominate only the axes it
 * actually observes well -- e.g. a narrow-FOV radar with a well-constrained forward-velocity
 * estimate but a nearly unobservable lateral/vertical one.
 *
 * Guards against numerical failure: the combined information is symmetrized (roundoff can break
 * exact symmetry) and solved with LDLT rather than a raw inverse. `valid` is false (and
 * `velocity` must not be used) whenever:
 *   - either input has a non-finite entry;
 *   - the combined information is not positive definite (e.g. both inputs disagree in sign along
 *     some axis -- should not happen for two genuine information matrices, but a caller error is
 *     not this function's problem to solve silently); or
 *   - the combined information is rank-deficient along some axis (neither sensor observes that
 *     axis at all -- e.g. a 2D radar and a purely-vertical-degenerate ICP solve sharing the same
 *     blind spot). LDLT can technically "solve" such a system (returning 0 for the unconstrained
 *     component when the matching right-hand side is also 0), but that 0 is an artifact of the
 *     decomposition, not a real measurement, so it must not be reported as a confident fused
 *     velocity. This is checked via the LDLT diagonal D: a near-zero entry relative to the
 *     largest one means that direction carries essentially no combined information.
 * Callers should fall back to the uncorrected ICP velocity whenever `valid` is false.
 */
inline RadarIcpVelocityBlendResult blend_icp_radar_velocity(const Eigen::Vector3d& icp_velocity,
                                                             const Eigen::Matrix3d& icp_information,
                                                             const Eigen::Vector3d& radar_velocity,
                                                             const Eigen::Matrix3d& radar_information) {
  RadarIcpVelocityBlendResult result;
  if (!icp_velocity.allFinite() || !radar_velocity.allFinite() || !icp_information.allFinite() ||
      !radar_information.allFinite()) {
    return result;
  }
  Eigen::Matrix3d combined_information = icp_information + radar_information;
  // Roundoff (and, defensively, a caller passing a not-quite-symmetric information matrix) can
  // break exact symmetry; LDLT below assumes a symmetric matrix, so restore it explicitly.
  combined_information = 0.5 * (combined_information + combined_information.transpose());
  const Eigen::LDLT<Eigen::Matrix3d> ldlt(combined_information);
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
    return result;
  }
  // Reject rank-deficient combined information (see doc comment above): every direction must
  // carry at least a small fraction of the strongest direction's information to be trusted.
  const Eigen::Vector3d diagonal_d = ldlt.vectorD();
  const double max_d = diagonal_d.maxCoeff();
  constexpr double kMinRelativeInformation = 1.0e-9;
  if (!(max_d > 0.0) || diagonal_d.minCoeff() <= kMinRelativeInformation * max_d) {
    return result;
  }
  const Eigen::Vector3d rhs = icp_information * icp_velocity + radar_information * radar_velocity;
  const Eigen::Vector3d fused_velocity = ldlt.solve(rhs);
  if (!fused_velocity.allFinite()) {
    return result;
  }
  result.valid = true;
  result.velocity = fused_velocity;
  return result;
}

} // namespace rko_lio::core
