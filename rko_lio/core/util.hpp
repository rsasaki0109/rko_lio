/*
 * MIT License
 *
 * Copyright (c) 2025 Meher V.R. Malladi.
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
#include "persistent_weak_direction.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <optional>
#include <sophus/se3.hpp>

namespace Eigen {
using Matrix3_6d = Matrix<double, 3, 6>;
using Vector6d = Matrix<double, 6, 1>;
using Matrix6d = Matrix<double, 6, 6>;
using Matrix12d = Matrix<double, 12, 12>;
} // namespace Eigen

namespace rko_lio::core {
// aliases
using Vector3dVector = std::vector<Eigen::Vector3d>;
using Secondsd = std::chrono::duration<double>;
using TimestampVector = std::vector<Secondsd>;

// constants and util funcs
constexpr double square(double x) { return x * x; }
constexpr double GRAVITY_MAG = 9.8107;
inline Eigen::Vector3d gravity() { return {0, 0, -GRAVITY_MAG}; }

// ============================================================================
// [v0.8 Phase 1, diagnostic-only] localizability / ICP-conditioning summary.
//
// Nothing below this comment and above `struct State` changes any estimate:
// it exposes the 6x6 Gauss-Newton Hessian `icp()` already computes and
// discards every iteration (see rko_lio/core/lio.cpp's `build_icp_linear_system`
// / `icp`), plus its eigen-decomposition, so downstream consumers (this
// fork's own ROS wrapper, and this repo's `graph_based_slam` degeneracy
// detector) can inspect solve conditioning without recomputing it. See
// docs/roadmap/v0.8.md (lidarslam-ros2, the downstream repo this fork patch
// was written for) for the full design rationale.
// ============================================================================

/**
 * Eigen-decomposition of a 6x6 ICP Gauss-Newton Hessian, ordered to match
 * Sophus::SE3d's se(3) tangent convention (translation rho first, rotation
 * phi second): axes are `[tx, ty, tz, rx, ry, rz]`.
 */
struct LocalizabilitySummary {
  /** Eigenvalues of H in ascending order; a value near zero along a given
   *  eigenvector indicates the solve is under-constrained in that direction
   *  (Zhang & Singh, "On Degeneracy of Optimization-based State Estimation
   *  Problems", ICRA 2016). */
  Eigen::Vector6d eigenvalues = Eigen::Vector6d::Zero();

  /** Eigenvectors of H, one per column, ordered to match `eigenvalues`.
   *  Sign-canonicalized (the largest-magnitude component of each eigenvector
   *  is made non-negative) so the summary is deterministic run-to-run for a
   *  bit-identical `H`, independent of the eigen-solver's internal sign
   *  convention. */
  Eigen::Matrix6d eigenvectors = Eigen::Matrix6d::Identity();

  /** Decompose `H` (assumed symmetric PSD, as `build_icp_linear_system`
   *  always produces) into its ascending eigenvalues/eigenvectors. */
  static LocalizabilitySummary from_hessian(const Eigen::Matrix6d& H) {
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix6d> solver(H);
    LocalizabilitySummary summary;
    summary.eigenvalues = solver.eigenvalues();
    summary.eigenvectors = solver.eigenvectors();
    for (int col = 0; col < summary.eigenvectors.cols(); ++col) {
      Eigen::Index max_row = 0;
      summary.eigenvectors.col(col).cwiseAbs().maxCoeff(&max_row);
      if (summary.eigenvectors(max_row, col) < 0.0) {
        summary.eigenvectors.col(col) *= -1.0;
      }
    }
    return summary;
  }
};

/**
 * Final-iteration ICP linear system for one `register_scan()` call, plus its
 * eigen-summary. Diagnostic-only: populated after a successful ICP solve so
 * callers can inspect conditioning; never fed back into `State::pose`.
 */
struct IcpDiagnostics {
  /** Final-iteration Gauss-Newton Hessian (averaged over correspondences,
   *  same value `icp()` solved `dx = H.ldlt().solve(-b)` against). */
  Eigen::Matrix6d H = Eigen::Matrix6d::Zero();
  /** Final-iteration Gauss-Newton gradient vector, same ordering as `H`. */
  Eigen::Vector6d b = Eigen::Vector6d::Zero();
  /** Eigen-decomposition of `H`. */
  LocalizabilitySummary localizability;
  /** Persistent weak-direction state after observing this scan. */
  PersistentWeakDirectionState persistent_weak_direction;
  /** ICP iterations in this scan where the persistent-direction intervention ran. */
  std::size_t degeneracy_intervention_count = 0;
};

struct DegeneracyPersistenceDiagnosticsSample {
  Secondsd time{0};
  PersistentWeakDirectionState persistent_weak_direction;
  std::size_t intervention_count = 0;
};

// data structs
struct State {
  Secondsd time{0};
  Sophus::SE3d pose;
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();

  /** [v0.8 Phase 1, diagnostic-only] Final-iteration ICP linear system and
   *  eigen-summary for the scan that produced this state, or `std::nullopt`
   *  when no ICP solve happened for this scan (first frame, a scan dropped
   *  during kidnap recovery, or a scan that triggered a relocalization/local
   *  reset -- see LIO::drop_failed_scan / LIO::recover_with_scan). Populated
   *  by LIO::register_scan; consumed by rko_lio/ros/node.cpp to fill the
   *  previously-always-zero nav_msgs/Odometry.pose.covariance field. */
  std::optional<IcpDiagnostics> icp_diagnostics = std::nullopt;
};

struct ImuBias {
  Eigen::Vector3d accelerometer = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyroscope = Eigen::Vector3d::Zero();
};

struct ImuControl {
  Secondsd time{0};
  Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
};

struct Timestamps {
  Secondsd min;
  Secondsd max;
  TimestampVector times;
};

struct LidarFrame {
  Timestamps timestamps;
  Vector3dVector points;
  /** Per-point reflectivity/intensity, same order as `points`. Empty when unavailable or
   *  when config.intensity_constraint is off (see rko_lio/ros/node.cpp's lidar_callback). */
  std::vector<float> intensities;
};

/** Accumulated IMU statistics over the interval between consecutive LiDAR scans. */
struct IntervalStats {
  /** Number of IMU samples accumulated during this interval. */
  int imu_count = 0;

  /** Sum of unbiased angular velocity samples. */
  Eigen::Vector3d angular_velocity_sum = Eigen::Vector3d::Zero();

  /** Sum of gravity-compensated body-frame accelerations. */
  Eigen::Vector3d body_acceleration_sum = Eigen::Vector3d::Zero();

  /** Sum of unbiased raw IMU accelerations. */
  Eigen::Vector3d imu_acceleration_sum = Eigen::Vector3d::Zero();

  /** Mean magnitude of raw IMU acceleration over the interval. */
  double imu_accel_mag_mean = 0;

  /** Variance accumulator for acceleration magnitude using Welford’s method. */
  double welford_sum_of_squares = 0;

  /**
   * Update accumulated statistics with a new IMU measurement.
   * @param unbiased_ang_vel Unbiased angular velocity.
   * @param uncompensated_unbiased_accel Uncompensated, unbiased acceleration.
   * @param compensated_accel Gravity-compensated acceleration.
   */
  void update(const Eigen::Vector3d& unbiased_ang_vel,
              const Eigen::Vector3d& uncompensated_unbiased_accel,
              const Eigen::Vector3d& compensated_accel) {
    ++imu_count;
    angular_velocity_sum += unbiased_ang_vel;
    imu_acceleration_sum += uncompensated_unbiased_accel;

    const double previous_mean = imu_accel_mag_mean;
    const double accel_norm = uncompensated_unbiased_accel.norm();

    imu_accel_mag_mean += (accel_norm - previous_mean) / imu_count;
    welford_sum_of_squares += (accel_norm - previous_mean) * (accel_norm - imu_accel_mag_mean);

    body_acceleration_sum += compensated_accel;
  }

  /** Reset all accumulated statistics to zero. */
  void reset() {
    imu_count = 0;
    angular_velocity_sum.setZero();
    body_acceleration_sum.setZero();
    imu_acceleration_sum.setZero();
    imu_accel_mag_mean = 0;
    welford_sum_of_squares = 0;
  }
};

/** Detail: return type for the acceleration Kalman filter containing gravity and variance estimates. */
struct AccelInfo {
  /** Variance of the raw imu acceleration magnitude. */
  double accel_mag_variance;
  Eigen::Vector3d local_gravity_estimate;
};
}; // namespace rko_lio::core
