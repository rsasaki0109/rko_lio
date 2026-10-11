/*
 * MIT License
 *
 * Copyright (c) 2026 rsasaki0109.
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
#include <Eigen/Eigenvalues>
#include <sophus/se3.hpp>

#include <algorithm>
#include <cstddef>

namespace rko_lio::core {

/** Error-state Kalman filter on attitude, position, velocity and accelerometer bias.
 *
 *  It propagates with every IMU sample (gyro for the attitude, specific force plus gravity
 *  for velocity and position) and is corrected by each registered scan pose with the
 *  registration's 6x6 covariance. Where the scan constrains the pose weakly, such as the
 *  roll about a circular tunnel's axis and the translation along it, the IMU holds it:
 *  the gyro keeps the attitude, so gravity stays out of the horizontal acceleration, and
 *  the integrated acceleration keeps the velocity. Gyro bias comes from the static
 *  initialization and is not estimated.
 *
 *  Error state (global): R = exp(dtheta) R_hat, p = p_hat + dp, v = v_hat + dv,
 *  b = b_hat + db; ordered [dp, dtheta, dv, db]. */
class InertialPoseFilter {
public:
  using Matrix12d = Eigen::Matrix<double, 12, 12>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

  struct Config {
    /** Accelerometer white noise (m/s^2/sqrt(Hz)), including vibration. */
    double accelerometer_noise = 0.2;
    /** Gyroscope white noise (rad/s/sqrt(Hz)). */
    double gyroscope_noise = 0.01;
    /** Accelerometer bias random walk (m/s^2/sqrt(s)). */
    double accelerometer_bias_random_walk = 0.002;
    double initial_attitude_sigma = 0.01;
    double initial_velocity_sigma = 0.5;
    double initial_bias_sigma = 0.1;
    /** Bias magnitude limit (m/s^2, 0: none). */
    double max_bias = 0.5;
  };

  explicit InertialPoseFilter(const Config& config) : _config(config) {}

  void reset(const Sophus::SE3d& pose, const Eigen::Vector3d& velocity_world, const Eigen::Vector3d& gravity) {
    _pose = pose;
    _velocity = velocity_world;
    _bias.setZero();
    _gravity = gravity;
    _covariance.setZero();
    _covariance.block<3, 3>(3, 3) = square(_config.initial_attitude_sigma) * Eigen::Matrix3d::Identity();
    _covariance.block<3, 3>(6, 6) = square(_config.initial_velocity_sigma) * Eigen::Matrix3d::Identity();
    _covariance.block<3, 3>(9, 9) = square(_config.initial_bias_sigma) * Eigen::Matrix3d::Identity();
    _initialized = true;
  }

  bool initialized() const { return _initialized; }

  /** Propagate over one IMU step with the unbiased angular velocity and specific force
   *  (body frame, gravity not removed). */
  void propagate(const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& specific_force, const double dt) {
    if (!_initialized || dt <= 0.0) {
      return;
    }
    const Eigen::Matrix3d R = _pose.so3().matrix();
    const Eigen::Vector3d force_world = R * (specific_force - _bias);
    const Eigen::Vector3d acceleration = force_world + _gravity;
    // Mean.
    _pose.translation() += _velocity * dt + 0.5 * acceleration * dt * dt;
    _velocity += acceleration * dt;
    _pose.so3() = _pose.so3() * Sophus::SO3d::exp(angular_velocity * dt);
    // Error state: dp' = dv, dv' = -[R(f-b)]x dtheta - R db, dtheta' = noise, db' = noise.
    Matrix12d F = Matrix12d::Identity();
    F.block<3, 3>(0, 6) = dt * Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 3) = -Sophus::SO3d::hat(force_world) * dt;
    F.block<3, 3>(6, 9) = -R * dt;
    Matrix12d Q = Matrix12d::Zero();
    Q.block<3, 3>(3, 3) = square(_config.gyroscope_noise) * dt * Eigen::Matrix3d::Identity();
    Q.block<3, 3>(6, 6) = square(_config.accelerometer_noise) * dt * Eigen::Matrix3d::Identity();
    Q.block<3, 3>(9, 9) = square(_config.accelerometer_bias_random_walk) * dt * Eigen::Matrix3d::Identity();
    _covariance = F * _covariance * F.transpose() + Q;
  }

  /** Correct with a registered pose and its covariance in the left-perturbation
   *  convention of the registration, T' = exp([rho, phi]) T. Returns false (and leaves
   *  the state unchanged) when the squared Mahalanobis distance exceeds ``gate`` > 0. */
  bool update(const Sophus::SE3d& measured, const Matrix6d& registration_covariance, const double gate = 0.0) {
    if (!_initialized) {
      return false;
    }
    // [rho, phi] -> [dp, dtheta]: dp = rho - [t]x phi, dtheta = phi.
    Matrix6d J = Matrix6d::Identity();
    J.block<3, 3>(0, 3) = -Sophus::SO3d::hat(measured.translation());
    const Matrix6d R_meas = J * registration_covariance * J.transpose();
    Eigen::Matrix<double, 6, 1> residual;
    residual.head<3>() = measured.translation() - _pose.translation();
    residual.tail<3>() = (measured.so3() * _pose.so3().inverse()).log();
    const Matrix6d S = _covariance.topLeftCorner<6, 6>() + R_meas;
    const Matrix6d S_inverse = S.ldlt().solve(Matrix6d::Identity());
    if (gate > 0.0 && residual.dot(S_inverse * residual) > gate) {
      ++_rejected;
      return false;
    }
    const Eigen::Matrix<double, 12, 6> K = _covariance.leftCols<6>() * S_inverse;
    const Eigen::Matrix<double, 12, 1> dx = K * residual;
    _pose.translation() += dx.segment<3>(0);
    _pose.so3() = Sophus::SO3d::exp(dx.segment<3>(3)) * _pose.so3();
    _velocity += dx.segment<3>(6);
    _bias += dx.segment<3>(9);
    if (_config.max_bias > 0.0 && _bias.norm() > _config.max_bias) {
      _bias *= _config.max_bias / _bias.norm();
    }
    Eigen::Matrix<double, 12, 12> I_KH = Matrix12d::Identity();
    I_KH.leftCols<6>() -= K;
    _covariance = I_KH * _covariance * I_KH.transpose() + K * R_meas * K.transpose();
    return true;
  }

  const Sophus::SE3d& pose() const { return _pose; }
  const Eigen::Vector3d& velocity() const { return _velocity; }
  const Eigen::Vector3d& bias() const { return _bias; }
  const Matrix12d& covariance() const { return _covariance; }
  std::size_t rejected_updates() const { return _rejected; }

private:
  static double square(const double x) { return x * x; }

  Config _config;
  bool _initialized = false;
  Sophus::SE3d _pose;
  Eigen::Vector3d _velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d _bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d _gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
  Matrix12d _covariance = Matrix12d::Zero();
  std::size_t _rejected = 0;
};

/** 6x6 covariance of a registered pose ([rho, phi], left perturbation) from the Hessian of
 *  a system averaged over ``residuals`` residuals of standard deviation ``sigma``.
 *  Eigen-directions with too little information get ``max_variance``; every direction
 *  gets at least the floors (position m^2 on the first three, rotation rad^2 on the
 *  last three, added in the eigenbasis-free sense of a diagonal). */
inline Eigen::Matrix<double, 6, 6> registration_pose_covariance(const Eigen::Matrix<double, 6, 6>& averaged_hessian,
                                                                const double residuals,
                                                                const double sigma,
                                                                const double max_variance,
                                                                const double position_floor_variance,
                                                                const double rotation_floor_variance) {
  const double scale = sigma * sigma / std::max(1.0, residuals);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig(averaged_hessian);
  Eigen::Matrix<double, 6, 1> inverse_eigenvalues;
  for (int i = 0; i < 6; ++i) {
    const double lambda = eig.eigenvalues()(i);
    inverse_eigenvalues(i) = lambda * max_variance > scale ? scale / lambda : max_variance;
  }
  Eigen::Matrix<double, 6, 6> covariance =
      eig.eigenvectors() * inverse_eigenvalues.asDiagonal() * eig.eigenvectors().transpose();
  covariance.diagonal().head<3>().array() += position_floor_variance;
  covariance.diagonal().tail<3>().array() += rotation_floor_variance;
  return covariance;
}

} // namespace rko_lio::core
