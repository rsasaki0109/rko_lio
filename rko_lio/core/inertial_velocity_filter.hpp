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
#include <sophus/so3.hpp>

#include <algorithm>

namespace rko_lio::core {

/** A position prior for one registration: the predicted world translation and its
 *  information (inverse covariance, 1/m^2). */
struct InertialTranslationPrior {
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
};

/** World-frame IMU preintegration over one scan interval.
 *
 *  Each sample's gravity-compensated body acceleration is rotated by the attitude at that
 *  sample, so fast turns within the interval keep the acceleration direction. ``Jv`` and
 *  ``Jp`` are the derivatives of ``dv`` and ``dp`` with respect to a body-frame
 *  accelerometer bias. */
struct InertialPreintegration {
  double dt = 0.0;
  Eigen::Vector3d dv = Eigen::Vector3d::Zero();
  Eigen::Vector3d dp = Eigen::Vector3d::Zero();
  Eigen::Matrix3d Jv = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d Jp = Eigen::Matrix3d::Zero();

  void reset() { *this = InertialPreintegration{}; }

  void add(const Eigen::Matrix3d& world_from_body, const Eigen::Vector3d& compensated_body_acceleration,
           const double step) {
    if (step <= 0.0) {
      return;
    }
    const Eigen::Vector3d a = world_from_body * compensated_body_acceleration;
    dp += dv * step + 0.5 * a * step * step;
    Jp += Jv * step - 0.5 * world_from_body * step * step;
    dv += a * step;
    Jv -= world_from_body * step;
    dt += step;
  }
};

/** Kalman filter on the world-frame velocity and a residual accelerometer bias.
 *
 *  The registration alone fixes the velocity in the directions the scan constrains. Along
 *  a tunnel axis or across a flat floor it does not, and a velocity copied from the
 *  registered poses carries every slip into the next prediction. This filter propagates
 *  with the IMU preintegration between scans and corrects with the registered
 *  displacement, weighted by the registration's own covariance: well-constrained
 *  directions follow the LiDAR, weak ones follow the IMU. The bias is the part of the
 *  accelerometer error left after the static initialization.
 *
 *  State x = [v (world, at the end of the interval), b (body)]. */
class InertialVelocityFilter {
public:
  struct Config {
    /** Acceleration uncertainty over one scan interval (m/s^2): IMU noise, vibration and
     *  the gravity leak of a small attitude error. */
    double acceleration_sigma = 0.5;
    /** Bias random walk (m/s^2/sqrt(s)). */
    double bias_random_walk = 0.01;
    double initial_velocity_sigma = 1.0;
    double initial_bias_sigma = 0.3;
    /** Floor (m) on the position uncertainty of both one registration and the prior.
     *  The registration's own covariance, from its Hessian, is millimetres in every
     *  well-constrained direction; trusted that much, each scan's displacement sets the
     *  velocity, and a wrong prediction pins the next scan. */
    double min_position_sigma = 0.0;
  };

  using Matrix6d = Eigen::Matrix<double, 6, 6>;

  explicit InertialVelocityFilter(const Config& config) : _config(config) { reset(Eigen::Vector3d::Zero()); }

  void reset(const Eigen::Vector3d& velocity_world) {
    _velocity = velocity_world;
    _bias.setZero();
    _covariance.setZero();
    _covariance.topLeftCorner<3, 3>() = square(_config.initial_velocity_sigma) * Eigen::Matrix3d::Identity();
    _covariance.bottomRightCorner<3, 3>() = square(_config.initial_bias_sigma) * Eigen::Matrix3d::Identity();
    _interval = 0.0;
  }

  /** Propagate over a scan interval of ``dt`` seconds; returns the prior on the end
   *  position. */
  InertialTranslationPrior predict(const Eigen::Vector3d& start_translation,
                                   const InertialPreintegration& preintegration,
                                   const double dt) {
    // p_end = p + v dt + dp + Jp b, v_end = v + dv + Jv b.
    Eigen::Matrix<double, 3, 6> G;
    G << dt * Eigen::Matrix3d::Identity(), preintegration.Jp;
    Eigen::Matrix3d position_covariance = G * _covariance * G.transpose();
    position_covariance.diagonal().array() +=
        square(0.5 * _config.acceleration_sigma * dt * dt) + square(_config.min_position_sigma);
    InertialTranslationPrior prior;
    prior.translation = start_translation + _velocity * dt + preintegration.dp + preintegration.Jp * _bias;
    prior.information = position_covariance.ldlt().solve(Eigen::Matrix3d::Identity());

    Matrix6d F = Matrix6d::Identity();
    F.topRightCorner<3, 3>() = preintegration.Jv;
    Matrix6d Q = Matrix6d::Zero();
    Q.topLeftCorner<3, 3>() = square(_config.acceleration_sigma * dt) * Eigen::Matrix3d::Identity();
    Q.bottomRightCorner<3, 3>() = square(_config.bias_random_walk) * dt * Eigen::Matrix3d::Identity();
    _velocity += preintegration.dv + preintegration.Jv * _bias;
    _covariance = F * _covariance * F.transpose() + Q;
    _preintegration = preintegration;
    _interval = dt;
    return prior;
  }

  /** Correct with the registered end position (the start position is the previous one)
   *  and the registration's position covariance (m^2). */
  void update(const Eigen::Vector3d& start_translation,
              const Eigen::Vector3d& registered_translation,
              const Eigen::Matrix3d& position_covariance) {
    if (_interval <= 0.0) {
      return;
    }
    const double dt = _interval;
    const Eigen::Matrix3d measurement_covariance =
        (position_covariance + square(_config.min_position_sigma) * Eigen::Matrix3d::Identity()) / square(dt);
    // Mean velocity over the interval as a function of the end state:
    // (p_end - p) / dt = v_end - dv - Jv b + (dp + Jp b) / dt.
    const Eigen::Vector3d measured = (registered_translation - start_translation) / dt;
    const Eigen::Matrix3d bias_jacobian = _preintegration.Jp / dt - _preintegration.Jv;
    const Eigen::Vector3d predicted =
        _velocity - _preintegration.dv + _preintegration.dp / dt + bias_jacobian * _bias;
    Eigen::Matrix<double, 3, 6> H;
    H << Eigen::Matrix3d::Identity(), bias_jacobian;
    const Eigen::Matrix3d S = H * _covariance * H.transpose() + measurement_covariance;
    const Eigen::Matrix<double, 6, 3> K = _covariance * H.transpose() * S.ldlt().solve(Eigen::Matrix3d::Identity());
    const Eigen::Matrix<double, 6, 1> dx = K * (measured - predicted);
    _velocity += dx.head<3>();
    _bias += dx.tail<3>();
    const Matrix6d I_KH = Matrix6d::Identity() - K * H;
    _covariance = I_KH * _covariance * I_KH.transpose() + K * measurement_covariance * K.transpose();
  }

  const Eigen::Vector3d& velocity() const { return _velocity; }
  const Eigen::Vector3d& bias() const { return _bias; }
  const Matrix6d& covariance() const { return _covariance; }

private:
  static double square(const double x) { return x * x; }

  Config _config;
  Eigen::Vector3d _velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d _bias = Eigen::Vector3d::Zero();
  Matrix6d _covariance = Matrix6d::Identity();
  InertialPreintegration _preintegration;
  double _interval = 0.0;
};

/** Covariance (m^2) of the registered translation from the 6x6 registration Hessian of a
 *  system averaged over ``residuals`` residuals of standard deviation ``sigma`` (m), in
 *  the left-perturbation convention T' = exp(dx) T. Directions the Hessian does not
 *  constrain get ``max_variance``. */
inline Eigen::Matrix3d registration_translation_covariance(const Eigen::Matrix<double, 6, 6>& averaged_hessian,
                                                           const Eigen::Vector3d& translation,
                                                           const double residuals,
                                                           const double sigma,
                                                           const double max_variance) {
  const double scale = sigma * sigma / std::max(1.0, residuals);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig(averaged_hessian);
  Eigen::Matrix<double, 6, 1> inverse_eigenvalues;
  for (int i = 0; i < 6; ++i) {
    const double lambda = eig.eigenvalues()(i);
    inverse_eigenvalues(i) = lambda * max_variance > scale ? scale / lambda : max_variance;
  }
  const Eigen::Matrix<double, 6, 6> pose_covariance =
      eig.eigenvectors() * inverse_eigenvalues.asDiagonal() * eig.eigenvectors().transpose();
  // t' = t + rho - [t]x phi under exp(dx) T.
  Eigen::Matrix<double, 3, 6> J;
  J << Eigen::Matrix3d::Identity(), -Sophus::SO3d::hat(translation);
  return J * pose_covariance * J.transpose();
}

} // namespace rko_lio::core
