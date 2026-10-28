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

// Causal position filter for the published trajectory (Config::output_filter_position_sigma).
//
// The registration places each scan with a few millimetres of noise from scan to scan; a filter
// that integrates the IMU between scans and takes the registered positions as measurements keeps
// the IMU's short-term accuracy and the registration's long-term one. State: world position,
// velocity and an acceleration bias (the gravity direction's and the accelerometer's residual
// errors). Only the output is filtered: the registration chain, the map and the state the next
// scan starts from are unchanged.

#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

namespace rko_lio::core {

class OutputPositionFilter {
public:
  struct Config {
    double position_sigma = 0.0;     // registered position noise (m)
    double acceleration_noise = 0.0; // white acceleration noise spectral density (m^2/s^3)
    double bias_noise = 0.0;         // acceleration bias random walk (m^2/s^5)
  };

  OutputPositionFilter() = default;
  explicit OutputPositionFilter(const Config& config) : _config(config) {}

  bool initialized() const { return _time.has_value(); }

  /** Start again at `position` (time in seconds) with the given world velocity. */
  void reset(const double time, const Eigen::Vector3d& position, const Eigen::Vector3d& velocity) {
    _x.setZero();
    _x.segment<3>(0) = position;
    _x.segment<3>(3) = velocity;
    _P.setZero();
    _P.block<3, 3>(0, 0).diagonal().setConstant(_config.position_sigma * _config.position_sigma);
    _P.block<3, 3>(3, 3).diagonal().setConstant(0.5 * 0.5);
    _P.block<3, 3>(6, 6).diagonal().setConstant(0.05 * 0.05);
    _time = time;
    _acceleration.setZero();
  }

  /** Move to `time` with `world_acceleration` (gravity removed) held since the last call. */
  void propagate(const double time, const Eigen::Vector3d& world_acceleration) {
    _acceleration = world_acceleration;
    advance(time);
  }

  /** Move to `time` with the last acceleration and fuse the registered `position`; returns the
   *  filtered position. */
  Eigen::Vector3d update(const double time, const Eigen::Vector3d& position) {
    advance(time);
    const Eigen::Matrix3d S =
        _P.block<3, 3>(0, 0) + _config.position_sigma * _config.position_sigma * Eigen::Matrix3d::Identity();
    const Eigen::Matrix<double, 9, 3> K = _P.block<9, 3>(0, 0) * S.inverse();
    _x += K * (position - _x.segment<3>(0));
    _P -= K * _P.block<3, 9>(0, 0);
    _P = 0.5 * (_P + _P.transpose()).eval();
    return _x.segment<3>(0);
  }

  /** update() with the measurement covariance R in place of position_sigma^2 I. */
  Eigen::Vector3d update(const double time, const Eigen::Vector3d& position, const Eigen::Matrix3d& R) {
    advance(time);
    const Eigen::Matrix3d S = _P.block<3, 3>(0, 0) + R;
    const Eigen::Matrix<double, 9, 3> K = _P.block<9, 3>(0, 0) * S.inverse();
    _x += K * (position - _x.segment<3>(0));
    _P -= K * _P.block<3, 9>(0, 0);
    _P = 0.5 * (_P + _P.transpose()).eval();
    return _x.segment<3>(0);
  }

  Eigen::Vector3d position() const { return _x.segment<3>(0); }
  Eigen::Vector3d velocity() const { return _x.segment<3>(3); }
  Eigen::Vector3d acceleration_bias() const { return _x.segment<3>(6); }

private:
  void advance(const double time) {
    if (!_time.has_value()) return;
    const double dt = time - *_time;
    if (!(dt > 0.0)) return;
    const Eigen::Vector3d a = _acceleration - _x.segment<3>(6);
    _x.segment<3>(0) += _x.segment<3>(3) * dt + 0.5 * a * dt * dt;
    _x.segment<3>(3) += a * dt;
    Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
    F.block<3, 3>(0, 3).diagonal().setConstant(dt);
    F.block<3, 3>(0, 6).diagonal().setConstant(-0.5 * dt * dt);
    F.block<3, 3>(3, 6).diagonal().setConstant(-dt);
    Eigen::Matrix<double, 9, 9> Q = Eigen::Matrix<double, 9, 9>::Zero();
    const double q = _config.acceleration_noise;
    Q.block<3, 3>(0, 0).diagonal().setConstant(q * dt * dt * dt / 3.0);
    Q.block<3, 3>(0, 3).diagonal().setConstant(q * dt * dt / 2.0);
    Q.block<3, 3>(3, 0).diagonal().setConstant(q * dt * dt / 2.0);
    Q.block<3, 3>(3, 3).diagonal().setConstant(q * dt);
    Q.block<3, 3>(6, 6).diagonal().setConstant(_config.bias_noise * dt);
    _P = F * _P * F.transpose() + Q;
    _time = time;
  }

  Config _config;
  Eigen::Matrix<double, 9, 1> _x = Eigen::Matrix<double, 9, 1>::Zero();
  Eigen::Matrix<double, 9, 9> _P = Eigen::Matrix<double, 9, 9>::Zero();
  std::optional<double> _time;
  Eigen::Vector3d _acceleration = Eigen::Vector3d::Zero();
};

/** Measurement covariance of a registered position from its registration's final Gauss-Newton
 *  Hessian H (Config::output_filter_measurement_noise). H is averaged over the residuals, ordered
 *  [t_world; r_world]; C, the translation block of H^-1, is the marginal translation covariance up
 *  to a scale. With C = V diag(lambda) V^T, s = trace(C) / 3 and s_ref the geometric mean of s over
 *  this and every earlier scan with a Hessian (so the scale is learned online):
 *    "fixed"     sigma^2 I
 *    "scalar"    sigma^2 clip((s / s_ref)^alpha) I
 *    "shape"     sigma^2 V clip((lambda / s)^alpha) V^T
 *    "marginal"  sigma^2 V clip((lambda / s_ref)^alpha) V^T
 *  clip limits each factor to [lo, hi]. A zero or singular H gives sigma^2 I and leaves s_ref. */
class MeasurementNoiseModel {
public:
  MeasurementNoiseModel() = default;
  MeasurementNoiseModel(const std::string& mode, const double sigma, const double alpha, const double lo,
                        const double hi)
      : _mode(mode), _variance(sigma * sigma), _alpha(alpha), _lo(lo), _hi(hi) {}

  Eigen::Matrix3d covariance(const Eigen::Matrix<double, 6, 6>& H) {
    Eigen::Matrix3d R = _variance * Eigen::Matrix3d::Identity();
    if (_mode == "fixed" || !(H.norm() > 0.0)) return R;
    const Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu(H);
    if (!lu.isInvertible()) return R;
    Eigen::Matrix3d C = lu.inverse().block<3, 3>(0, 0);
    C = 0.5 * (C + C.transpose()).eval();
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(C);
    const Eigen::Vector3d lambda = eig.eigenvalues().cwiseMax(1e-30);
    const double s = lambda.sum() / 3.0;
    _log_sum += std::log(s);
    ++_count;
    const double s_ref = std::exp(_log_sum / static_cast<double>(_count));
    const auto clip = [this](const double f) { return std::clamp(std::pow(f, _alpha), _lo, _hi); };
    Eigen::Vector3d f;
    for (int k = 0; k < 3; ++k) {
      f(k) = _mode == "scalar" ? clip(s / s_ref) : _mode == "shape" ? clip(lambda(k) / s) : clip(lambda(k) / s_ref);
    }
    return _variance * eig.eigenvectors() * f.asDiagonal() * eig.eigenvectors().transpose();
  }

private:
  std::string _mode = "fixed";
  double _variance = 0.0;
  double _alpha = 1.0;
  double _lo = 0.25;
  double _hi = 4.0;
  double _log_sum = 0.0;
  long _count = 0;
};

} // namespace rko_lio::core
