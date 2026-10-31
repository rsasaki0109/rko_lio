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
#include <Eigen/LU>

#include <optional>

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

} // namespace rko_lio::core
