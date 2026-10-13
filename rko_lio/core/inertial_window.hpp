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
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <vector>
#include <utility>

namespace rko_lio::core {

/** IMU preintegration between two scans in the frame of the first (Forster et al.),
 *  with first-order corrections for changes of the accelerometer and gyroscope biases
 *  away from the ones the samples were integrated with. Gravity is not removed. */
struct ImuPreintegration {
  double dt = 0.0;
  Sophus::SO3d dR;
  Eigen::Vector3d dv = Eigen::Vector3d::Zero();
  Eigen::Vector3d dp = Eigen::Vector3d::Zero();
  Eigen::Matrix3d dR_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dba = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dba = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dbg = Eigen::Matrix3d::Zero();
  /** Biases the samples were corrected with before add(). */
  Eigen::Vector3d accelerometer_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyroscope_bias = Eigen::Vector3d::Zero();

  /** Add one sample (bias-corrected angular velocity and specific force) held for ``step`` s. */
  void add(const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& specific_force, const double step) {
    if (step <= 0.0) {
      return;
    }
    const Eigen::Matrix3d R = dR.matrix();
    const Eigen::Matrix3d a_hat = Sophus::SO3d::hat(specific_force);
    dp += dv * step + 0.5 * R * specific_force * step * step;
    dp_dba += dv_dba * step - 0.5 * R * step * step;
    dp_dbg += dv_dbg * step - 0.5 * R * a_hat * dR_dbg * step * step;
    dv += R * specific_force * step;
    dv_dba -= R * step;
    dv_dbg -= R * a_hat * dR_dbg * step;
    const Eigen::Vector3d phi = angular_velocity * step;
    const Sophus::SO3d increment = Sophus::SO3d::exp(phi);
    dR_dbg = increment.inverse().matrix() * dR_dbg - right_jacobian(phi) * step;
    dR = dR * increment;
    dt += step;
  }

  static Eigen::Matrix3d right_jacobian(const Eigen::Vector3d& phi) {
    const double angle = phi.norm();
    const Eigen::Matrix3d phi_hat = Sophus::SO3d::hat(phi);
    if (angle < 1e-6) {
      return Eigen::Matrix3d::Identity() - 0.5 * phi_hat;
    }
    return Eigen::Matrix3d::Identity() - (1.0 - std::cos(angle)) / (angle * angle) * phi_hat +
           (angle - std::sin(angle)) / (angle * angle * angle) * phi_hat * phi_hat;
  }
};

/** BIEVR-LIO style inertial window: the registered poses of the last scans are held
 *  fixed, and the velocity at every scan and the accelerometer and gyroscope biases are
 *  solved from the IMU preintegration between them (Gauss-Newton). The registration
 *  never sees the IMU; the IMU only gives the next prediction and deskew a velocity and
 *  biases that agree with seconds of registered motion instead of the last few scans.
 *  The first velocity of the window is held at its previous estimate. */
class InertialWindow {
public:
  struct Config {
    /** Window length (s). */
    double length_sec = 10.0;
    /** Intervals needed before a solve. */
    std::size_t min_intervals = 10;
    /** Residual weights (1 / sigma) of the rotation (rad), velocity (m/s) and position (m). */
    double rotation_weight = 1.0;
    double velocity_weight = 1.0;
    double position_weight = 1.0;
    /** Weights (1 / sigma) of priors pulling the biases to their previous estimates. */
    double accelerometer_bias_prior_weight = 1.0;
    double gyroscope_bias_prior_weight = 10.0;
    /** Bias magnitude limits (0: none). */
    double max_accelerometer_bias = 1.0;
    double max_gyroscope_bias = 0.1;
    int iterations = 2;
    /** Also solve the direction of gravity in the world frame (two unknowns), as BIEVR-LIO
     *  does. A tilt of the registered attitude then moves the gravity estimate instead of
     *  leaking into the accelerometer bias: 2 deg of tilt is 0.34 m/s^2 of false bias. */
    bool estimate_gravity = false;
    /** Weight (1 / sigma) of the prior pulling the gravity direction (unit vector) to its
     *  previous estimate. */
    double gravity_prior_weight = 5.0;
    /** Largest angle (deg) the gravity estimate may take from the first, seeded gravity;
     *  0 leaves it free. Keeps a window fitted to bad registrations from tilting the
     *  propagation away. */
    double gravity_max_tilt_deg = 0.0;
    /** Cauchy scale (in weighted squared-residual units) of a robust kernel on each
     *  interval; 0 is plain least squares. An interval whose registered poses disagree
     *  with the IMU is down-weighted instead of fitted. */
    double robust_cost_scale = 0.0;
    /** Attitude smoothing: sigma (rad) of each registered attitude about the gyro chain; 0
     *  holds the registered attitudes. A registration whose rotation jitters from scan to
     *  scan otherwise projects gravity into the velocity residuals (1 deg is 0.17 m/s^2). */
    double attitude_prior_sigma = 0.0;
    /** Sigma (rad) of the gyro rotation over one interval in the attitude smoothing. */
    double attitude_gyro_sigma = 0.001;
  };

  struct State {
    double time = 0.0;
    Sophus::SE3d pose;
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  };

  InertialWindow() = default;
  explicit InertialWindow(const Config& config) : _config(config) {}

  void reset() {
    _states.clear();
    _intervals.clear();
  }

  /** Start from the current biases (the ones the next samples will be corrected with). */
  void set_biases(const Eigen::Vector3d& accelerometer, const Eigen::Vector3d& gyroscope) {
    _accelerometer_bias = accelerometer;
    _gyroscope_bias = gyroscope;
  }

  /** Add a registered scan. ``preintegration`` covers the IMU since the previous scan and
   *  is ignored for the first state. ``velocity`` (world) seeds the new state. */
  void add(const double time, const Sophus::SE3d& pose, const Eigen::Vector3d& velocity,
           const ImuPreintegration& preintegration) {
    if (!_states.empty()) {
      _intervals.push_back(preintegration);
    }
    _states.push_back({time, pose, velocity});
    while (_states.size() > 2 && time - _states.front().time > _config.length_sec) {
      _states.pop_front();
      _intervals.pop_front();
    }
  }

  /** Solve; false when there are too few intervals (the estimates are unchanged).
   *  ``gravity`` is the world gravity; with ``estimate_gravity`` it only seeds the first
   *  solve, and the window keeps its own estimate (see gravity()). */
  bool solve(const Eigen::Vector3d& gravity) {
    const std::size_t m = _intervals.size();
    if (m < std::max<std::size_t>(1, _config.min_intervals)) {
      return false;
    }
    if (!_config.estimate_gravity || !_gravity_seeded) {
      _gravity = gravity;
      _gravity_seeded = true;
      _gravity_reference = gravity.normalized();
    }
    // Unknowns: velocities of states 1..m (state 0 is held), accelerometer bias, gyro bias
    // and, with estimate_gravity, two tangent coordinates of the gravity direction.
    const bool estimate_gravity = _config.estimate_gravity;
    const int n = static_cast<int>(3 * m + 6) + (estimate_gravity ? 2 : 0);
    const int ba = static_cast<int>(3 * m);
    const int bg = ba + 3;
    const int gi = bg + 3;
    const Eigen::Vector3d accelerometer_prior = _accelerometer_bias;
    const Eigen::Vector3d gyroscope_prior = _gyroscope_bias;
    const double gravity_magnitude = _gravity.norm();
    const Eigen::Vector3d gravity_prior = _gravity / gravity_magnitude;
    for (int iteration = 0; iteration < std::max(1, _config.iterations); ++iteration) {
      // Tangent basis of the gravity direction at the current estimate.
      const Eigen::Vector3d direction = _gravity / gravity_magnitude;
      const Eigen::Vector3d e1 = direction.unitOrthogonal();
      Eigen::Matrix<double, 3, 2> tangent;
      tangent << e1, direction.cross(e1);
      std::vector<Sophus::SO3d> attitudes(m + 1);
      for (std::size_t k = 0; k <= m; ++k) {
        attitudes[k] = _states[k].pose.so3();
      }
      if (_config.attitude_prior_sigma > 0.0) {
        smooth_attitudes(attitudes);
      }
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
      Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
      double cost_sum = 0.0;
      double robust_weight_sum = 0.0;
      for (std::size_t k = 0; k < m; ++k) {
        const ImuPreintegration& pre = _intervals[k];
        const State& si = _states[k];
        const State& sj = _states[k + 1];
        const Eigen::Matrix3d Ri_T = attitudes[k].inverse().matrix();
        const Eigen::Vector3d dba = _accelerometer_bias - pre.accelerometer_bias;
        const Eigen::Vector3d dbg = _gyroscope_bias - pre.gyroscope_bias;
        const double dt = pre.dt;
        // 9 residuals [rotation, velocity, position] and their Jacobians in the unknowns
        // that appear: v_i (if free), v_j, ba, bg.
        Eigen::Matrix<double, 9, 1> r;
        Eigen::Matrix<double, 9, 3> J_vi = Eigen::Matrix<double, 9, 3>::Zero();
        Eigen::Matrix<double, 9, 3> J_vj = Eigen::Matrix<double, 9, 3>::Zero();
        Eigen::Matrix<double, 9, 3> J_ba = Eigen::Matrix<double, 9, 3>::Zero();
        Eigen::Matrix<double, 9, 3> J_bg = Eigen::Matrix<double, 9, 3>::Zero();
        const Sophus::SO3d corrected = pre.dR * Sophus::SO3d::exp(pre.dR_dbg * dbg);
        r.segment<3>(0) = (corrected.inverse() * attitudes[k].inverse() * attitudes[k + 1]).log();
        J_bg.block<3, 3>(0, 0) = -pre.dR_dbg;
        r.segment<3>(3) = Ri_T * (sj.velocity - si.velocity - _gravity * dt) - (pre.dv + pre.dv_dba * dba + pre.dv_dbg * dbg);
        J_vi.block<3, 3>(3, 0) = -Ri_T;
        J_vj.block<3, 3>(3, 0) = Ri_T;
        J_ba.block<3, 3>(3, 0) = -pre.dv_dba;
        J_bg.block<3, 3>(3, 0) = -pre.dv_dbg;
        r.segment<3>(6) = Ri_T * (sj.pose.translation() - si.pose.translation() - si.velocity * dt - 0.5 * _gravity * dt * dt) -
                          (pre.dp + pre.dp_dba * dba + pre.dp_dbg * dbg);
        J_vi.block<3, 3>(6, 0) = -Ri_T * dt;
        J_ba.block<3, 3>(6, 0) = -pre.dp_dba;
        J_bg.block<3, 3>(6, 0) = -pre.dp_dbg;
        Eigen::Matrix<double, 9, 2> J_g = Eigen::Matrix<double, 9, 2>::Zero();
        J_g.block<3, 2>(3, 0) = -Ri_T * dt * gravity_magnitude * tangent;
        J_g.block<3, 2>(6, 0) = -0.5 * Ri_T * dt * dt * gravity_magnitude * tangent;
        Eigen::Matrix<double, 9, 1> w;
        w << Eigen::Vector3d::Constant(_config.rotation_weight), Eigen::Vector3d::Constant(_config.velocity_weight),
            Eigen::Vector3d::Constant(_config.position_weight);
        r = w.asDiagonal() * r;
        const double interval_cost = r.squaredNorm();
        cost_sum += interval_cost;
        if (k + 1 == m) _last_newest_cost = interval_cost;
        // Cauchy weight (iteratively reweighted least squares).
        const double robust_weight =
            _config.robust_cost_scale > 0.0 ? 1.0 / (1.0 + interval_cost / _config.robust_cost_scale) : 1.0;
        robust_weight_sum += robust_weight;
        const Eigen::Matrix<double, 9, 1> sw = std::sqrt(robust_weight) * w;
        r *= std::sqrt(robust_weight);
        J_vi = sw.asDiagonal() * J_vi;
        J_vj = sw.asDiagonal() * J_vj;
        J_ba = sw.asDiagonal() * J_ba;
        J_bg = sw.asDiagonal() * J_bg;
        J_g = sw.asDiagonal() * J_g;
        // Scatter into the normal equations. Column offsets; -1 for the held state 0.
        const int ci = k == 0 ? -1 : static_cast<int>(3 * (k - 1));
        const int cj = static_cast<int>(3 * k);
        const std::array<std::pair<int, const Eigen::Matrix<double, 9, 3>*>, 4> blocks = {
            {{ci, &J_vi}, {cj, &J_vj}, {ba, &J_ba}, {bg, &J_bg}}};
        for (const auto& [col_a, J_a] : blocks) {
          if (col_a < 0) continue;
          b.segment<3>(col_a) += J_a->transpose() * r;
          for (const auto& [col_b, J_b] : blocks) {
            if (col_b < 0) continue;
            H.block<3, 3>(col_a, col_b) += J_a->transpose() * *J_b;
          }
          if (estimate_gravity) {
            const Eigen::Matrix<double, 3, 2> cross = J_a->transpose() * J_g;
            H.block<3, 2>(col_a, gi) += cross;
            H.block<2, 3>(gi, col_a) += cross.transpose();
          }
        }
        if (estimate_gravity) {
          b.segment<2>(gi) += J_g.transpose() * r;
          H.block<2, 2>(gi, gi) += J_g.transpose() * J_g;
        }
      }
      _last_mean_cost = cost_sum / static_cast<double>(m);
      _last_mean_robust_weight = robust_weight_sum / static_cast<double>(m);
      // Bias priors.
      const double wa = _config.accelerometer_bias_prior_weight;
      const double wg = _config.gyroscope_bias_prior_weight;
      H.block<3, 3>(ba, ba) += wa * wa * Eigen::Matrix3d::Identity();
      b.segment<3>(ba) += wa * wa * (_accelerometer_bias - accelerometer_prior);
      H.block<3, 3>(bg, bg) += wg * wg * Eigen::Matrix3d::Identity();
      b.segment<3>(bg) += wg * wg * (_gyroscope_bias - gyroscope_prior);
      if (estimate_gravity) {
        // Prior on the unit direction, as BIEVR-LIO's: w * (direction - prior).
        const double wd = _config.gravity_prior_weight;
        const Eigen::Matrix<double, 3, 2> J_prior = wd * tangent;
        H.block<2, 2>(gi, gi) += J_prior.transpose() * J_prior;
        b.segment<2>(gi) += J_prior.transpose() * (wd * (direction - gravity_prior));
      }
      const Eigen::VectorXd dx = H.ldlt().solve(-b);
      if (!dx.allFinite()) {
        return false;
      }
      for (std::size_t k = 1; k <= m; ++k) {
        _states[k].velocity += dx.segment<3>(static_cast<int>(3 * (k - 1)));
      }
      _accelerometer_bias += dx.segment<3>(ba);
      _gyroscope_bias += dx.segment<3>(bg);
      if (estimate_gravity) {
        _gravity = gravity_magnitude * limit_tilt((direction + tangent * dx.segment<2>(gi)).normalized());
      }
      clamp(_accelerometer_bias, _config.max_accelerometer_bias);
      clamp(_gyroscope_bias, _config.max_gyroscope_bias);
    }
    ++_solves;
    return true;
  }

  const std::deque<State>& states() const { return _states; }
  Eigen::Vector3d latest_velocity() const { return _states.empty() ? Eigen::Vector3d::Zero() : _states.back().velocity; }
  const Eigen::Vector3d& accelerometer_bias() const { return _accelerometer_bias; }
  const Eigen::Vector3d& gyroscope_bias() const { return _gyroscope_bias; }
  /** The world gravity the last solve used (estimated with ``estimate_gravity``). */
  const Eigen::Vector3d& gravity() const { return _gravity; }
  std::size_t solves() const { return _solves; }
  /** Weighted residual cost of the newest interval and the window's mean, at the start of the
   *  last iteration of the last solve. */
  double last_newest_cost() const { return _last_newest_cost; }
  double last_mean_cost() const { return _last_mean_cost; }
  /** Mean robust weight of the intervals in the last solve (1 without the kernel). */
  double last_mean_robust_weight() const { return _last_mean_robust_weight; }

private:
  // Rotate ``direction`` back onto the cone of gravity_max_tilt_deg around the reference.
  Eigen::Vector3d limit_tilt(const Eigen::Vector3d& direction) const {
    const double limit = _config.gravity_max_tilt_deg * M_PI / 180.0;
    const double angle = std::acos(std::clamp(direction.dot(_gravity_reference), -1.0, 1.0));
    if (limit <= 0.0 || angle <= limit) {
      return direction;
    }
    const Eigen::Vector3d axis = _gravity_reference.cross(direction).normalized();
    return Eigen::AngleAxisd(limit, axis) * _gravity_reference;
  }

  /** Pull the registered attitudes onto the gyro chain (current gyro bias): minimise
   *  sum |Log(R_reg^T R)|^2 / sp^2 + sum |Log(dR^T R_i^T R_j)|^2 / sg^2 (Gauss-Newton). */
  void smooth_attitudes(std::vector<Sophus::SO3d>& attitudes) const {
    const std::size_t m = _intervals.size();
    const int n = static_cast<int>(3 * (m + 1));
    const double wp2 = 1.0 / (_config.attitude_prior_sigma * _config.attitude_prior_sigma);
    const double wg2 = 1.0 / (_config.attitude_gyro_sigma * _config.attitude_gyro_sigma);
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    for (int iteration = 0; iteration < 2; ++iteration) {
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
      Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
      for (std::size_t k = 0; k <= m; ++k) {
        const int i = static_cast<int>(3 * k);
        H.block<3, 3>(i, i) += wp2 * I;
        b.segment<3>(i) += wp2 * (_states[k].pose.so3().inverse() * attitudes[k]).log();
      }
      for (std::size_t k = 0; k < m; ++k) {
        const ImuPreintegration& pre = _intervals[k];
        const Sophus::SO3d corrected = pre.dR * Sophus::SO3d::exp(pre.dR_dbg * (_gyroscope_bias - pre.gyroscope_bias));
        const Sophus::SO3d Q = attitudes[k].inverse() * attitudes[k + 1];
        // r(d_i, d_j) ~ r + d_j - Q^T d_i for R_i <- R_i Exp(d_i), R_j <- R_j Exp(d_j).
        const Eigen::Vector3d r = (corrected.inverse() * Q).log();
        const Eigen::Matrix3d Qm = Q.matrix();
        const int i = static_cast<int>(3 * k);
        const int j = i + 3;
        H.block<3, 3>(i, i) += wg2 * I;
        H.block<3, 3>(j, j) += wg2 * I;
        H.block<3, 3>(i, j) -= wg2 * Qm;
        H.block<3, 3>(j, i) -= wg2 * Qm.transpose();
        b.segment<3>(i) -= wg2 * Qm * r;
        b.segment<3>(j) += wg2 * r;
      }
      const Eigen::VectorXd dx = H.ldlt().solve(-b);
      if (!dx.allFinite()) {
        return;
      }
      for (std::size_t k = 0; k <= m; ++k) {
        attitudes[k] = attitudes[k] * Sophus::SO3d::exp(dx.segment<3>(static_cast<int>(3 * k)));
      }
    }
  }

  static void clamp(Eigen::Vector3d& bias, const double limit) {
    if (limit > 0.0 && bias.norm() > limit) {
      bias *= limit / bias.norm();
    }
  }

  Config _config;
  std::deque<State> _states;
  std::deque<ImuPreintegration> _intervals;
  Eigen::Vector3d _accelerometer_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d _gyroscope_bias = Eigen::Vector3d::Zero();
  std::size_t _solves = 0;
  Eigen::Vector3d _gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
  bool _gravity_seeded = false;
  Eigen::Vector3d _gravity_reference = Eigen::Vector3d(0.0, 0.0, -1.0);
  double _last_newest_cost = 0.0;
  double _last_mean_cost = 0.0;
  double _last_mean_robust_weight = 1.0;
};

} // namespace rko_lio::core
