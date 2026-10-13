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

#include <rko_lio/core/inertial_window.hpp>

#include <algorithm>
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using rko_lio::core::ImuPreintegration;
using rko_lio::core::InertialWindow;

namespace {
const Eigen::Vector3d kGravity(0.0, 0.0, -9.81);
constexpr double kImuStep = 0.001;
constexpr double kScan = 0.1;
const Eigen::Vector3d kAngularVelocity(0.05, -0.1, 0.2);

// A smooth trajectory: rotating at a constant body rate, accelerating sinusoidally.
Eigen::Vector3d position(const double t) {
  return {3.0 * t + 0.5 * std::sin(t), 0.4 * std::cos(0.7 * t), 0.1 * std::sin(1.3 * t)};
}
Eigen::Vector3d velocity(const double t) {
  return {3.0 + 0.5 * std::cos(t), -0.28 * std::sin(0.7 * t), 0.13 * std::cos(1.3 * t)};
}
Eigen::Vector3d acceleration(const double t) {
  return {-0.5 * std::sin(t), -0.196 * std::cos(0.7 * t), -0.169 * std::sin(1.3 * t)};
}
Sophus::SO3d attitude(const double t) { return Sophus::SO3d::exp(kAngularVelocity * t); }
Sophus::SE3d pose(const double t) { return {attitude(t), position(t)}; }

// Preintegrate [t0, t0 + kScan] from samples carrying the given biases, corrected with
// the given bias estimates (as rko corrects samples with imu_bias before integrating).
ImuPreintegration preintegrate(const double t0, const Eigen::Vector3d& true_accelerometer_bias,
                               const Eigen::Vector3d& true_gyroscope_bias, const Eigen::Vector3d& estimated_accelerometer_bias,
                               const Eigen::Vector3d& estimated_gyroscope_bias,
                               const Eigen::Vector3d& gravity = kGravity) {
  ImuPreintegration pre;
  pre.accelerometer_bias = estimated_accelerometer_bias;
  pre.gyroscope_bias = estimated_gyroscope_bias;
  const int steps = static_cast<int>(std::lround(kScan / kImuStep));
  for (int i = 0; i < steps; ++i) {
    const double t = t0 + (i + 0.5) * kImuStep;
    const Eigen::Vector3d force = attitude(t).inverse() * (acceleration(t) - gravity) + true_accelerometer_bias;
    const Eigen::Vector3d gyro = kAngularVelocity + true_gyroscope_bias;
    pre.add(gyro - estimated_gyroscope_bias, force - estimated_accelerometer_bias, kImuStep);
  }
  return pre;
}
} // namespace

TEST_CASE("Preintegration predicts the next pose and velocity", "[inertial_window]") {
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const double t0 = 1.0;
  const ImuPreintegration pre = preintegrate(t0, zero, zero, zero, zero);
  const Sophus::SO3d Ri = attitude(t0);
  const double dt = pre.dt;
  REQUIRE(dt == Catch::Approx(kScan));
  REQUIRE(((Ri * pre.dR).inverse() * attitude(t0 + dt)).log().norm() < 1e-9);
  const Eigen::Vector3d v = velocity(t0) + kGravity * dt + Ri * pre.dv;
  const Eigen::Vector3d p = position(t0) + velocity(t0) * dt + 0.5 * kGravity * dt * dt + Ri * pre.dp;
  REQUIRE((v - velocity(t0 + dt)).norm() < 1e-4);
  REQUIRE((p - position(t0 + dt)).norm() < 1e-5);
}

TEST_CASE("Bias Jacobians match reintegration", "[inertial_window]") {
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const Eigen::Vector3d ba(0.05, -0.03, 0.02), bg(0.002, -0.001, 0.003);
  // Samples carry the biases; integrate once uncorrected and once corrected.
  const ImuPreintegration raw = preintegrate(2.0, ba, bg, zero, zero);
  const ImuPreintegration corrected = preintegrate(2.0, ba, bg, ba, bg);
  const Eigen::Vector3d dv = raw.dv + raw.dv_dba * ba + raw.dv_dbg * bg;
  const Eigen::Vector3d dp = raw.dp + raw.dp_dba * ba + raw.dp_dbg * bg;
  const Sophus::SO3d dR = raw.dR * Sophus::SO3d::exp(raw.dR_dbg * bg);
  REQUIRE((dv - corrected.dv).norm() < 1e-5);
  REQUIRE((dp - corrected.dp).norm() < 1e-6);
  REQUIRE((dR.inverse() * corrected.dR).log().norm() < 1e-6);
}

TEST_CASE("The window recovers velocity and biases from exact poses", "[inertial_window]") {
  const Eigen::Vector3d ba(0.08, -0.05, 0.04), bg(0.004, -0.002, 0.003);
  InertialWindow::Config config;
  config.accelerometer_bias_prior_weight = 0.01;
  config.gyroscope_bias_prior_weight = 0.01;
  InertialWindow window(config);
  window.set_biases(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  Eigen::Vector3d accelerometer_estimate = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyroscope_estimate = Eigen::Vector3d::Zero();
  // The first velocity is the true one (the window holds it); later seeds are off by 1 m/s.
  window.add(0.0, pose(0.0), velocity(0.0), {});
  for (int k = 1; k <= 150; ++k) {
    const double t = k * kScan;
    const ImuPreintegration pre = preintegrate(t - kScan, ba, bg, accelerometer_estimate, gyroscope_estimate);
    window.add(t, pose(t), velocity(t) + Eigen::Vector3d(1.0, 0.0, 0.0), pre);
    if (window.solve(kGravity)) {
      accelerometer_estimate = window.accelerometer_bias();
      gyroscope_estimate = window.gyroscope_bias();
    }
  }
  REQUIRE((window.latest_velocity() - velocity(15.0)).norm() < 0.01);
  REQUIRE((window.accelerometer_bias() - ba).norm() < 0.01);
  REQUIRE((window.gyroscope_bias() - bg).norm() < 5e-4);
  REQUIRE(window.states().size() == 101);
}

TEST_CASE("One slipped registration barely moves the window velocity", "[inertial_window]") {
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  InertialWindow window;
  window.add(0.0, pose(0.0), velocity(0.0), {});
  for (int k = 1; k <= 60; ++k) {
    const double t = k * kScan;
    Sophus::SE3d registered = pose(t);
    if (k == 60) {
      registered.translation().x() += 0.3; // slips 0.3 m along x
    }
    window.add(t, registered, velocity(t), preintegrate(t - kScan, zero, zero, zero, zero));
    window.solve(kGravity);
  }
  // Differencing the last two poses would be off by 3 m/s.
  REQUIRE((window.latest_velocity() - velocity(6.0)).norm() < 0.5);
}

TEST_CASE("A tilted gravity is estimated instead of leaking into the accelerometer bias", "[inertial_window]") {
  // The registered world is tilted by 2 deg against the true one, as when the registration
  // cannot fix the roll in a round tunnel: in it, the true gravity is kGravity turned by 2 deg.
  const Eigen::Vector3d tilted = Sophus::SO3d::exp(Eigen::Vector3d(0.035, 0.0, 0.0)) * kGravity;
  const auto run = [&](const bool estimate_gravity) {
    InertialWindow::Config config;
    config.estimate_gravity = estimate_gravity;
    config.accelerometer_bias_prior_weight = 0.01;
    config.gyroscope_bias_prior_weight = 0.01;
    InertialWindow window(config);
    window.set_biases(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    Eigen::Vector3d accelerometer_estimate = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyroscope_estimate = Eigen::Vector3d::Zero();
    window.add(0.0, pose(0.0), velocity(0.0), {});
    for (int k = 1; k <= 300; ++k) {
      const double t = k * kScan;
      const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
      window.add(t, pose(t), velocity(t),
                 preintegrate(t - kScan, zero, zero, accelerometer_estimate, gyroscope_estimate, tilted));
      if (window.solve(kGravity)) {
        accelerometer_estimate = window.accelerometer_bias();
        gyroscope_estimate = window.gyroscope_bias();
      }
    }
    return window;
  };
  const InertialWindow estimated = run(true);
  const double tilt_error = std::acos(std::clamp(estimated.gravity().normalized().dot(tilted.normalized()), -1.0, 1.0));
  REQUIRE(tilt_error < 0.1 * M_PI / 180.0);
  REQUIRE(estimated.accelerometer_bias().norm() < 0.02);
  REQUIRE((estimated.latest_velocity() - velocity(30.0)).norm() < 0.02);
  // Held at the untilted gravity, the window explains the tilt with a false bias.
  const InertialWindow held = run(false);
  REQUIRE(held.accelerometer_bias().norm() > 0.1);
}

