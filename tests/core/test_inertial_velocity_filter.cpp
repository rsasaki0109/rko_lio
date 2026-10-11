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

#include <rko_lio/core/inertial_velocity_filter.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using rko_lio::core::InertialPreintegration;
using rko_lio::core::InertialVelocityFilter;
using rko_lio::core::registration_translation_covariance;

namespace {
constexpr double kScan = 0.1;
constexpr int kImuPerScan = 10;

// Preintegrate a constant body acceleration with no rotation.
InertialPreintegration constant(const Eigen::Vector3d& body_acceleration) {
  InertialPreintegration pre;
  for (int i = 0; i < kImuPerScan; ++i) {
    pre.add(Eigen::Matrix3d::Identity(), body_acceleration, kScan / kImuPerScan);
  }
  return pre;
}

// Step a filter through n scans of constant world acceleration, measuring the true
// position (plus a per-scan offset) with the given covariance; returns the true velocity.
Eigen::Vector3d run(InertialVelocityFilter& filter, const Eigen::Vector3d& acceleration, const int scans,
                    const Eigen::Matrix3d& measurement_covariance, const Eigen::Vector3d& measurement_offset_per_scan) {
  Eigen::Vector3d p = Eigen::Vector3d::Zero(), v = Eigen::Vector3d::Zero(), estimate = Eigen::Vector3d::Zero();
  for (int i = 0; i < scans; ++i) {
    const Eigen::Vector3d p_next = p + v * kScan + 0.5 * acceleration * kScan * kScan;
    filter.predict(estimate, constant(acceleration), kScan);
    const Eigen::Vector3d measured = p_next + measurement_offset_per_scan * (i + 1);
    filter.update(estimate, measured, measurement_covariance);
    estimate = measured;
    v += acceleration * kScan;
    p = p_next;
  }
  return v;
}
} // namespace

TEST_CASE("Preintegration rotates each sample by its own attitude", "[inertial_velocity_filter]") {
  // Turning 90 deg about z over the interval while accelerating along body x.
  InertialPreintegration pre;
  const int steps = 100;
  for (int i = 0; i < steps; ++i) {
    const double yaw = 0.5 * M_PI * (i + 0.5) / steps;
    pre.add(Sophus::SO3d::rotZ(yaw).matrix(), Eigen::Vector3d(1.0, 0.0, 0.0), 1.0 / steps);
  }
  // The integral of (cos, sin) over a quarter turn is (2/pi, 2/pi).
  REQUIRE(pre.dv.x() == Catch::Approx(2.0 / M_PI).margin(1e-3));
  REQUIRE(pre.dv.y() == Catch::Approx(2.0 / M_PI).margin(1e-3));
  REQUIRE(pre.dt == Catch::Approx(1.0));
}

TEST_CASE("Accurate registrations set the velocity", "[inertial_velocity_filter]") {
  InertialVelocityFilter filter({});
  const Eigen::Vector3d truth = run(filter, Eigen::Vector3d(0.5, 0.0, 0.0), 50, 1e-6 * Eigen::Matrix3d::Identity(),
                                    Eigen::Vector3d::Zero());
  REQUIRE((filter.velocity() - truth).norm() < 0.01);
}

TEST_CASE("A slipping axis follows the IMU, not the registration", "[inertial_velocity_filter]") {
  // The registration slips 5 cm per scan along x (0.5 m/s of false motion) but reports
  // that axis as unconstrained; y and z are well constrained.
  Eigen::Matrix3d covariance = 1e-6 * Eigen::Matrix3d::Identity();
  covariance(0, 0) = 100.0;
  InertialVelocityFilter filter({});
  run(filter, Eigen::Vector3d::Zero(), 100, covariance, Eigen::Vector3d(0.05, 0.0, 0.0));
  // Copying the velocity from the slipping poses would give 0.5 m/s.
  REQUIRE(std::abs(filter.velocity().x()) < 0.1);
}

TEST_CASE("A constant accelerometer bias is estimated", "[inertial_velocity_filter]") {
  InertialVelocityFilter filter({});
  const Eigen::Vector3d bias(0.2, -0.1, 0.05);
  for (int i = 0; i < 300; ++i) {
    // Static sensor whose accelerometer reads the bias.
    filter.predict(Eigen::Vector3d::Zero(), constant(bias), kScan);
    filter.update(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 1e-6 * Eigen::Matrix3d::Identity());
  }
  REQUIRE((filter.bias() - bias).norm() < 0.02);
}

TEST_CASE("Registration covariance is large only along the unconstrained axis", "[inertial_velocity_filter]") {
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity();
  H(0, 0) = 0.0; // no information along x
  const Eigen::Matrix3d covariance =
      registration_translation_covariance(H, Eigen::Vector3d::Zero(), 1000.0, 0.05, 100.0);
  REQUIRE(covariance(0, 0) == Catch::Approx(100.0));
  REQUIRE(covariance(1, 1) == Catch::Approx(0.05 * 0.05 / 1000.0));
}
