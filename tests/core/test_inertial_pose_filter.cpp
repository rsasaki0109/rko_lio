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

#include <rko_lio/core/inertial_pose_filter.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using rko_lio::core::InertialPoseFilter;
using rko_lio::core::registration_pose_covariance;
using Matrix6d = InertialPoseFilter::Matrix6d;

namespace {
constexpr double kScan = 0.1;
constexpr int kImuPerScan = 20;
const Eigen::Vector3d kGravity(0.0, 0.0, -9.81);

struct Truth {
  Sophus::SE3d pose;
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
};

// Advance truth and filter through one scan of constant world acceleration and body
// angular velocity, feeding the filter the exact IMU readings.
void imu_scan(InertialPoseFilter& filter, Truth& truth, const Eigen::Vector3d& acceleration,
              const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& accelerometer_bias) {
  const double dt = kScan / kImuPerScan;
  for (int i = 0; i < kImuPerScan; ++i) {
    const Eigen::Vector3d force = truth.pose.so3().inverse() * (acceleration - kGravity) + accelerometer_bias;
    filter.propagate(angular_velocity, force, dt);
    truth.pose.translation() += truth.velocity * dt + 0.5 * acceleration * dt * dt;
    truth.velocity += acceleration * dt;
    truth.pose.so3() = truth.pose.so3() * Sophus::SO3d::exp(angular_velocity * dt);
  }
}

Matrix6d diagonal(const double position_variance, const double rotation_variance) {
  Matrix6d covariance = Matrix6d::Zero();
  covariance.diagonal() << position_variance, position_variance, position_variance, rotation_variance,
      rotation_variance, rotation_variance;
  return covariance;
}

InertialPoseFilter make_filter(const Truth& truth) {
  InertialPoseFilter::Config config;
  config.max_bias = 0.0;
  InertialPoseFilter filter(config);
  filter.reset(truth.pose, truth.velocity, kGravity);
  return filter;
}
} // namespace

TEST_CASE("Exact IMU and poses keep the filter on the truth", "[inertial_pose_filter]") {
  Truth truth;
  InertialPoseFilter filter = make_filter(truth);
  const Eigen::Vector3d acceleration(0.5, -0.2, 0.1);
  const Eigen::Vector3d angular_velocity(0.05, -0.1, 0.3);
  for (int scan = 0; scan < 50; ++scan) {
    imu_scan(filter, truth, acceleration, angular_velocity, Eigen::Vector3d::Zero());
    REQUIRE(filter.update(truth.pose, diagonal(1e-4, 1e-6)));
  }
  REQUIRE((filter.pose().translation() - truth.pose.translation()).norm() < 1e-6);
  REQUIRE((filter.pose().so3().inverse() * truth.pose.so3()).log().norm() < 1e-8);
  REQUIRE((filter.velocity() - truth.velocity).norm() < 1e-5);
}

TEST_CASE("The IMU carries translation the registration cannot see", "[inertial_pose_filter]") {
  // A tunnel along x: started from the prediction, the registration drags back 30 % of
  // each 1 m step (slip) but says x is free in its covariance.
  Truth truth;
  truth.velocity = Eigen::Vector3d(10.0, 0.0, 0.0);
  InertialPoseFilter filter = make_filter(truth);
  Matrix6d covariance = diagonal(1e-4, 1e-6);
  covariance(0, 0) = 100.0;
  for (int scan = 0; scan < 50; ++scan) {
    imu_scan(filter, truth, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    Sophus::SE3d slipped = truth.pose;
    slipped.translation().x() = filter.pose().translation().x() - 0.3;
    REQUIRE(filter.update(slipped, covariance));
  }
  REQUIRE(truth.pose.translation().x() == Catch::Approx(50.0));
  // Trusting the registration would have lost 15 m.
  REQUIRE(filter.pose().translation().x() > 49.0);
  REQUIRE(filter.velocity().x() > 9.8);
}

TEST_CASE("The gyro holds a roll the registration cannot see", "[inertial_pose_filter]") {
  // A circular tunnel along x: the registration's roll drifts but is reported as free.
  Truth truth;
  InertialPoseFilter filter = make_filter(truth);
  Matrix6d covariance = diagonal(1e-4, 1e-6);
  covariance(3, 3) = 1.0;
  for (int scan = 0; scan < 50; ++scan) {
    imu_scan(filter, truth, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    Sophus::SE3d rolled = truth.pose;
    rolled.so3() = Sophus::SO3d::rotX(0.01 * (scan + 1)) * rolled.so3();
    REQUIRE(filter.update(rolled, covariance));
  }
  // The measured roll reached 0.5 rad; the filter stays near level, so gravity does not
  // leak into the horizontal acceleration.
  REQUIRE(std::abs(filter.pose().so3().log().x()) < 0.02);
  REQUIRE(filter.velocity().norm() < 0.05);
}

TEST_CASE("An accelerometer bias is estimated from well-registered poses", "[inertial_pose_filter]") {
  Truth truth;
  InertialPoseFilter filter = make_filter(truth);
  const Eigen::Vector3d bias(0.1, -0.05, 0.08);
  for (int scan = 0; scan < 600; ++scan) {
    imu_scan(filter, truth, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, scan % 40 < 20 ? 0.5 : -0.5), bias);
    REQUIRE(filter.update(truth.pose, diagonal(1e-4, 1e-6)));
  }
  REQUIRE((filter.bias() - bias).norm() < 0.02);
}

TEST_CASE("The gate rejects a registration jump", "[inertial_pose_filter]") {
  Truth truth;
  InertialPoseFilter filter = make_filter(truth);
  for (int scan = 0; scan < 10; ++scan) {
    imu_scan(filter, truth, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    REQUIRE(filter.update(truth.pose, diagonal(1e-4, 1e-6), 50.0));
  }
  imu_scan(filter, truth, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  Sophus::SE3d jumped = truth.pose;
  jumped.translation().y() += 2.0;
  REQUIRE_FALSE(filter.update(jumped, diagonal(1e-4, 1e-6), 50.0));
  REQUIRE(filter.rejected_updates() == 1);
  REQUIRE(filter.pose().translation().norm() < 1e-6);
}

TEST_CASE("Registration covariance caps unconstrained directions", "[inertial_pose_filter]") {
  Matrix6d hessian = Matrix6d::Identity();
  hessian(0, 0) = 1e-12;
  const Matrix6d covariance = registration_pose_covariance(hessian, 1000.0, 0.1, 100.0, 0.0, 0.0);
  REQUIRE(covariance(0, 0) == Catch::Approx(100.0));
  REQUIRE(covariance(1, 1) == Catch::Approx(1e-5));
  REQUIRE(covariance(4, 4) == Catch::Approx(1e-5));
  const Matrix6d floored = registration_pose_covariance(hessian, 1000.0, 0.1, 100.0, 0.01, 0.001);
  REQUIRE(floored(1, 1) == Catch::Approx(1e-5 + 0.01));
  REQUIRE(floored(4, 4) == Catch::Approx(1e-5 + 0.001));
}
