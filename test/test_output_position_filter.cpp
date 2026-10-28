// Tests for the causal output position filter.
// MIT License. Copyright (c) 2026 Sasaki.

#include "rko_lio/core/output_position_filter.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

namespace {

using rko_lio::core::MeasurementNoiseModel;
using rko_lio::core::OutputPositionFilter;

// A walk with a 2-Hz sway, its exact acceleration, scans at 10 Hz and IMU samples at 400 Hz.
Eigen::Vector3d position_at(const double t) {
  const double w = 2.0 * M_PI * 2.0;
  return {0.5 * t + 0.03 * std::sin(w * t), 0.2 * t, 0.02 * std::sin(w * t)};
}

Eigen::Vector3d acceleration_at(const double t) {
  const double w = 2.0 * M_PI * 2.0;
  return {-0.03 * w * w * std::sin(w * t), 0.0, -0.02 * w * w * std::sin(w * t)};
}

// RMS position error of the raw measurements and of the filter over the last 20 s of a 30-s walk.
std::pair<double, double> run_walk(const Eigen::Vector3d& accelerometer_bias) {
  OutputPositionFilter filter({0.02, 0.001, 1.0e-4});
  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0.0, 0.006);
  filter.reset(0.0, position_at(0.0), Eigen::Vector3d(0.5 + 0.03 * 4.0 * M_PI, 0.2, 0.02 * 4.0 * M_PI));
  double raw_sq = 0.0;
  double filtered_sq = 0.0;
  int n = 0;
  for (int scan = 1; scan <= 300; ++scan) {
    const double t_scan = 0.1 * scan;
    for (int k = 1; k <= 40; ++k) {
      const double t = t_scan - 0.1 + 0.0025 * k;
      filter.propagate(t, acceleration_at(t) + accelerometer_bias);
    }
    const Eigen::Vector3d measured =
        position_at(t_scan) + Eigen::Vector3d(noise(rng), noise(rng), noise(rng));
    const Eigen::Vector3d filtered = filter.update(t_scan, measured);
    if (t_scan > 10.0) {
      raw_sq += (measured - position_at(t_scan)).squaredNorm();
      filtered_sq += (filtered - position_at(t_scan)).squaredNorm();
      ++n;
    }
  }
  return {std::sqrt(raw_sq / n), std::sqrt(filtered_sq / n)};
}

TEST(OutputPositionFilter, RemovesScanToScanNoise) {
  const auto [raw, filtered] = run_walk(Eigen::Vector3d::Zero());
  EXPECT_LT(filtered, 0.5 * raw);
}

TEST(OutputPositionFilter, LearnsAnAccelerationBias) {
  // 0.05 m/s^2 is a 0.3-deg error of the gravity direction.
  const auto [raw, filtered] = run_walk(Eigen::Vector3d(0.05, -0.03, 0.02));
  EXPECT_LT(filtered, 0.6 * raw);
}

TEST(OutputPositionFilter, ResetStartsAtTheGivenPosition) {
  OutputPositionFilter filter({0.02, 0.001, 1.0e-4});
  EXPECT_FALSE(filter.initialized());
  filter.reset(5.0, Eigen::Vector3d(1.0, 2.0, 3.0), Eigen::Vector3d::Zero());
  EXPECT_TRUE(filter.initialized());
  EXPECT_NEAR((filter.update(5.0, Eigen::Vector3d(1.0, 2.0, 3.0)) - Eigen::Vector3d(1.0, 2.0, 3.0)).norm(), 0.0, 1e-12);
}

TEST(OutputPositionFilter, UpdateWithSigmaSquaredIdentityIsTheScalarUpdate) {
  OutputPositionFilter scalar({0.02, 0.003, 1.0e-5});
  OutputPositionFilter full({0.02, 0.003, 1.0e-5});
  scalar.reset(0.0, Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, 0.0, 0.0));
  full.reset(0.0, Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, 0.0, 0.0));
  for (int k = 1; k <= 20; ++k) {
    const double t = 0.1 * k;
    scalar.propagate(t - 0.05, Eigen::Vector3d(0.1, 0.0, 0.0));
    full.propagate(t - 0.05, Eigen::Vector3d(0.1, 0.0, 0.0));
    const Eigen::Vector3d p(t, 0.01 * (k % 3), 0.0);
    EXPECT_EQ(scalar.update(t, p), full.update(t, p, 0.02 * 0.02 * Eigen::Matrix3d::Identity()));
  }
}

TEST(MeasurementNoiseModel, FollowsTheRegistrationHessian) {
  // H averaged over residuals, [t; r]: x is weakly constrained (a corridor along x).
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity();
  H(0, 0) = 0.01;
  const Eigen::Matrix<double, 6, 6> zero = Eigen::Matrix<double, 6, 6>::Zero();
  // fixed: sigma^2 I whatever H is
  MeasurementNoiseModel fixed("fixed", 0.02, 1.0, 0.25, 4.0);
  EXPECT_TRUE(fixed.covariance(H).isApprox(4e-4 * Eigen::Matrix3d::Identity()));
  // shape: the weak axis gets the noise (C = diag(100, 1, 1), mean 34); the others go down, limited to lo
  MeasurementNoiseModel shape("shape", 0.02, 1.0, 0.25, 4.0);
  const Eigen::Matrix3d R = shape.covariance(H);
  EXPECT_NEAR(R(0, 0), 4e-4 * 100.0 / 34.0, 1e-12);
  EXPECT_NEAR(R(1, 1), 4e-4 * 0.25, 1e-12);
  EXPECT_NEAR(R(0, 1), 0.0, 1e-12);
  // scalar: a scan like the average of those before gets sigma^2 I; a weaker one more
  MeasurementNoiseModel scalar("scalar", 0.02, 1.0, 0.25, 4.0);
  EXPECT_TRUE(scalar.covariance(Eigen::Matrix<double, 6, 6>::Identity()).isApprox(4e-4 * Eigen::Matrix3d::Identity()));
  EXPECT_TRUE(scalar.covariance(Eigen::Matrix<double, 6, 6>::Identity()).isApprox(4e-4 * Eigen::Matrix3d::Identity()));
  EXPECT_GT(scalar.covariance(0.5 * Eigen::Matrix<double, 6, 6>::Identity())(0, 0), 4e-4);
  // a scan without a Hessian keeps sigma^2 I
  EXPECT_TRUE(scalar.covariance(zero).isApprox(4e-4 * Eigen::Matrix3d::Identity()));
}

} // namespace
