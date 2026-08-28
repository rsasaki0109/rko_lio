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

#include <gtest/gtest.h>

#include <vector>

#include "rko_lio/core/piecewise_gyro_deskew.hpp"

namespace {

using rko_lio::core::Secondsd;
using rko_lio::core::PiecewiseAngularVelocityIntegrator;
using rko_lio::core::TimedAngularVelocity;
using rko_lio::core::integrate_piecewise_angular_velocity;

double rotation_angle(const Sophus::SO3d& rotation) {
  return rotation.log().norm();
}

}  // namespace

TEST(PiecewiseGyroDeskew, ConstantRateMatchesClosedFormRotation) {
  const Eigen::Vector3d angular_velocity(0.0, 0.0, 2.0);
  const Sophus::SO3d rotation = integrate_piecewise_angular_velocity(
      {}, Secondsd(1.0), Secondsd(1.25), angular_velocity);
  EXPECT_NEAR(rotation_angle(rotation), 0.5, 1.0e-12);
}

TEST(PiecewiseGyroDeskew, RespectsDurationOfEachRateSegment) {
  const std::vector<TimedAngularVelocity> samples{
      {Secondsd(0.0), Eigen::Vector3d::Zero()},
      {Secondsd(0.75), Eigen::Vector3d(0.0, 0.0, 4.0)}};
  const Sophus::SO3d rotation = integrate_piecewise_angular_velocity(
      samples, Secondsd(0.0), Secondsd(1.0), Eigen::Vector3d::Zero());
  EXPECT_NEAR(rotation_angle(rotation), 1.0, 1.0e-12);
}

TEST(PiecewiseGyroDeskew, PointTimeUsesOnlyPastGyroSegments) {
  const std::vector<TimedAngularVelocity> samples{
      {Secondsd(0.0), Eigen::Vector3d::Zero()},
      {Secondsd(0.75), Eigen::Vector3d(0.0, 0.0, 4.0)}};
  const Sophus::SO3d rotation = integrate_piecewise_angular_velocity(
      samples, Secondsd(0.0), Secondsd(0.5), Eigen::Vector3d::Zero());
  EXPECT_NEAR(rotation_angle(rotation), 0.0, 1.0e-12);
}

TEST(PiecewiseGyroDeskew, RejectsReversedInterval) {
  EXPECT_THROW(
      integrate_piecewise_angular_velocity(
          {}, Secondsd(2.0), Secondsd(1.0), Eigen::Vector3d::Zero()),
      std::invalid_argument);
}

TEST(PiecewiseGyroDeskew, PrefixIntegratorMatchesLegacyOrderedProducts) {
  const std::vector<TimedAngularVelocity> samples{
      {Secondsd(0.5), Eigen::Vector3d(0.1, -0.2, 0.3)},
      {Secondsd(1.0), Eigen::Vector3d(-0.4, 0.2, 0.1)},
      {Secondsd(1.0), Eigen::Vector3d(0.3, 0.1, -0.2)},
      {Secondsd(1.4), Eigen::Vector3d(0.2, -0.5, 0.4)},
      {Secondsd(1.8), Eigen::Vector3d(-0.1, 0.6, 0.2)}};
  const Secondsd start_time(1.0);
  const Eigen::Vector3d fallback(0.7, -0.8, 0.9);
  const PiecewiseAngularVelocityIntegrator integrator(samples, start_time, fallback);

  for (const double end_time : {1.0, 1.1, 1.4, 1.65, 1.8, 2.0}) {
    const Sophus::SO3d expected = integrate_piecewise_angular_velocity(
        samples, start_time, Secondsd(end_time), fallback);
    const Sophus::SO3d actual = integrator.integrateUntil(Secondsd(end_time));
    EXPECT_EQ(actual.matrix(), expected.matrix()) << "end_time=" << end_time;
  }
}

TEST(PiecewiseGyroDeskew, PrefixIntegratorRejectsReversedInterval) {
  const PiecewiseAngularVelocityIntegrator integrator(
      {}, Secondsd(2.0), Eigen::Vector3d::Zero());
  EXPECT_THROW(integrator.integrateUntil(Secondsd(1.0)), std::invalid_argument);
}

TEST(PiecewiseGyroDeskew, PrefixIntegratorPreservesLegacyForOutOfOrderSamples) {
  const std::vector<TimedAngularVelocity> samples{
      {Secondsd(1.3), Eigen::Vector3d(0.2, 0.1, -0.3)},
      {Secondsd(1.2), Eigen::Vector3d(-0.4, 0.5, 0.1)}};
  const Secondsd start_time(1.0);
  const Secondsd end_time(1.7);
  const Eigen::Vector3d fallback(0.1, -0.2, 0.3);
  const PiecewiseAngularVelocityIntegrator integrator(samples, start_time, fallback);

  EXPECT_EQ(
      integrator.integrateUntil(end_time).matrix(),
      integrate_piecewise_angular_velocity(
          samples, start_time, end_time, fallback).matrix());
}
