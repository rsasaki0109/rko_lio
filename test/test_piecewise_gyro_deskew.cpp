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
