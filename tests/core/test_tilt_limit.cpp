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

#include <rko_lio/core/gravity_alignment.hpp>

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using rko_lio::core::limit_tilt_correction;
using rko_lio::core::tilt_limit_applies;

TEST_CASE("The tilt limit applies only when set and at or above the gate speed", "[tilt]") {
  REQUIRE_FALSE(tilt_limit_applies(0.0, 20.0, 0.0));
  REQUIRE(tilt_limit_applies(0.02, 20.0, 0.0));
  REQUIRE(tilt_limit_applies(0.02, 0.0, 0.0));
  REQUIRE(tilt_limit_applies(0.02, 3.0, 3.0));
  REQUIRE_FALSE(tilt_limit_applies(0.02, 0.8, 3.0));
}

TEST_CASE("The limit caps roll and pitch and keeps the yaw correction", "[tilt]") {
  const Sophus::SO3d predicted = Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.2, 0.3));
  const Eigen::Vector3d correction(0.03, 0.04, 0.2);  // 0.05 rad of tilt, 0.2 rad of yaw
  const Sophus::SO3d registered = Sophus::SO3d::exp(correction) * predicted;
  bool limited = false;
  const Sophus::SO3d result = limit_tilt_correction(predicted, registered, 0.01, &limited);
  REQUIRE(limited);
  const Eigen::Vector3d applied = (result * predicted.inverse()).log();
  REQUIRE(applied.head<2>().norm() == Catch::Approx(0.01).margin(1e-9));
  REQUIRE(applied.z() == Catch::Approx(0.2).margin(1e-9));
  REQUIRE(applied.x() / applied.y() == Catch::Approx(0.75).margin(1e-9));

  const Sophus::SO3d small = Sophus::SO3d::exp(Eigen::Vector3d(0.001, 0.0, 0.1)) * predicted;
  REQUIRE(limit_tilt_correction(predicted, small, 0.01, &limited).matrix().isApprox(small.matrix()));
  REQUIRE_FALSE(limited);
}
