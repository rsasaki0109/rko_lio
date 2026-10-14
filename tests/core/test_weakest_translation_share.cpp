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

#include <rko_lio/core/degeneracy_aware_solve.hpp>

#include <cmath>
#include <limits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using rko_lio::core::weakest_translation_share;

TEST_CASE("A scan that sees every direction alike shares a third", "[degeneracy]") {
  const double share = weakest_translation_share(Eigen::Matrix<double, 6, 6>::Identity());
  REQUIRE(share == Catch::Approx(1.0 / 3.0).margin(1e-12));
}

TEST_CASE("A tunnel's axis is the weakest translation, whatever the rotation block", "[degeneracy]") {
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  // Translation information along a rotated axis is 1 % of the across-tunnel directions.
  const Eigen::Matrix3d rotation = Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.3, 0.2, 0.9).normalized()).matrix();
  H.block<3, 3>(0, 0) = rotation * Eigen::Vector3d(0.01, 1.0, 1.0).asDiagonal() * rotation.transpose();
  H.block<3, 3>(3, 3) = 50.0 * Eigen::Matrix3d::Identity();
  const double share = weakest_translation_share(H);
  REQUIRE(share == Catch::Approx(0.01 / 2.01).margin(1e-9));
}

TEST_CASE("An empty or broken system observes nothing to compare", "[degeneracy]") {
  REQUIRE(std::isnan(weakest_translation_share(Eigen::Matrix<double, 6, 6>::Zero())));
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity();
  H(0, 0) = std::numeric_limits<double>::quiet_NaN();
  REQUIRE(std::isnan(weakest_translation_share(H)));
}
