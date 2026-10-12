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

#include <rko_lio/core/gait_speed.hpp>

#include <cmath>
#include <random>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using rko_lio::core::GaitSpeedEstimator;

namespace {
// Specific force of a handheld sensor carried by someone walking at `cadence` steps/s.
void walk(GaitSpeedEstimator& gait, const double t0, const double duration, const double cadence,
          const double amplitude, const double noise) {
  std::mt19937 rng(7);
  std::normal_distribution<double> n(0.0, noise);
  for (double t = t0; t < t0 + duration; t += 0.01) {
    const double bounce = amplitude * std::sin(2.0 * M_PI * cadence * t);
    gait.add_imu(t, Eigen::Vector3d(n(rng), n(rng), 9.81 + bounce + n(rng)));
  }
}
} // namespace

TEST_CASE("The cadence is the step rhythm of the specific force", "[gait_speed]") {
  GaitSpeedEstimator gait;
  walk(gait, 0.0, 5.0, 1.8, 2.0, 0.3);
  const auto cadence = gait.cadence();
  REQUIRE(cadence.has_value());
  REQUIRE(*cadence == Catch::Approx(1.8).margin(0.03));
}

TEST_CASE("Standing still is not walking", "[gait_speed]") {
  GaitSpeedEstimator gait;
  walk(gait, 0.0, 5.0, 1.8, 0.0, 0.05);
  REQUIRE_FALSE(gait.cadence().has_value());
}

TEST_CASE("The step length ignores stalls and runaways", "[gait_speed]") {
  GaitSpeedEstimator gait;
  REQUIRE_FALSE(gait.step_length().has_value());
  for (int i = 0; i < 50; ++i) gait.observe(0.75 * 1.8, 1.8);  // tracking
  for (int i = 0; i < 50; ++i) gait.observe(0.05, 1.8);        // stalled
  for (int i = 0; i < 50; ++i) gait.observe(4.0, 1.8);         // running away
  REQUIRE_FALSE(gait.step_length().has_value());
  for (int i = 0; i < 50; ++i) gait.observe(0.75 * 1.8, 1.8);
  REQUIRE(gait.step_length().has_value());
  REQUIRE(*gait.step_length() == Catch::Approx(0.75).margin(1e-9));
  REQUIRE_FALSE(gait.plausible_step(0.05 / 1.8));
}

TEST_CASE("The step length is frozen once learned", "[gait_speed]") {
  GaitSpeedEstimator::Config config;
  config.min_calibration_samples = 10;
  GaitSpeedEstimator gait(config);
  for (int i = 0; i < 10; ++i) gait.observe(0.7 * 2.0, 2.0);
  // Later the registered speed follows the gait; it must not move the step.
  for (int i = 0; i < 100; ++i) gait.observe(0.9 * 2.0, 2.0);
  REQUIRE(*gait.step_length() == Catch::Approx(0.7).margin(1e-9));
}
