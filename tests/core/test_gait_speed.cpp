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

using rko_lio::core::apply_gait_speed;
using rko_lio::core::GaitSpeedEstimator;
using rko_lio::core::hold_gait_speed;

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

TEST_CASE("Scans that see the travel refine the learned step", "[gait_speed]") {
  GaitSpeedEstimator::Config config;
  config.min_calibration_samples = 10;
  GaitSpeedEstimator gait(config);
  gait.refine(0.7 * 2.0, 2.0);  // nothing to refine before the step is learned
  REQUIRE(gait.refined_sample_count() == 0);
  for (int i = 0; i < 10; ++i) gait.observe(0.8 * 2.0, 2.0);  // learned while slipping
  REQUIRE(*gait.step_length() == Catch::Approx(0.8).margin(1e-9));
  for (int i = 0; i < 9; ++i) gait.refine(0.7 * 2.0, 2.0);
  gait.refine(4.0, 2.0);  // a runaway is not a step
  REQUIRE(*gait.step_length() == Catch::Approx(0.8).margin(1e-9));
  gait.refine(0.7 * 2.0, 2.0);
  REQUIRE(gait.refined_sample_count() == 10);
  REQUIRE(*gait.step_length() == Catch::Approx(0.7).margin(1e-9));
}

TEST_CASE("Setting the gait speed stops a turning sensor from pumping it", "[gait_speed]") {
  // A walker stops and turns the sensor by 180 degrees in 20 scans; the registration has no
  // hold along the tunnel and returns the predicted velocity unchanged.
  Eigen::Vector3d kept = Eigen::Vector3d::UnitX();
  Eigen::Vector3d set = Eigen::Vector3d::UnitX();
  for (int i = 1; i <= 20; ++i) {
    const double angle = M_PI * i / 20.0;
    const Eigen::Vector3d u(std::cos(angle), std::sin(angle), 0.0);
    kept = apply_gait_speed(kept, u, 1.0, false);
    set = apply_gait_speed(set, u, 1.0, true);
  }
  REQUIRE(kept.norm() > 2.0);
  REQUIRE(set.norm() == Catch::Approx(1.0).margin(1e-9));
  // Straight walking: both give the gait speed along the direction of travel.
  const Eigen::Vector3d u = Eigen::Vector3d::UnitX();
  const Eigen::Vector3d slow(0.4, 0.05, 0.0);
  REQUIRE(apply_gait_speed(slow, u, 1.2, false).dot(u) == Catch::Approx(1.2).margin(1e-12));
  REQUIRE(apply_gait_speed(slow, u, 1.2, true).norm() == Catch::Approx(1.2).margin(1e-12));
}

TEST_CASE("Only the last second tells standing from stepping", "[gait_speed]") {
  GaitSpeedEstimator gait;
  walk(gait, 0.0, 3.0, 1.8, 2.0, 0.05);
  REQUIRE(gait.magnitude_std(1.0) > 1.0);
  // Then the walker stops: the 4 s cadence window still holds steps, the last second does not.
  walk(gait, 3.0, 1.5, 1.8, 0.0, 0.05);
  REQUIRE(gait.magnitude_std(1.0) < 0.15);
  REQUIRE(GaitSpeedEstimator().magnitude_std(1.0) == 0.0);
}

TEST_CASE("Without a cadence the walker keeps its bounds", "[gait_speed]") {
  const Eigen::Vector3d u = Eigen::Vector3d::UnitX();
  const double last_speed = 1.0;
  // A walker standing still: no speed, whatever the registration says.
  REQUIRE(hold_gait_speed(Eigen::Vector3d(7.0, 0.0, 0.0), u, last_speed, true, true).isZero());
  for (const bool set_speed : {false, true}) {
    // A registration running away along the tunnel: no faster than the last gait speed.
    const Eigen::Vector3d runaway = hold_gait_speed(Eigen::Vector3d(7.0, 0.2, 0.0), u, last_speed, false, set_speed);
    REQUIRE(runaway.norm() == Catch::Approx(last_speed).margin(1e-12));
    REQUIRE(runaway.dot(u) > 0.9 * last_speed);
    // Running away backwards: the walker does not walk backwards without turning.
    const Eigen::Vector3d backwards =
        hold_gait_speed(Eigen::Vector3d(-5.0, 0.1, 0.0), u, last_speed, false, set_speed);
    REQUIRE(backwards.dot(u) == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(backwards.norm() <= last_speed + 1e-12);
  }
  // A plausible registration passes unchanged.
  const Eigen::Vector3d slow(0.6, 0.05, 0.0);
  REQUIRE((hold_gait_speed(slow, u, last_speed, false, false) - slow).norm() == Catch::Approx(0.0).margin(1e-12));
}
