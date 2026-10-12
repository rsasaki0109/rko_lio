#include <gtest/gtest.h>

#include "rko_lio/core/stall_guard.hpp"

namespace {

using rko_lio::core::StallGuard;
using rko_lio::core::StallGuardConfig;

constexpr double kDt = 0.1;

// Drive at 9 m/s for 3 s, then let the registered speed fall to zero over 2 s while the
// accelerometer reads `accel_during_drop`. Returns the speed the guard gives on the last scan.
double drop_to_zero(StallGuard& guard, const double accel_during_drop) {
  const StallGuardConfig config;
  double time = 0.0;
  double speed = 9.0;
  for (int scan = 0; scan < 30; ++scan) {
    speed = guard.update(time += kDt, kDt, 9.0, 0.0, speed, config);
  }
  for (int scan = 1; scan <= 20; ++scan) {
    speed = guard.update(time += kDt, kDt, 9.0 * (1.0 - scan / 20.0), accel_during_drop, speed, config);
  }
  return speed;
}

TEST(StallGuard, CruiseIsLeftAlone) {
  StallGuard guard;
  const StallGuardConfig config;
  double time = 0.0;
  for (int scan = 0; scan < 200; ++scan) {
    const double registered = 9.0 + (scan % 2 == 0 ? 0.8 : -0.8);
    EXPECT_DOUBLE_EQ(guard.update(time += kDt, kDt, registered, 0.0, 9.0, config), registered);
  }
  EXPECT_EQ(guard.hold_count, 0U);
}

TEST(StallGuard, StallWithoutBrakingHoldsTheSpeed) {
  StallGuard guard;
  EXPECT_NEAR(drop_to_zero(guard, 0.0), 9.0, 1.0e-9);
  EXPECT_TRUE(guard.holding());
  EXPECT_EQ(guard.hold_count, 1U);
}

TEST(StallGuard, BrakingTheAccelerometerSeesIsLeftAlone) {
  StallGuard guard;
  EXPECT_NEAR(drop_to_zero(guard, -4.5), 0.0, 1.0e-9);
  EXPECT_FALSE(guard.holding());
}

TEST(StallGuard, HoldIsReleasedWhenTheRegistrationAgreesAgain) {
  StallGuard guard;
  const StallGuardConfig config;
  drop_to_zero(guard, 0.0);
  double time = 10.0;
  for (int scan = 0; scan < 50; ++scan) {
    EXPECT_DOUBLE_EQ(guard.update(time += kDt, kDt, 0.0, 0.0, 9.0, config), 9.0);
  }
  for (int scan = 1; scan < config.exit_scans; ++scan) {
    guard.update(time += kDt, kDt, 9.5, 0.0, 9.0, config);
  }
  EXPECT_TRUE(guard.holding());
  EXPECT_DOUBLE_EQ(guard.update(time += kDt, kDt, 9.5, 0.0, 9.0, config), 9.5);
  EXPECT_FALSE(guard.holding());
}

TEST(StallGuard, ASlowdownTheAccelerometerMissesIsLeftAlone) {
  // 16 -> 12 m/s over 2 s with a tilt error hiding the braking: too much speed is kept for a stall.
  StallGuard guard;
  const StallGuardConfig config;
  double time = 0.0;
  double speed = 16.0;
  for (int scan = 0; scan < 30; ++scan) speed = guard.update(time += kDt, kDt, 16.0, 0.0, speed, config);
  for (int scan = 1; scan <= 20; ++scan) {
    speed = guard.update(time += kDt, kDt, 16.0 - 0.2 * scan, 0.0, speed, config);
  }
  EXPECT_NEAR(speed, 12.0, 1.0e-9);
  EXPECT_EQ(guard.hold_count, 0U);
}

TEST(StallGuard, SlowManoeuvresAreLeftAlone) {
  StallGuard guard;
  const StallGuardConfig config;
  double time = 0.0;
  for (int scan = 0; scan < 50; ++scan) {
    EXPECT_DOUBLE_EQ(guard.update(time += kDt, kDt, scan % 10 < 5 ? 2.5 : 0.0, 0.0, 2.5, config),
                     scan % 10 < 5 ? 2.5 : 0.0);
  }
  EXPECT_EQ(guard.hold_count, 0U);
}

} // namespace
