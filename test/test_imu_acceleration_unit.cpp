#include <gtest/gtest.h>
#include <stdexcept>

#include "rko_lio/core/imu_acceleration_unit.hpp"

using rko_lio::core::acceleration_scale;
using rko_lio::core::STANDARD_GRAVITY;

TEST(ImuAccelerationUnit, ExplicitUnits) {
  EXPECT_DOUBLE_EQ(acceleration_scale("mps2", 1.0), 1.0);
  EXPECT_DOUBLE_EQ(acceleration_scale("g", 9.8), STANDARD_GRAVITY);
}

TEST(ImuAccelerationUnit, AutoTellsGFromMetresPerSecondSquared) {
  // A Livox driver at rest reads about 1 g; tilted or moving, 0.6 to 1.5 g.
  EXPECT_DOUBLE_EQ(acceleration_scale("auto", 1.003), STANDARD_GRAVITY);
  EXPECT_DOUBLE_EQ(acceleration_scale("auto", 1.5), STANDARD_GRAVITY);
  // The same sensor in m/s^2 stays as it is.
  EXPECT_DOUBLE_EQ(acceleration_scale("auto", 9.84), 1.0);
  EXPECT_DOUBLE_EQ(acceleration_scale("auto", 6.0), 1.0);
}

TEST(ImuAccelerationUnit, RejectsUnknownUnits) {
  EXPECT_THROW(acceleration_scale("G", 1.0), std::invalid_argument);
}
