#include <gtest/gtest.h>

#include "rko_lio/core/process_timestamps.hpp"

#include <vector>

namespace {

using rko_lio::core::Secondsd;
using rko_lio::core::TimestampProcessingConfig;
using rko_lio::core::process_timestamps;

void expect_ntu_offset(const rko_lio::core::Timestamps& timestamps) {
  ASSERT_EQ(timestamps.times.size(), 3U);
  EXPECT_NEAR(timestamps.min.count(), 99.9, 1.0e-12);
  EXPECT_NEAR(timestamps.max.count(), 100.0, 1.0e-12);
  EXPECT_NEAR(timestamps.times[0].count(), 99.9, 1.0e-12);
  EXPECT_NEAR(timestamps.times[1].count(), 99.95, 1.0e-12);
  EXPECT_NEAR(timestamps.times[2].count(), 100.0, 1.0e-12);
}

TEST(ProcessTimestamps, AppliesOffsetAfterRelativeConversion) {
  TimestampProcessingConfig config;
  config.force_relative = true;
  config.offset_seconds = -0.1;
  expect_ntu_offset(process_timestamps(
      std::vector<double>{0.0, 0.05, 0.1}, Secondsd(100.0), config));
}

TEST(ProcessTimestamps, AppliesOffsetToAbsolutePointTimes) {
  TimestampProcessingConfig config;
  config.force_absolute = true;
  config.offset_seconds = -0.1;
  expect_ntu_offset(process_timestamps(
      std::vector<double>{100.0, 100.05, 100.1}, Secondsd(100.0), config));
}

TEST(ProcessTimestamps, ZeroOffsetPreservesExistingBehavior) {
  TimestampProcessingConfig config;
  config.force_relative = true;
  const auto timestamps = process_timestamps(
      std::vector<double>{0.0, 0.1}, Secondsd(42.0), config);
  EXPECT_NEAR(timestamps.min.count(), 42.0, 1.0e-12);
  EXPECT_NEAR(timestamps.max.count(), 42.1, 1.0e-12);
}

} // namespace
