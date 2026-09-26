#include <gtest/gtest.h>
#include "rko_lio/core/gyro_deskew_history.hpp"
#include <limits>

using namespace rko_lio::core;
using namespace std::chrono_literals;

TEST(GyroDeskewHistory, ConstantRateAndEndIdentity) {
  GyroDeskewHistory h;
  const Eigen::Vector3d w(0, 0, 2);
  h.add(0ms, 10ms, w);
  h.add(10ms, 20ms, w);
  ASSERT_TRUE(h.at(15ms));
  EXPECT_NEAR((h.at(20ms)->inverse() * h.at(5ms).value()).log().z(), -.03, 1e-12);
  EXPECT_LT((h.at(20ms)->inverse() * h.at(20ms).value()).log().norm(), 1e-12);
  EXPECT_NEAR(h.at(15ms)->log().z(), .03, 1e-12);
}
TEST(GyroDeskewHistory, ChangingNoncommutingRatesPreserveOrder) {
  GyroDeskewHistory h;
  const Eigen::Vector3d x(10, 0, 0), y(0, 10, 0);
  h.add(0ms, 10ms, x);
  h.add(10ms, 20ms, y);
  const auto expected = Sophus::SO3d::exp(x * .01) * Sophus::SO3d::exp(y * .005);
  EXPECT_LT((expected.inverse() * h.at(15ms).value()).log().norm(), 1e-12);
  EXPECT_GT((Sophus::SO3d::exp((x + y) * .0075).inverse() * h.at(15ms).value()).log().norm(), .03);
}
TEST(GyroDeskewHistory, ScanTailAndOverlapRetainContinuousReference) {
  GyroDeskewHistory h;
  h.add(0ms, 8ms, {0, 0, 1});
  ASSERT_TRUE(h.finish_scan(10ms));
  h.add(10ms, 13ms, {0, 0, 2});
  ASSERT_TRUE(h.finish_scan(15ms));
  EXPECT_TRUE(h.covers(7ms, 15ms));
  EXPECT_NEAR((h.at(15ms)->inverse() * h.at(7ms).value()).log().z(), -.013, 1e-12);
}
TEST(GyroDeskewHistory, GapsAndInvalidSamplesCannotBridgeMissingMotion) {
  GyroDeskewHistory h;
  h.add(0ms, 5ms, {0, 0, 1});
  EXPECT_FALSE(h.finish_scan(26ms));
  EXPECT_EQ(h.size(), 0u);
  h.add(30ms, 35ms, {0, 0, 1});
  h.add(40ms, 45ms, {0, 0, 1});
  EXPECT_FALSE(h.at(35ms));
  EXPECT_TRUE(h.at(42ms));
  h.add(45ms, 45ms, {0, 0, 1});
  EXPECT_EQ(h.size(), 0u);
  h.add(45ms, 70ms, {0, 0, 1});
  EXPECT_EQ(h.size(), 0u);
  h.add(50ms, 49ms, {0, 0, 1});
  EXPECT_EQ(h.size(), 0u);
  h.add(50ms, 55ms, {std::numeric_limits<double>::quiet_NaN(), 0, 0});
  EXPECT_EQ(h.size(), 0u);
}
TEST(GyroDeskewHistory, RetentionIsBoundedByTimeAndCount) {
  GyroDeskewHistory h;
  for (int i = 0; i < 10000; ++i) h.add(Nsec(i * 1000), Nsec((i + 1) * 1000), {0, 0, 1});
  EXPECT_LE(h.size(), 4096u);
  EXPECT_FALSE(h.at(0ms));
  h.clear();
  for (int i = 0; i < 1000; ++i) h.add(i * 5ms, (i + 1) * 5ms, {0, 0, 1});
  EXPECT_LE(h.size(), 403u);
  EXPECT_FALSE(h.at(1s));
  EXPECT_TRUE(h.covers(4900ms, 5s));
  h.clear();
  EXPECT_FALSE(h.at(5s));
  EXPECT_FALSE(h.finish_scan(5s));
}
