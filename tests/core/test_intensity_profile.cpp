/*
 * MIT License
 *
 * Copyright (c) 2026 Sasaki
 */

#include <rko_lio/core/intensity_profile.hpp>
#include <rko_lio/core/correlation_peak_selector.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

rko_lio::core::IntensityProfile make_profile(const std::vector<double>& values) {
  return {
      .values = values,
      .filled = std::vector<bool>(values.size(), true),
      .filled_count = values.size(),
      .valid = true,
  };
}

std::vector<double> shifted(const std::vector<double>& values, const int bins) {
  std::vector<double> result(values.size(), 0.0);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const long source = static_cast<long>(i) - bins;
    if (source >= 0 && source < static_cast<long>(values.size())) {
      result[i] = values[static_cast<std::size_t>(source)];
    }
  }
  return result;
}

}  // namespace

TEST_CASE("intensity profile reports a separated secondary peak", "[intensity_profile]") {
  std::vector<double> texture(41);
  for (std::size_t i = 0; i < texture.size(); ++i) {
    texture[i] = std::sin(0.37 * static_cast<double>(i)) +
                 0.3 * std::cos(0.11 * static_cast<double>(i * i));
  }
  const auto reference = make_profile(texture);
  const auto current = make_profile(shifted(texture, 2));
  rko_lio::core::IntensityProfileConfig config;
  config.bin_size_m = 0.25;
  config.max_shift_m = 1.5;
  config.min_correlation = 0.5;
  config.min_peak_margin = 0.05;
  config.min_filled_bins = 20;

  const auto result =
      rko_lio::core::estimate_profile_shift(reference, current, config);

  REQUIRE(result.valid);
  REQUIRE_FALSE(result.ambiguous);
  REQUIRE(result.peak_margin >= config.min_peak_margin);
  REQUIRE(std::abs(result.shift_m + 0.5) < 0.15);
}

TEST_CASE("intensity profile rejects periodic alias peaks", "[intensity_profile]") {
  std::vector<double> periodic(41);
  for (std::size_t i = 0; i < periodic.size(); ++i) {
    periodic[i] = (i % 2 == 0) ? 1.0 : -1.0;
  }
  const auto profile = make_profile(periodic);
  rko_lio::core::IntensityProfileConfig config;
  config.bin_size_m = 0.25;
  config.max_shift_m = 1.5;
  config.min_correlation = 0.5;
  config.min_peak_margin = 0.05;
  config.min_filled_bins = 20;

  const auto result =
      rko_lio::core::estimate_profile_shift(profile, profile, config);

  REQUIRE_FALSE(result.valid);
  REQUIRE(result.ambiguous);
  REQUIRE(result.correlation > 0.9);
  REQUIRE(result.second_best_correlation > 0.9);
  REQUIRE(result.peak_margin < config.min_peak_margin);
}

TEST_CASE("zero peak margin preserves legacy acceptance", "[intensity_profile]") {
  std::vector<double> periodic(41);
  for (std::size_t i = 0; i < periodic.size(); ++i) {
    periodic[i] = (i % 2 == 0) ? 1.0 : -1.0;
  }
  const auto profile = make_profile(periodic);
  rko_lio::core::IntensityProfileConfig config;
  config.bin_size_m = 0.25;
  config.max_shift_m = 1.5;
  config.min_correlation = 0.5;
  config.min_peak_margin = 0.0;
  config.min_filled_bins = 20;

  const auto result =
      rko_lio::core::estimate_profile_shift(profile, profile, config);

  REQUIRE(result.valid);
  REQUIRE_FALSE(result.ambiguous);
}

TEST_CASE("intensity profile preserves the raw scores used by peak policy",
          "[intensity_profile]") {
  const auto profile = make_profile(std::vector<double>(41, 2.0));
  rko_lio::core::IntensityProfileConfig config;
  config.bin_size_m = 0.25;
  config.max_shift_m = 1.5;
  config.min_correlation = 0.5;
  config.min_peak_margin = 0.0;
  config.min_filled_bins = 20;

  const auto result =
      rko_lio::core::estimate_profile_shift(profile, profile, config);

  REQUIRE(result.valid);
  REQUIRE(result.has_competing_peak);
  REQUIRE(result.correlation == 4.0);
  REQUIRE(result.second_best_correlation == 4.0);
  REQUIRE(result.peak_margin == 0.0);
}

TEST_CASE("generic peak selector exposes rejection reason", "[correlation_peak_selector]") {
  const std::vector<rko_lio::core::CorrelationPeakCandidate> candidates = {
      {-2, 0.92, 50},
      {-1, 0.95, 50},
      {0, 0.96, 50},
      {1, 0.94, 50},
      {2, 0.93, 50},
  };
  rko_lio::core::CorrelationPeakPolicy policy;
  policy.minimum_score = 0.6;
  policy.minimum_support = 40;
  policy.secondary_exclusion_radius = 1;
  policy.minimum_peak_margin = 0.05;

  const auto selection =
      rko_lio::core::select_correlation_peak(candidates, policy);

  REQUIRE_FALSE(selection.accepted);
  REQUIRE(selection.second_best.has_value());
  REQUIRE(selection.rejection ==
          rko_lio::core::CorrelationPeakRejection::ambiguous);
}

TEST_CASE("generic peak selector exclusion radius is policy controlled",
          "[correlation_peak_selector]") {
  const std::vector<rko_lio::core::CorrelationPeakCandidate> candidates = {
      {-2, 0.70, 50},
      {-1, 0.94, 50},
      {0, 0.96, 50},
      {1, 0.93, 50},
      {2, 0.60, 50},
  };
  rko_lio::core::CorrelationPeakPolicy policy;
  policy.minimum_score = 0.6;
  policy.minimum_support = 40;
  policy.minimum_peak_margin = 0.05;

  policy.secondary_exclusion_radius = 0;
  REQUIRE_FALSE(
      rko_lio::core::select_correlation_peak(candidates, policy).accepted);

  policy.secondary_exclusion_radius = 1;
  REQUIRE(
      rko_lio::core::select_correlation_peak(candidates, policy).accepted);
}
