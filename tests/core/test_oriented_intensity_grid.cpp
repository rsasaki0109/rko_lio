/*
 * MIT License
 *
 * Copyright (c) 2026 Sasaki
 */

#include <rko_lio/core/oriented_intensity_grid.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

struct SyntheticGridInput {
  std::vector<Eigen::Vector3d> points;
  std::vector<float> intensities;
};

SyntheticGridInput make_textured_patch(const Eigen::Vector3d& translation) {
  SyntheticGridInput input;
  for (int longitudinal = -4; longitudinal <= 4; ++longitudinal) {
    for (int lateral = -2; lateral <= 2; ++lateral) {
      const double height =
          0.15 * std::sin(0.7 * static_cast<double>(longitudinal)) +
          0.08 * static_cast<double>(lateral * lateral);
      input.points.emplace_back(
          static_cast<double>(longitudinal),
          static_cast<double>(lateral),
          height);
      input.points.back() += translation;
      input.intensities.push_back(static_cast<float>(
          20.0 +
          3.0 * std::sin(0.83 * static_cast<double>(longitudinal)) +
          1.7 * std::cos(1.31 * static_cast<double>(lateral)) +
          0.2 * static_cast<double>(longitudinal * lateral)));
    }
  }
  return input;
}

rko_lio::core::OrientedIntensityGridConfig grid_config() {
  rko_lio::core::OrientedIntensityGridConfig config;
  config.bin_size_m = 1.0;
  config.half_length_m = 6.0;
  config.half_width_m = 4.0;
  config.max_longitudinal_shift_m = 2.0;
  config.max_lateral_shift_m = 2.0;
  config.min_correlation = 0.5;
  config.min_peak_margin = 0.02;
  config.min_filled_cells = 20;
  return config;
}

}  // namespace

TEST_CASE("oriented intensity grid recovers a two-dimensional translation",
          "[oriented_intensity_grid]") {
  const auto reference_input = make_textured_patch(Eigen::Vector3d::Zero());
  const auto current_input =
      make_textured_patch(Eigen::Vector3d(1.0, -1.0, 0.0));
  const auto config = grid_config();
  const Eigen::Vector3d longitudinal = Eigen::Vector3d::UnitX();
  const Eigen::Vector3d lateral = Eigen::Vector3d::UnitY();
  const Eigen::Vector3d origin = Eigen::Vector3d::Zero();

  const auto reference = rko_lio::core::build_oriented_intensity_grid(
      reference_input.points,
      reference_input.intensities,
      longitudinal,
      lateral,
      origin,
      config);
  const auto current = rko_lio::core::build_oriented_intensity_grid(
      current_input.points,
      current_input.intensities,
      longitudinal,
      lateral,
      origin,
      config);
  const auto result = rko_lio::core::estimate_oriented_grid_shift(
      reference, current, config);

  REQUIRE(reference.valid);
  REQUIRE(current.valid);
  REQUIRE(result.valid);
  REQUIRE_FALSE(result.ambiguous);
  REQUIRE(std::abs(result.longitudinal_shift_m + 1.0) < 0.2);
  REQUIRE(std::abs(result.lateral_shift_m - 1.0) < 0.2);
  REQUIRE(result.correlation > 0.95);
  REQUIRE(result.intensity_correlation > 0.95);
  REQUIRE(result.height_correlation > 0.95);
  REQUIRE(
      std::abs(
          result.correlation -
          (config.intensity_weight * result.intensity_correlation +
           config.height_weight * result.height_correlation) /
              (config.intensity_weight + config.height_weight)) <
      1.0e-12);
  REQUIRE(result.overlap_cells >= config.min_filled_cells);
}

TEST_CASE("height channel disambiguates repeated reflectivity",
          "[oriented_intensity_grid]") {
  auto reference_input = make_textured_patch(Eigen::Vector3d::Zero());
  auto current_input =
      make_textured_patch(Eigen::Vector3d(1.0, 0.0, 0.0));
  for (std::size_t index = 0;
       index < reference_input.intensities.size();
       ++index) {
    const int longitudinal = static_cast<int>(index / 5) - 4;
    const float repeated =
        longitudinal % 2 == 0 ? 10.0F : 30.0F;
    reference_input.intensities[index] = repeated;
    current_input.intensities[index] = repeated;
  }
  auto config = grid_config();
  config.intensity_weight = 1.0;
  config.height_weight = 1.0;
  config.min_peak_margin = 0.01;
  const Eigen::Vector3d longitudinal = Eigen::Vector3d::UnitX();
  const Eigen::Vector3d lateral = Eigen::Vector3d::UnitY();
  const Eigen::Vector3d origin = Eigen::Vector3d::Zero();

  const auto reference = rko_lio::core::build_oriented_intensity_grid(
      reference_input.points,
      reference_input.intensities,
      longitudinal,
      lateral,
      origin,
      config);
  const auto current = rko_lio::core::build_oriented_intensity_grid(
      current_input.points,
      current_input.intensities,
      longitudinal,
      lateral,
      origin,
      config);
  const auto result = rko_lio::core::estimate_oriented_grid_shift(
      reference, current, config);

  REQUIRE(result.valid);
  REQUIRE(std::abs(result.longitudinal_shift_m + 1.0) < 0.2);
}

TEST_CASE("oriented intensity grid rejects collinear axes",
          "[oriented_intensity_grid]") {
  const auto input = make_textured_patch(Eigen::Vector3d::Zero());
  const auto grid = rko_lio::core::build_oriented_intensity_grid(
      input.points,
      input.intensities,
      Eigen::Vector3d::UnitX(),
      2.0 * Eigen::Vector3d::UnitX(),
      Eigen::Vector3d::Zero(),
      grid_config());

  REQUIRE_FALSE(grid.valid);
  REQUIRE(grid.filled_count == 0);
}
