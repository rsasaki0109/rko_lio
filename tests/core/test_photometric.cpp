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

#include <rko_lio/core/photometric_features.hpp>
#include <rko_lio/core/photometric_image.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace {

using rko_lio::core::IntensityImageConfig;
using rko_lio::core::LidarImageModel;

LidarImageModel make_model(int rows = 32, int cols = 512) {
  LidarImageModel model;
  model.rows = rows;
  model.cols = cols;
  for (int r = 0; r < rows; ++r) {
    const double degrees = 22.5 - 45.0 * r / (rows - 1);
    model.altitudes_rad.push_back(degrees * M_PI / 180.0);
  }
  model.pixel_shift_by_row.assign(rows, 0);
  return model;
}

IntensityImageConfig plain_image_config() {
  IntensityImageConfig config;
  config.intensity_scale = 1.0;
  config.line_removal = false;
  config.brightness_filter = false;
  config.blur = true;
  config.min_range_m = 0.5;
  config.max_range_m = 30.0;
  return config;
}

// Smooth texture on the walls of a long corridor (x along the corridor).
float texture(const Eigen::Vector3d& p) {
  return static_cast<float>(120.0 + 60.0 * std::sin(2.0 * M_PI * p.x() / 1.3) * std::cos(2.0 * M_PI * p.z() / 1.7) +
                            30.0 * std::sin(2.0 * M_PI * (p.x() + p.y()) / 0.9));
}

// Ray-cast an organized scan of a 40 x 6 x 4 m corridor from the given world pose.
void render_corridor(const LidarImageModel& model,
                     const Sophus::SE3d& world_from_sensor,
                     std::vector<Eigen::Vector3d>& cloud,
                     std::vector<float>& intensities) {
  const std::size_t n = static_cast<std::size_t>(model.rows) * model.cols;
  cloud.assign(n, Eigen::Vector3d::Zero());
  intensities.assign(n, 0.0F);
  const Eigen::Vector3d origin = world_from_sensor.translation();
  for (int row = 0; row < model.rows; ++row) {
    for (int col = 0; col < model.cols; ++col) {
      const double elevation = model.altitudes_rad[row];
      const double azimuth = (model.cols / 2.0 - col) * 2.0 * M_PI / model.cols;
      const Eigen::Vector3d direction_sensor(std::cos(elevation) * std::cos(azimuth),
                                             std::cos(elevation) * std::sin(azimuth), std::sin(elevation));
      const Eigen::Vector3d direction = world_from_sensor.so3() * direction_sensor;
      double t_best = std::numeric_limits<double>::max();
      const double bounds[3][2] = {{-20.0, 20.0}, {-3.0, 3.0}, {-2.0, 2.0}};
      for (int axis = 0; axis < 3; ++axis) {
        for (const double bound : bounds[axis]) {
          if (std::abs(direction[axis]) < 1e-9) {
            continue;
          }
          const double t = (bound - origin[axis]) / direction[axis];
          if (t > 0.1 && t < t_best) {
            t_best = t;
          }
        }
      }
      const Eigen::Vector3d hit = origin + t_best * direction;
      const int index = model.cloud_index(row, col);
      cloud[index] = t_best * direction_sensor;
      intensities[index] = texture(hit);
    }
  }
}

rko_lio::core::PhotometricFrame frame_of(const LidarImageModel& model,
                                         const IntensityImageConfig& config,
                                         const std::vector<Eigen::Vector3d>& cloud,
                                         const std::vector<float>& intensities) {
  return rko_lio::core::build_photometric_frame(model, config, cloud, intensities,
                                                std::vector<int>(cloud.size(), 0), {Sophus::SE3d()});
}

} // namespace

TEST_CASE("image model maps cloud indices and pixels both ways", "[photometric]") {
  LidarImageModel model = make_model(8, 64);
  for (int r = 0; r < model.rows; ++r) {
    model.pixel_shift_by_row[r] = (r % 2 == 0) ? 3 : -5;
  }
  model.u_shift = 7;
  for (int index = 0; index < model.rows * model.cols; ++index) {
    const auto [row, col] = model.pixel_of_cloud_index(index);
    REQUIRE(model.cloud_index(row, col) == index);
  }
}

TEST_CASE("projection hits the pixel a beam was cast from and its Jacobian matches", "[photometric]") {
  const LidarImageModel model = make_model();
  const int row = 10;
  const int col = 300;
  const double elevation = model.altitudes_rad[row];
  const double azimuth = (model.cols / 2.0 - col) * 2.0 * M_PI / model.cols;
  const Eigen::Vector3d point =
      4.0 * Eigen::Vector3d(std::cos(elevation) * std::cos(azimuth), std::cos(elevation) * std::sin(azimuth),
                            std::sin(elevation));
  Eigen::Vector2d uv;
  REQUIRE(model.project(point, uv));
  REQUIRE(uv.x() == Catch::Approx(col).margin(1e-9));
  REQUIRE(uv.y() == Catch::Approx(row).margin(1e-9));

  const Eigen::Matrix<double, 2, 3> analytic = model.projection_jacobian(point);
  for (int axis = 0; axis < 3; ++axis) {
    Eigen::Vector3d step = Eigen::Vector3d::Zero();
    step[axis] = 1e-5;
    Eigen::Vector2d plus;
    Eigen::Vector2d minus;
    REQUIRE(model.project(point + step, plus));
    REQUIRE(model.project(point - step, minus));
    const Eigen::Vector2d numeric = (plus - minus) / 2e-5;
    REQUIRE(analytic(0, axis) == Catch::Approx(numeric.x()).margin(1e-6));
    // The vertical scale is linearized over the field of view, as in COIN-LIO.
    REQUIRE(analytic(1, axis) == Catch::Approx(numeric.y()).epsilon(0.05).margin(1e-6));
  }
}

TEST_CASE("brightness normalization divides by the local mean", "[photometric]") {
  rko_lio::core::FloatImage image(5, 9, 20.0F);
  rko_lio::core::normalize_brightness(image, 3, 3);
  for (const float value : image.data) {
    REQUIRE(value == Catch::Approx(140.0 * 20.0 / 21.0));
  }
}

TEST_CASE("every deskewed point projects back to the pixel it was captured at", "[photometric]") {
  const LidarImageModel model = make_model();
  std::vector<Eigen::Vector3d> cloud;
  std::vector<float> intensities;
  render_corridor(model, Sophus::SE3d(), cloud, intensities);
  const auto frame = frame_of(model, plain_image_config(), cloud, intensities);
  REQUIRE(frame.valid());
  int checked = 0;
  for (int row = 2; row < model.rows - 2; row += 3) {
    for (int col = 5; col < model.cols - 5; col += 17) {
      const int index = model.cloud_index(row, col);
      rko_lio::core::CapturedProjection projection;
      REQUIRE(rko_lio::core::project_captured(frame, model, frame.points_end[index], false, projection));
      REQUIRE(projection.uv.x() == Catch::Approx(col).margin(1e-6));
      REQUIRE(projection.uv.y() == Catch::Approx(row).margin(1e-6));
      ++checked;
    }
  }
  REQUIRE(checked > 100);
}

TEST_CASE("photometric terms alone recover motion along a featureless corridor axis", "[photometric]") {
  const LidarImageModel model = make_model();
  const IntensityImageConfig image_config = plain_image_config();
  rko_lio::core::PhotometricFeatureConfig feature_config;
  feature_config.num_features = 80;
  feature_config.grad_min = 2.0;
  feature_config.margin = 4;
  feature_config.min_range_m = 0.5;
  feature_config.ncc_threshold = 0.5;

  // Reference scan at the origin; patches along the corridor axis (x).
  std::vector<Eigen::Vector3d> cloud;
  std::vector<float> intensities;
  render_corridor(model, Sophus::SE3d(), cloud, intensities);
  const auto reference = frame_of(model, image_config, cloud, intensities);
  rko_lio::core::PhotometricFeatureManager manager(feature_config);
  manager.update(reference, model, Sophus::SE3d(), {Eigen::Vector3d::UnitX()});
  REQUIRE(manager.features().size() > 20);

  // The sensor moved 0.5 m along the corridor; start the solve 0.2 m short.
  const Sophus::SE3d truth(Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, 0.05)), Eigen::Vector3d(0.5, 0.1, 0.0));
  render_corridor(model, truth, cloud, intensities);
  const auto current = frame_of(model, image_config, cloud, intensities);
  Sophus::SE3d estimate(truth.so3(), truth.translation() - Eigen::Vector3d(0.2, 0.0, 0.0));
  const double error_before = (estimate.translation() - truth.translation()).norm();
  for (int iteration = 0; iteration < 15; ++iteration) {
    const auto system = rko_lio::core::build_photometric_system(manager.features(), current, model, feature_config,
                                                                estimate, Sophus::SE3d());
    REQUIRE(system.patches > 10);
    // Only translation along x is solved here: the corridor geometry pins the rest.
    const double dx = -system.b(0) / system.H(0, 0);
    estimate = Sophus::SE3d::exp((Eigen::Matrix<double, 6, 1>() << dx, 0, 0, 0, 0, 0).finished()) * estimate;
  }
  const double error_after = (estimate.translation() - truth.translation()).norm();
  REQUIRE(error_before == Catch::Approx(0.2));
  REQUIRE(error_after < 0.02);
}

TEST_CASE("weak directions are the axes few surface normals face", "[photometric]") {
  // Floor, ceiling and side walls of a corridor along x: nothing faces x.
  std::vector<Eigen::Vector3d> normals;
  for (int i = 0; i < 200; ++i) {
    normals.emplace_back(0.0, 1.0, 0.0);
    normals.emplace_back(0.0, 0.0, 1.0);
  }
  normals.emplace_back(1.0, 0.0, 0.0);
  const auto weak = rko_lio::core::weak_translation_directions(normals, 25.0);
  REQUIRE(weak.size() == 1);
  REQUIRE(std::abs(weak[0].x()) == Catch::Approx(1.0).margin(1e-6));
}
