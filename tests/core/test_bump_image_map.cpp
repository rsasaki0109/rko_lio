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

#include <rko_lio/core/bump_image_map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <set>
#include <tuple>
#include <vector>

namespace {

using rko_lio::core::BumpImageMap;

// Ground with 5 cm relief: flat to a plane fit, but the relief fixes x, y and yaw.
double relief(const double x, const double y) {
  return 0.05 * std::sin(2.0 * M_PI * x / 0.7) * std::cos(2.0 * M_PI * y / 0.9);
}

std::vector<Eigen::Vector3d> ground(const double half_extent, const double step) {
  std::vector<Eigen::Vector3d> points;
  for (double x = -half_extent; x <= half_extent; x += step) {
    for (double y = -half_extent; y <= half_extent; y += step) {
      points.emplace_back(x, y, relief(x, y));
    }
  }
  return points;
}

BumpImageMap mapped_ground() {
  BumpImageMap map;
  const std::vector<Eigen::Vector3d> points = ground(4.0, 0.02);
  map.integrate(points, std::vector<double>(points.size(), 5.0));
  return map;
}

} // namespace

TEST_CASE("Bump image stores the surface relief", "[bump_image_map]") {
  const BumpImageMap map = mapped_ground();
  REQUIRE(!map.empty());
  const Eigen::Vector3d query(0.3, -0.4, relief(0.3, -0.4));
  const rko_lio::core::BumpVoxel* voxel = map.find(query);
  REQUIRE(voxel != nullptr);
  const Eigen::Vector3d p = voxel->image_from_world * query;
  double height = 0.0, dx = 0.0, dy = 0.0;
  REQUIRE(rko_lio::core::bump_height_and_gradient(*voxel, p.x() / 0.05, p.y() / 0.05, height, dx, dy));
  REQUIRE(std::abs(p.z() - height) < 0.01);
}

TEST_CASE("Bump image registration recovers an in-plane offset", "[bump_image_map]") {
  const BumpImageMap map = mapped_ground();
  const Sophus::SE3d truth(Sophus::SO3d::rotZ(0.01), Eigen::Vector3d(0.04, -0.03, 0.01));
  std::vector<Eigen::Vector3d> scan;
  for (const Eigen::Vector3d& p : ground(3.0, 0.1)) scan.push_back(truth.inverse() * p);

  Sophus::SE3d pose;
  for (int i = 0; i < 30; ++i) {
    const auto system = rko_lio::core::build_bump_image_system(map, scan, pose, 0.1);
    REQUIRE(system.residuals > 0);
    const Eigen::Matrix<double, 6, 1> dx = system.H.ldlt().solve(-system.b);
    pose = Sophus::SE3d::exp(dx) * pose;
  }
  REQUIRE((pose.translation() - truth.translation()).norm() < 0.005);
  REQUIRE((pose.so3().inverse() * truth.so3()).log().norm() < 0.002);
}

TEST_CASE("Informed sampling keeps the voxels with relief", "[bump_image_map]") {
  BumpImageMap map;
  // Relief for x > 0, a flat plane for x < 0.
  std::vector<Eigen::Vector3d> points;
  for (const Eigen::Vector3d& p : ground(4.0, 0.02)) points.emplace_back(p.x(), p.y(), p.x() >= 0.0 ? p.z() : 0.0);
  map.integrate(points, std::vector<double>(points.size(), 5.0));

  std::vector<Eigen::Vector3d> scan;
  for (const Eigen::Vector3d& p : points) {
    if (std::fmod(std::abs(p.x()), 0.1) < 0.02 && std::fmod(std::abs(p.y()), 0.1) < 0.02) scan.push_back(p);
  }
  const std::vector<Eigen::Vector3d> sampled =
      rko_lio::core::sample_informed_points(map, scan, Sophus::SE3d(), 20);
  // Every voxel keeps one point; only voxels with relief keep all of theirs.
  std::set<std::tuple<int, int, int>> flat_voxels;
  for (const Eigen::Vector3d& p : scan) {
    if (p.x() >= 0.0) continue;
    const Eigen::Vector3i v = rko_lio::core::point_to_voxel(p, 2.0);
    flat_voxels.emplace(v.x(), v.y(), v.z());
  }
  int relief_side = 0, flat_side = 0;
  for (const Eigen::Vector3d& p : sampled) (p.x() >= 0.0 ? relief_side : flat_side)++;
  REQUIRE(flat_side == static_cast<int>(flat_voxels.size()));
  REQUIRE(relief_side > flat_side);
}
