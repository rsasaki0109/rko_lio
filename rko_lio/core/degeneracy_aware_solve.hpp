/*
 * MIT License
 *
 * Copyright (c) 2026 Sasaki
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

#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>

namespace rko_lio::core {

struct DegeneracyAwareSolveConfig {
  double well_conditioned_ratio = 1.0e-6;
  double multiplicity_relative_gap = 1.0e-8;
  double degenerate_prior_weight = 0.25;
};

struct DegeneracyAwareSolveResult {
  Eigen::Matrix<double, 6, 1> update = Eigen::Matrix<double, 6, 1>::Zero();
  int degenerate_count = 0;
  int non_observable_count = 0;
  bool used_prior = false;
  bool valid = false;
};

inline DegeneracyAwareSolveResult solve_degeneracy_aware(
    const Eigen::Matrix<double, 6, 6>& H,
    const Eigen::Matrix<double, 6, 1>& b,
    const Eigen::Matrix<double, 6, 1>& prior_update,
    const DegeneracyAwareSolveConfig& config = {}) {
  DegeneracyAwareSolveResult result;
  if (!H.allFinite() || !b.allFinite() || !prior_update.allFinite()) {
    return result;
  }

  const Eigen::Matrix<double, 6, 6> symmetric = 0.5 * (H + H.transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(symmetric);
  if (solver.info() != Eigen::Success) {
    return result;
  }
  const Eigen::Matrix<double, 6, 1> eigenvalues = solver.eigenvalues();
  const Eigen::Matrix<double, 6, 6> eigenvectors = solver.eigenvectors();
  const double trace = eigenvalues.sum();
  constexpr double kEigenvalueFloor = 1.0e-12;
  if (!(trace > kEigenvalueFloor) || !eigenvalues.allFinite() || !eigenvectors.allFinite()) {
    return result;
  }

  std::array<double, 6> contribution{};
  std::array<bool, 6> weak{};
  std::array<int, 6> cluster{};
  std::array<int, 6> cluster_size{};
  for (int i = 0; i < 6; ++i) {
    contribution[i] = eigenvalues(i) / trace;
    weak[i] = contribution[i] < config.well_conditioned_ratio;
    if (i > 0) {
      const bool merge = weak[i] && weak[i - 1] &&
                         contribution[i] - contribution[i - 1] <= config.multiplicity_relative_gap;
      cluster[i] = merge ? cluster[i - 1] : cluster[i - 1] + 1;
    }
    ++cluster_size[cluster[i]];
  }

  const double prior_weight = std::max(0.0, std::min(1.0, config.degenerate_prior_weight));
  for (int i = 0; i < 6; ++i) {
    const Eigen::Matrix<double, 6, 1> axis = eigenvectors.col(i);
    const bool non_observable = weak[i] && cluster_size[cluster[i]] >= 2;
    double component = 0.0;
    if (!weak[i] && eigenvalues(i) > kEigenvalueFloor) {
      component = -axis.dot(b) / eigenvalues(i);
    } else if (non_observable) {
      ++result.non_observable_count;
    } else {
      ++result.degenerate_count;
      const double geometric_component =
          eigenvalues(i) > kEigenvalueFloor ? -axis.dot(b) / eigenvalues(i) : 0.0;
      component =
          (1.0 - prior_weight) * geometric_component + prior_weight * axis.dot(prior_update);
      result.used_prior = result.used_prior || prior_weight > 0.0;
    }
    result.update += component * axis;
  }

  result.valid = result.update.allFinite();
  if (!result.valid) {
    result.update.setZero();
    result.used_prior = false;
  }
  return result;
}

} // namespace rko_lio::core
