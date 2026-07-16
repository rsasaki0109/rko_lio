/*
 * MIT License
 *
 * Copyright (c) 2025 Meher V.R. Malladi.
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
#include <cstddef>
#include <deque>
#include <limits>

namespace rko_lio::core {

struct PersistentWeakDirectionConfig {
  std::size_t min_consecutive_scans = 3;
  double min_absolute_cosine = 0.98;
  double min_translation_fraction = 0.99;
  bool require_multiscan_observability = false;
  std::size_t observability_window_scans = 10;
  std::size_t observability_min_scans = 5;
  double max_aggregate_directional_information_ratio = 1.0e-6;
};

struct PersistentWeakDirectionState {
  Eigen::Matrix<double, 6, 1> axis = Eigen::Matrix<double, 6, 1>::Zero();
  std::size_t consecutive_scans = 0;
  double matched_absolute_cosine = 0.0;
  bool candidate_available = false;
  std::size_t observability_window_scans = 0;
  double aggregate_directional_information_ratio =
      std::numeric_limits<double>::infinity();
  bool multiscan_observability_confirmed = false;
  bool confirmed = false;
};

class PersistentWeakDirectionTracker {
public:
  PersistentWeakDirectionState observe(const Eigen::Matrix<double, 6, 6>& H,
                                       const double well_conditioned_ratio,
                                       const double multiplicity_relative_gap,
                                       const PersistentWeakDirectionConfig& raw_config = {}) {
    const PersistentWeakDirectionConfig config = sanitized(raw_config);
    const Eigen::Matrix<double, 6, 6> symmetric = 0.5 * (H + H.transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(symmetric);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
        !solver.eigenvectors().allFinite()) {
      reset();
      return state_;
    }

    const Eigen::Matrix<double, 6, 1> eigenvalues = solver.eigenvalues();
    const double trace = eigenvalues.sum();
    if (!(trace > 1.0e-12)) {
      reset();
      return state_;
    }

    std::array<double, 6> contribution{};
    std::array<bool, 6> weak{};
    std::array<int, 6> cluster{};
    std::array<int, 6> cluster_size{};
    for (int i = 0; i < 6; ++i) {
      contribution[i] = eigenvalues(i) / trace;
      weak[i] = contribution[i] < well_conditioned_ratio;
      if (i > 0) {
        const bool merge = weak[i] && weak[i - 1] &&
                           contribution[i] - contribution[i - 1] <= multiplicity_relative_gap;
        cluster[i] = merge ? cluster[i - 1] : cluster[i - 1] + 1;
      }
      ++cluster_size[cluster[i]];
    }

    Eigen::Matrix<double, 6, 1> candidate = Eigen::Matrix<double, 6, 1>::Zero();
    bool candidate_available = false;
    double best_match = -1.0;
    for (int i = 0; i < 6; ++i) {
      if (!weak[i] || cluster_size[cluster[i]] != 1) {
        continue;
      }
      const Eigen::Matrix<double, 6, 1> axis = solver.eigenvectors().col(i).normalized();
      if (axis.head<3>().squaredNorm() < config.min_translation_fraction) {
        continue;
      }
      const double match = state_.candidate_available ? std::abs(state_.axis.dot(axis)) : 0.0;
      if (!candidate_available || match > best_match) {
        candidate = axis;
        candidate_available = true;
        best_match = match;
      }
    }

    if (!candidate_available) {
      reset();
      return state_;
    }

    const double matched_cosine = state_.candidate_available ? std::abs(state_.axis.dot(candidate)) : 0.0;
    if (state_.candidate_available && matched_cosine >= config.min_absolute_cosine) {
      if (state_.axis.dot(candidate) < 0.0) {
        candidate = -candidate;
      }
      ++state_.consecutive_scans;
      state_.matched_absolute_cosine = matched_cosine;
    } else {
      normalized_information_window_.clear();
      state_.consecutive_scans = 1;
      state_.matched_absolute_cosine = 0.0;
    }

    normalized_information_window_.push_back(symmetric / trace);
    while (normalized_information_window_.size() > config.observability_window_scans) {
      normalized_information_window_.pop_front();
    }

    state_.axis = candidate;
    state_.candidate_available = true;
    state_.observability_window_scans = normalized_information_window_.size();
    Eigen::Matrix<double, 6, 6> aggregate = Eigen::Matrix<double, 6, 6>::Zero();
    for (const auto& normalized_information : normalized_information_window_) {
      aggregate += normalized_information;
    }
    const double aggregate_trace = aggregate.trace();
    if (aggregate_trace > 1.0e-12) {
      state_.aggregate_directional_information_ratio =
          candidate.dot(aggregate * candidate) / aggregate_trace;
    }
    state_.multiscan_observability_confirmed =
        state_.observability_window_scans >= config.observability_min_scans &&
        state_.aggregate_directional_information_ratio <=
            config.max_aggregate_directional_information_ratio;
    state_.confirmed = state_.consecutive_scans >= config.min_consecutive_scans &&
                       (!config.require_multiscan_observability ||
                        state_.multiscan_observability_confirmed);
    return state_;
  }

  void reset() {
    state_ = PersistentWeakDirectionState{};
    normalized_information_window_.clear();
  }

  const PersistentWeakDirectionState& state() const { return state_; }

private:
  static PersistentWeakDirectionConfig sanitized(PersistentWeakDirectionConfig config) {
    config.min_consecutive_scans = std::max<std::size_t>(1, config.min_consecutive_scans);
    config.min_absolute_cosine = std::max(0.0, std::min(1.0, config.min_absolute_cosine));
    config.min_translation_fraction =
        std::max(0.0, std::min(1.0, config.min_translation_fraction));
    config.observability_window_scans =
        std::max<std::size_t>(1, config.observability_window_scans);
    config.observability_min_scans =
        std::max<std::size_t>(1, std::min(config.observability_min_scans,
                                         config.observability_window_scans));
    config.max_aggregate_directional_information_ratio =
        std::max(0.0, config.max_aggregate_directional_information_ratio);
    return config;
  }

  PersistentWeakDirectionState state_{};
  std::deque<Eigen::Matrix<double, 6, 6>> normalized_information_window_;
};

} // namespace rko_lio::core
