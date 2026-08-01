/*
 * MIT License
 *
 * Copyright (c) 2026 Sasaki
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>

namespace rko_lio::core {

/** Streaming overlap-local Pearson correlation.
 *
 * Matchers add only samples that overlap for the candidate transformation.
 * The resulting score is bounded to [-1, 1]. A missing result means fewer
 * than two samples or zero variance in either channel.
 */
class PearsonCorrelationAccumulator {
public:
  void add(const double value_a, const double value_b) {
    sum_a_ += value_a;
    sum_b_ += value_b;
    sum_aa_ += value_a * value_a;
    sum_bb_ += value_b * value_b;
    sum_ab_ += value_a * value_b;
    ++support_;
  }

  [[nodiscard]] std::size_t support() const { return support_; }

  [[nodiscard]] std::optional<double> score() const {
    if (support_ < 2) {
      return std::nullopt;
    }
    const double count = static_cast<double>(support_);
    const double covariance = sum_ab_ - sum_a_ * sum_b_ / count;
    const double variance_a = sum_aa_ - sum_a_ * sum_a_ / count;
    const double variance_b = sum_bb_ - sum_b_ * sum_b_ / count;
    const double denominator =
        std::sqrt(std::max(0.0, variance_a) *
                  std::max(0.0, variance_b));
    if (denominator <= 1.0e-12) {
      return std::nullopt;
    }
    return std::clamp(covariance / denominator, -1.0, 1.0);
  }

private:
  double sum_a_ = 0.0;
  double sum_b_ = 0.0;
  double sum_aa_ = 0.0;
  double sum_bb_ = 0.0;
  double sum_ab_ = 0.0;
  std::size_t support_ = 0;
};

}  // namespace rko_lio::core
