/*
 * MIT License
 *
 * Copyright (c) 2026 Sasaki
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace rko_lio::core {

/** One hypothesis from any discrete correlation search.
 *
 * The selector is deliberately independent of intensity profiles and spatial
 * dimensionality. A 1D shift search, an oriented height image, or another
 * appearance matcher can all expose their hypotheses through this type.
 */
struct CorrelationPeakCandidate {
  long offset_index = 0;
  double score = -1.0;
  std::size_t support = 0;
  std::array<long, 3> offset_coordinates{};
  std::size_t offset_dimensions = 0;
};

struct CorrelationPeakPolicy {
  double minimum_score = 0.6;
  std::size_t minimum_support = 1;
  std::size_t secondary_exclusion_radius = 1;
  double minimum_peak_margin = 0.0;
};

enum class CorrelationPeakRejection {
  none,
  no_candidates,
  insufficient_support,
  low_score,
  ambiguous,
};

struct CorrelationPeakSelection {
  bool accepted = false;
  CorrelationPeakCandidate best;
  std::optional<CorrelationPeakCandidate> second_best;
  double peak_margin = 0.0;
  CorrelationPeakRejection rejection =
      CorrelationPeakRejection::no_candidates;
};

inline bool correlation_peak_offsets_are_neighbours(
    const CorrelationPeakCandidate& lhs,
    const CorrelationPeakCandidate& rhs,
    const std::size_t radius) {
  if (lhs.offset_dimensions == 0 || rhs.offset_dimensions == 0) {
    return std::abs(lhs.offset_index - rhs.offset_index) <=
           static_cast<long>(radius);
  }
  if (lhs.offset_dimensions != rhs.offset_dimensions ||
      lhs.offset_dimensions > lhs.offset_coordinates.size()) {
    return false;
  }
  for (std::size_t dimension = 0; dimension < lhs.offset_dimensions;
       ++dimension) {
    if (std::abs(lhs.offset_coordinates[dimension] -
                 rhs.offset_coordinates[dimension]) >
        static_cast<long>(radius)) {
      return false;
    }
  }
  return true;
}

/** Select the best supported correlation hypothesis and reject aliases.
 *
 * Immediate neighbours of the best discrete offset describe the width of the
 * same peak and are excluded from the competing-peak search. The radius is a
 * policy input so matchers with different sampling resolution can reuse this
 * selector without changing its implementation.
 */
inline CorrelationPeakSelection select_correlation_peak(
    const std::vector<CorrelationPeakCandidate>& candidates,
    const CorrelationPeakPolicy& policy) {
  CorrelationPeakSelection result;
  if (candidates.empty()) {
    return result;
  }

  result.best = *std::max_element(
      candidates.begin(), candidates.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.score < rhs.score; });
  for (const auto& candidate : candidates) {
    if (candidate.support < policy.minimum_support ||
        correlation_peak_offsets_are_neighbours(
            candidate,
            result.best,
            policy.secondary_exclusion_radius)) {
      continue;
    }
    if (!result.second_best.has_value() ||
        candidate.score > result.second_best->score) {
      result.second_best = candidate;
    }
  }

  result.peak_margin = result.second_best.has_value()
                           ? result.best.score - result.second_best->score
                           : 0.0;
  if (result.best.support < policy.minimum_support) {
    result.rejection = CorrelationPeakRejection::insufficient_support;
    return result;
  }
  if (result.best.score < policy.minimum_score) {
    result.rejection = CorrelationPeakRejection::low_score;
    return result;
  }
  if (result.second_best.has_value() &&
      result.peak_margin <
          std::max(0.0, policy.minimum_peak_margin)) {
    result.rejection = CorrelationPeakRejection::ambiguous;
    return result;
  }
  result.accepted = true;
  result.rejection = CorrelationPeakRejection::none;
  return result;
}

}  // namespace rko_lio::core
