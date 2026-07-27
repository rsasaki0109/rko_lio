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

// Localizability-aware ICP correspondence weighting (X-ICP-inspired) for soft
// along-corridor degeneracy. In a self-similar tunnel the ICP information
// along the travel axis is real but tiny: measured on the NTNU tunnel, only
// ~1.3-7.8% of planar points carry normals with |n . axis| > 0.3 (door
// frames, niches, drains), and the mean per-point (n . axis)^2 is ~0.015, so
// the axis-observing minority is swamped by the axis-neutral wall majority
// whose nearest-neighbor associations act as a zero-motion attractor.
// Multiplying each correspondence by 1 + boost * (n . axis)^2 raises the
// effective information share of the axis-observing points without touching
// anything else. Pure functions only, unit-testable in isolation like
// gravity_alignment.hpp / intensity_profile.hpp.

#pragma once

#include <Eigen/Core>

#include <optional>

namespace rko_lio::core {

/**
 * Correspondence weight for localizability-aware ICP.
 *
 * `normal` is the (orientation-arbitrary) unit surface normal of the matched
 * map neighborhood, if one is available; `axis` is the unit weak/travel
 * direction in the same (world) frame; `boost` scales how strongly
 * axis-observing surfaces are amplified. Returns 1.0 (neutral) when there is
 * no normal, a non-positive boost, or a degenerate axis, so enabling the
 * machinery without normals or motion changes nothing.
 */
inline double localizability_weight(const std::optional<Eigen::Vector3d>& normal,
                                    const Eigen::Vector3d& axis,
                                    const double boost) {
  if (!normal.has_value() || !(boost > 0.0)) {
    return 1.0;
  }
  const double axis_norm_sq = axis.squaredNorm();
  if (axis_norm_sq < 1.0e-12) {
    return 1.0;
  }
  const double cosine = normal->dot(axis) / std::sqrt(axis_norm_sq);
  return 1.0 + boost * cosine * cosine;
}

} // namespace rko_lio::core
