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

#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>

namespace rko_lio::core {

/** Decides when a bump-image registration falls back to point-to-point.
 *
 *  A scan exceeds the gate when its bump result turns the pose more than
 *  ``limit_deg`` away from the IMU prediction. It falls back only when at least
 *  ``min_count`` of the last ``window`` scans (itself included) exceeded, or when
 *  it exceeded ``hard_limit_deg`` (0 disables). A window and count of 1 fall back
 *  on every exceeding scan. */
class BumpRotationGate {
public:
  BumpRotationGate(const double limit_deg, const int window, const int min_count, const double hard_limit_deg)
      : _limit_deg(limit_deg),
        _window(static_cast<std::size_t>(std::max(1, window))),
        _min_count(static_cast<std::size_t>(std::max(1, min_count))),
        _hard_limit_deg(hard_limit_deg) {}

  /** Record this scan's rotation correction (deg); true when it should fall back. */
  bool fall_back(const double rotation_deg) {
    const bool exceeded = rotation_deg > _limit_deg;
    _recent.push_back(exceeded);
    while (_recent.size() > _window) {
      _recent.pop_front();
    }
    if (!exceeded) {
      return false;
    }
    const auto count = static_cast<std::size_t>(std::count(_recent.begin(), _recent.end(), true));
    return count >= _min_count || (_hard_limit_deg > 0.0 && rotation_deg > _hard_limit_deg);
  }

private:
  double _limit_deg;
  std::size_t _window;
  std::size_t _min_count;
  double _hard_limit_deg;
  std::deque<bool> _recent;
};

} // namespace rko_lio::core
