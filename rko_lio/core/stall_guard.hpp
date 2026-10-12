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


// Speed hold for a vehicle whose registration stalls.
//
// In a featureless tunnel the registration of a 16-beam LiDAR can lock to the
// map behind it: the speed falls from 9 m/s to zero within two seconds and
// stays there for up to a minute while the vehicle drives on. A car cannot
// brake like that without the accelerometer seeing it. Over two seconds the
// accelerometer's speed change is good to about 2 m/s even with a few degrees
// of tilt error leaking gravity into it; integrated over a minute it is not.
// So the guard compares the registered speed with the accelerometer only over
// a short window, and while the registration stays stalled it holds the speed
// from before the drop instead of integrating the accelerometer.

#pragma once

#include <cmath>
#include <deque>

namespace rko_lio::core {

struct StallGuardConfig {
  /** Window (s) over which the registered speed drop is compared with the accelerometer. */
  double window_sec = 2.0;
  /** Detect a stall when the registered speed falls this much (m/s) below the speed the
   *  accelerometer allows over the window. */
  double min_drop_mps = 3.0;
  /** ...and to below this fraction of it: the registration locks to the map behind and
   *  the speed collapses towards zero, while a real slowdown keeps most of it. */
  double max_speed_ratio = 0.5;
  /** Stay out of the way below this speed (m/s): the direction of travel is unreliable. */
  double min_speed_mps = 3.0;
  /** Release the hold when the registered speed agrees with it this closely (m/s)... */
  double exit_agreement_mps = 1.0;
  /** ...on this many consecutive scans. */
  int exit_scans = 10;
  /** Release the hold after this long (s) regardless. */
  double max_hold_sec = 120.0;
};

class StallGuard {
public:
  /**
   * Feed one scan and get the speed along the direction of travel to use for it.
   *
   * `registered_speed`: the registration's speed along the previous direction of travel (m/s).
   * `accel_along`: the gravity-compensated measured acceleration along that direction (m/s^2).
   * `previous_speed`: the speed of the state before this scan (m/s).
   */
  double update(const double time_sec, const double dt, const double registered_speed,
                const double accel_along, const double previous_speed, const StallGuardConfig& config) {
    _accel_integral += accel_along * dt;
    _recent.push_back(registered_speed);
    if (_recent.size() > kRecentScans) _recent.pop_front();
    double speed = registered_speed;
    if (_holding) {
      _agreement = std::abs(registered_speed - _hold_speed) <= config.exit_agreement_mps ? _agreement + 1 : 0;
      if (_agreement >= config.exit_scans || time_sec - _hold_start_sec > config.max_hold_sec) {
        _holding = false;
      } else {
        speed = _hold_speed;
      }
    } else if (previous_speed >= config.min_speed_mps && !_history.empty() &&
               time_sec - _history.front().time_sec >= 0.5 * config.window_sec) {
      // Reference: the first scans of the window, averaged against single-scan noise.
      double reference_speed = 0.0;
      double reference_integral = 0.0;
      std::size_t count = 0;
      for (const auto& sample : _history) {
        if (count == kRecentScans) break;
        reference_speed += sample.speed;
        reference_integral += sample.accel_integral;
        ++count;
      }
      reference_speed /= static_cast<double>(count);
      reference_integral /= static_cast<double>(count);
      const double allowed_speed = reference_speed + (_accel_integral - reference_integral);
      const double recent = recent_speed();
      if (allowed_speed - recent > config.min_drop_mps && recent < config.max_speed_ratio * allowed_speed) {
        _holding = true;
        _hold_speed = allowed_speed;
        _hold_start_sec = time_sec;
        _agreement = 0;
        ++hold_count;
        speed = _hold_speed;
      }
    }
    if (!_holding && previous_speed < config.min_speed_mps) _history.clear();
    _history.push_back({time_sec, speed, _accel_integral});
    while (!_history.empty() && time_sec - _history.front().time_sec > config.window_sec) _history.pop_front();
    held_scan_count += _holding ? 1U : 0U;
    return speed;
  }

  bool holding() const { return _holding; }

  /** Stalls detected, and scans whose speed was held. */
  std::size_t hold_count = 0;
  std::size_t held_scan_count = 0;

private:
  static constexpr std::size_t kRecentScans = 3;
  struct Sample {
    double time_sec;
    double speed;
    double accel_integral;
  };
  double recent_speed() const {
    double sum = 0.0;
    for (const double speed : _recent) sum += speed;
    return sum / static_cast<double>(_recent.size());
  }
  std::deque<Sample> _history;
  std::deque<double> _recent;
  double _accel_integral = 0.0;
  bool _holding = false;
  double _hold_speed = 0.0;
  double _hold_start_sec = 0.0;
  int _agreement = 0;
};

} // namespace rko_lio::core
