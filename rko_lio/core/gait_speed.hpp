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

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

namespace rko_lio::core {

/** Walking speed from the step rhythm of a handheld sensor (pedestrian dead reckoning).
 *  The cadence is the dominant frequency of the specific-force magnitude over the last
 *  few seconds; the step length is learned as the median of registered speed over cadence
 *  while the registration still runs free, and then frozen: once the gait drives the
 *  prediction, the registered speed follows it and would only teach the step length back. Where a scan says nothing along the direction of travel
 *  (a featureless tunnel), cadence times step length keeps the speed of the prediction
 *  instead of letting it decay to zero or run away. */
class GaitSpeedEstimator {
public:
  struct Config {
    /** Length (s) of the accelerometer window for the cadence. */
    double window_sec = 4.0;
    /** Cadence search band (steps/s). */
    double min_cadence = 1.2;
    double max_cadence = 3.0;
    /** Walking needs this share of the 0.3-6 Hz power at the cadence peak (+-0.1 Hz)... */
    double min_peak_fraction = 0.25;
    /** ... and this standard deviation (m/s^2) of the magnitude. */
    double min_magnitude_std = 0.5;
    /** Accepted step lengths (m) when learning; others are stalls or runaways. */
    double min_step_length = 0.5;
    double max_step_length = 1.0;
    /** Samples (scans with a plausible step length) learned before the step is frozen. */
    std::size_t min_calibration_samples = 100;
  };

  GaitSpeedEstimator() = default;
  explicit GaitSpeedEstimator(const Config& config) : _config(config) {}

  /** Add one specific-force sample (any frame; only its magnitude is used). */
  void add_imu(const double time, const Eigen::Vector3d& specific_force) {
    _samples.emplace_back(time, specific_force.norm());
    while (!_samples.empty() && _samples.front().first < time - _config.window_sec) {
      _samples.pop_front();
    }
  }

  /** Cadence (steps/s) when the window shows walking. */
  std::optional<double> cadence() const {
    if (_samples.size() < 32) return std::nullopt;
    const double span = _samples.back().first - _samples.front().first;
    if (span < 0.8 * _config.window_sec) return std::nullopt;
    double mean = 0.0;
    for (const auto& [t, a] : _samples) mean += a;
    mean /= static_cast<double>(_samples.size());
    double variance = 0.0;
    for (const auto& [t, a] : _samples) variance += (a - mean) * (a - mean);
    variance /= static_cast<double>(_samples.size());
    if (std::sqrt(variance) < _config.min_magnitude_std) return std::nullopt;
    // Hann-windowed DFT on a 0.01 Hz grid over 0.3-6 Hz (samples need not be uniform).
    const double t0 = _samples.front().first;
    auto power = [&](const double f) {
      double re = 0.0, im = 0.0;
      for (const auto& [t, a] : _samples) {
        const double u = (t - t0) / span;
        const double w = 0.5 - 0.5 * std::cos(2.0 * M_PI * u);
        const double phase = 2.0 * M_PI * f * (t - t0);
        re += w * (a - mean) * std::cos(phase);
        im -= w * (a - mean) * std::sin(phase);
      }
      return re * re + im * im;
    };
    double total = 0.0, best = 0.0, best_f = 0.0;
    std::vector<std::pair<double, double>> spectrum;
    for (double f = 0.3; f <= 6.0; f += 0.01) {
      const double p = power(f);
      spectrum.emplace_back(f, p);
      total += p;
      if (f >= _config.min_cadence && f <= _config.max_cadence && p > best) {
        best = p;
        best_f = f;
      }
    }
    double peak = 0.0;
    for (const auto& [f, p] : spectrum) {
      if (std::abs(f - best_f) <= 0.1) peak += p;
    }
    if (total <= 0.0 || peak / total < _config.min_peak_fraction) return std::nullopt;
    return best_f;
  }

  /** Whether a step length (m) of registered speed over cadence is plausible. */
  bool plausible_step(const double step) const {
    return step >= _config.min_step_length && step <= _config.max_step_length;
  }

  /** Learn from a registered speed (m/s) with the cadence seen then; ignored once frozen. */
  void observe(const double registered_speed, const double cadence) {
    if (_frozen || cadence <= 0.0) return;
    const double step = registered_speed / cadence;
    if (!plausible_step(step)) return;
    _steps.push_back(step);
    if (_steps.size() >= std::max<std::size_t>(1, _config.min_calibration_samples)) {
      std::vector<double> values = _steps;
      std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
      _frozen = values[values.size() / 2];
    }
  }

  /** The frozen step length (m), once learned. */
  std::optional<double> step_length() const { return _frozen; }

  void reset() {
    _samples.clear();
    _steps.clear();
    _frozen.reset();
  }

private:
  Config _config;
  std::deque<std::pair<double, double>> _samples;
  std::vector<double> _steps;
  std::optional<double> _frozen;
};

/** The velocity at the gait speed along the unit direction of travel u. By default only the
 *  component along u is replaced and the rest kept from the registration; with set_speed the
 *  result is also scaled to the gait speed, so the registration gives the direction and the
 *  gait the speed. Keeping the rest lets a turn of the sensor pump the speed: the old forward
 *  speed stays as a sideways component while the new direction gets the gait speed again. */
inline Eigen::Vector3d apply_gait_speed(const Eigen::Vector3d& velocity, const Eigen::Vector3d& u,
                                        const double speed, const bool set_speed) {
  Eigen::Vector3d result = velocity + (speed - velocity.dot(u)) * u;
  if (set_speed && speed > 0.0) result *= speed / result.norm();
  return result;
}

} // namespace rko_lio::core
