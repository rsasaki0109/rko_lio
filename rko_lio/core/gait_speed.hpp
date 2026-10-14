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

  /** Standard deviation (m/s^2) of the magnitude over the last ``window_sec`` (0 with fewer
   *  than two samples). Steps shake a handheld sensor; standing hardly does. */
  double magnitude_std(const double window_sec) const {
    if (_samples.empty()) return 0.0;
    const double start = _samples.back().first - window_sec;
    double sum = 0.0, sum_squares = 0.0;
    std::size_t count = 0;
    for (auto it = _samples.rbegin(); it != _samples.rend() && it->first > start; ++it) {
      sum += it->second;
      sum_squares += it->second * it->second;
      ++count;
    }
    if (count < 2) return 0.0;
    const double mean = sum / static_cast<double>(count);
    return std::sqrt(std::max(0.0, sum_squares / static_cast<double>(count) - mean * mean));
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

/** Step events of a walker from the specific force magnitude: a causal 0.8-4 Hz band-pass
 *  (Butterworth high- and low-pass biquads at the measured sample rate) and its peaks, each
 *  confirmed ``confirm_sec`` later. A peak counts above ``relative_amplitude`` times the running
 *  median of the counted peaks (and ``min_amplitude``). The 4 s cadence window lags a stop or a
 *  restart by about 2 s; the steps themselves do not. */
class StepDetector {
public:
  struct Config {
    double low_hz = 0.8;
    double high_hz = 4.0;
    double confirm_sec = 0.15;
    double min_interval_sec = 0.3;
    double min_amplitude = 0.3;
    double relative_amplitude = 0.25;
    std::size_t typical_peaks = 40;
    std::size_t min_typical_peaks = 10;
  };

  StepDetector() = default;
  explicit StepDetector(const Config& config) : _config(config) {}

  void add(const double time, const double magnitude) {
    if (_last_time && time <= *_last_time) return;
    if (!_ready) {
      if (_last_time) _intervals.push_back(time - *_last_time);
      _last_time = time;
      if (_intervals.size() >= 50) design();
      return;
    }
    _last_time = time;
    const double y = _low.step(_high.step(magnitude));
    _recent.emplace_back(time, y);
    // Every sample at least confirm_sec old is decided once: a peak if it is the largest of the
    // +-confirm_sec around it. IMU stamps come in bursts, so several can become due at once.
    const double confirm = _config.confirm_sec;
    for (const auto& [candidate_time, candidate_value] : _recent) {
      if (candidate_time > time - confirm) break;
      if (_last_decided && candidate_time <= *_last_decided) continue;
      _last_decided = candidate_time;
      if (candidate_time - _recent.front().first < confirm) continue;  // its past is not all here
      bool peak = true;
      for (const auto& [t, v] : _recent) {
        if (t < candidate_time - confirm) continue;
        if (t > candidate_time + confirm) break;
        if (v > candidate_value) {
          peak = false;
          break;
        }
      }
      if (peak) count(candidate_time, candidate_value);
    }
    while (!_recent.empty() && _recent.front().first < time - 3.0 * confirm) _recent.pop_front();
  }

  /** Time of the last counted step. */
  std::optional<double> last_step() const { return _last_step; }

private:
  void count(const double peak_time, const double amplitude) {
    if (_last_step && peak_time - *_last_step < _config.min_interval_sec) return;
    double threshold = _config.min_amplitude;
    if (_peaks.size() >= _config.min_typical_peaks) {
      std::vector<double> values(_peaks.begin(), _peaks.end());
      std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
      threshold = std::max(threshold, _config.relative_amplitude * values[values.size() / 2]);
    }
    if (amplitude <= threshold) return;
    _last_step = peak_time;
    _peaks.push_back(amplitude);
    while (_peaks.size() > _config.typical_peaks) _peaks.pop_front();
  }

  struct Biquad {
    double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
    double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
    bool primed = false;
    double step(const double x) {
      if (!primed) {
        // Start from rest at the first input (the high-pass otherwise rings on gravity).
        x1 = x2 = x;
        y1 = y2 = x * (b0 + b1 + b2) / (1.0 + a1 + a2);
        primed = true;
      }
      const double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1;
      x1 = x;
      y2 = y1;
      y1 = y;
      return y;
    }
  };

  // Butterworth (Q = 1/sqrt 2) biquads from the RBJ cookbook.
  static Biquad design_biquad(const double cutoff_hz, const double sample_hz, const bool high_pass) {
    const double w0 = 2.0 * M_PI * cutoff_hz / sample_hz;
    const double alpha = std::sin(w0) / std::sqrt(2.0);
    const double c = std::cos(w0);
    const double a0 = 1.0 + alpha;
    Biquad q;
    if (high_pass) {
      q.b0 = (1.0 + c) / 2.0 / a0;
      q.b1 = -(1.0 + c) / a0;
      q.b2 = (1.0 + c) / 2.0 / a0;
    } else {
      q.b0 = (1.0 - c) / 2.0 / a0;
      q.b1 = (1.0 - c) / a0;
      q.b2 = (1.0 - c) / 2.0 / a0;
    }
    q.a1 = -2.0 * c / a0;
    q.a2 = (1.0 - alpha) / a0;
    return q;
  }

  void design() {
    std::vector<double> values(_intervals.begin(), _intervals.end());
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    const double dt = values[values.size() / 2];
    if (dt <= 0.0) return;
    const double sample_hz = 1.0 / dt;
    const double high_hz = std::min(_config.high_hz, 0.45 * sample_hz);
    _high = design_biquad(_config.low_hz, sample_hz, true);
    _low = design_biquad(high_hz, sample_hz, false);
    _ready = true;
  }

  Config _config;
  bool _ready = false;
  std::optional<double> _last_time;
  std::vector<double> _intervals;
  Biquad _high, _low;
  std::deque<std::pair<double, double>> _recent;
  std::optional<double> _last_decided;
  std::deque<double> _peaks;
  std::optional<double> _last_step;
};

/** The velocity of a walker whose cadence has dropped out (turning, stopping) from the
 *  registered ``velocity``: zero while ``standing``, otherwise never backwards along the
 *  direction of travel ``u`` and no faster than ``last_speed``, the last gait speed. Without
 *  steps there is nothing to speed up with, while a registration with no hold along the
 *  travel (a tunnel) runs away with its own prediction. */
inline Eigen::Vector3d hold_gait_speed(const Eigen::Vector3d& velocity, const Eigen::Vector3d& u,
                                       const double last_speed, const bool standing, const bool set_speed) {
  if (standing) return Eigen::Vector3d::Zero();
  const double along = std::clamp(velocity.dot(u), 0.0, last_speed);
  Eigen::Vector3d result = apply_gait_speed(velocity, u, along, set_speed);
  const double speed = result.norm();
  if (speed > last_speed) result *= last_speed / speed;
  return result;
}

} // namespace rko_lio::core
