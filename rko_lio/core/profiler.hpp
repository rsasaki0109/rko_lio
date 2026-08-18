/*
 * MIT License
 *
 * Copyright (c) 2025 Meher V.R. Malladi.
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

/**
 * @file profiler.hpp
 * @brief Utility classes for scoped timing and profiling measurements.
 *
 * @deprecated This profiling utility is deprecated and scheduled for removal in a future release.
 * A more fully featured alternative, likely tracy, will be used instead.
 *
 * Contains the ScopedProfiler RAII class for timing named code blocks,
 * a simple Timer class for quick timing prints, and the SCOPED_PROFILER macro for convenience.
 */
#pragma once
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rko_lio::core {

/**
 * Scoped RAII profiler that measures elapsed time for a named code block.
 *
 * This class records the execution time between construction and destruction (or explicit finish call)
 * and aggregates statistics including count, total time, and max time per named scope.
 *
 * The stats are saved in a static map. Will survive till the end of execution.
 *
 */
class ScopedProfiler {
public:
  explicit ScopedProfiler(std::string name_) : name(std::move(name_)), start(Clock::now()) {}
  ScopedProfiler(const ScopedProfiler&) = delete;
  ScopedProfiler(ScopedProfiler&&) = delete;
  ScopedProfiler& operator=(const ScopedProfiler&) = delete;
  ScopedProfiler& operator=(ScopedProfiler&&) = delete;

  /// Explicitly finish timing and update aggregated statistics.
  void finish() {
    if (!finished) {
      auto& entry = profile_data.map[name];
      const MilliSeconds elapsed = Clock::now() - start;

      ++entry.count;

      // Update Welford's online algorithm for variance
      const double delta = (elapsed - entry.mean).count();
      entry.mean += MilliSeconds{delta / entry.count};
      entry.M2 += delta * (elapsed - entry.mean).count();

      if (elapsed > entry.max_time) {
        entry.max_time = elapsed;
      }

      entry.total += elapsed;

      finished = true;
    }
  }

  /// Automatically finish timing on destruction if not already finished.
  ~ScopedProfiler() { finish(); }

  /// Print aggregated profiling results to stdout.
  static void print_results() { profile_data.print_results(); }

private:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;
  using Seconds = std::chrono::duration<double>;
  using MilliSeconds = std::chrono::duration<double, std::milli>;

  struct ProfilingInfo {
    size_t count{0};
    MilliSeconds total{};
    MilliSeconds max_time{};
    MilliSeconds mean{};
    double M2{0.0}; // Welford variance accumulator

    MilliSeconds stddev() const {
      if (count < 2) {
        return MilliSeconds{0.0};
      }
      return MilliSeconds{std::sqrt(M2 / (count - 1))};
    }
  };

  class ProfilingInfoMap {
  public:
    std::unordered_map<std::string, ProfilingInfo> map;
    ProfilingInfoMap() = default;
    ProfilingInfoMap(const ProfilingInfoMap&) = delete;
    ProfilingInfoMap(ProfilingInfoMap&&) = delete;
    ProfilingInfoMap& operator=(const ProfilingInfoMap&) = delete;
    ProfilingInfoMap& operator=(ProfilingInfoMap&&) = delete;
    void print_results() {
      if (!map.empty()) {
        std::cout << "Profiling results\n";
      }
      for (const auto& [name, info] : map) {
        std::cout << "\t" << name << ":\n"
                  << "\t\tExecution count: " << info.count << "\n"
                  << "\t\tAverage time: " << std::fixed << std::setprecision(2) << info.mean.count() << " ms\n"
                  << "\t\tStd. dev. time: " << info.stddev().count() << " ms\n"
                  << "\t\tAverage frequency: " << (1.0 / Seconds(info.mean).count()) << "Hz\n"
                  << "\t\tMax (worst-case) frequency: " << (1.0 / Seconds(info.max_time).count()) << "Hz\n";
      }
    }
    ~ProfilingInfoMap() { print_results(); }
  };
  static inline ProfilingInfoMap profile_data;

  std::string name;
  TimePoint start;
  bool finished = false;
};

/** Simple timing utility that prints elapsed time for a scope or code block. */
struct Timer {
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;
  using Duration = std::chrono::duration<double>;

  Timer() : label("Execution"), start_time(Clock::now()) {}
  explicit Timer(const std::string& label) : label(label), start_time(Clock::now()) {}
  Timer(const Timer&) = default;
  Timer(Timer&&) = default;
  Timer& operator=(const Timer&) = default;
  Timer& operator=(Timer&&) = default;
  ~Timer() {
    auto end_time = Clock::now();
    Duration duration = end_time - start_time;
    std::cout << label << " took " << duration.count() << " seconds.\n";
  }
  std::string label;
  TimePoint start_time;
};

/**
 * [instrumentation, additive-only] Lightweight gauge for tracking local-map
 * growth (active voxel count and stored point count) over the course of a
 * run, so map growth is visible alongside the ScopedProfiler timing output.
 *
 * This is purely observational: call sites push samples explicitly (see
 * `MapGrowthGauge::sample`), and nothing here reads back into or otherwise
 * influences any numerical computation. Samples are stored in a static map
 * (keyed by a caller-chosen name, e.g. "RegistrationMap") and a
 * first/median/last summary is printed to stdout at program exit, mirroring
 * ScopedProfiler's static-destructor reporting style.
 */
class MapGrowthGauge {
public:
  struct Sample {
    std::size_t scan_index;
    std::size_t active_voxels;
    std::size_t stored_points;
  };

  /// Record one growth sample under the given gauge name.
  static void sample(const std::string& name,
                     std::size_t scan_index,
                     std::size_t active_voxels,
                     std::size_t stored_points) {
    data().map[name].push_back({scan_index, active_voxels, stored_points});
  }

  /// Print aggregated gauge results to stdout.
  static void print_results() { data().print_results(); }

private:
  class GaugeData {
  public:
    std::unordered_map<std::string, std::vector<Sample>> map;
    GaugeData() = default;
    GaugeData(const GaugeData&) = delete;
    GaugeData(GaugeData&&) = delete;
    GaugeData& operator=(const GaugeData&) = delete;
    GaugeData& operator=(GaugeData&&) = delete;
    void print_results() const {
      if (!map.empty()) {
        std::cout << "Map growth gauge results\n";
      }
      for (const auto& [name, samples] : map) {
        if (samples.empty()) {
          continue;
        }
        std::vector<Sample> sorted_by_scan = samples;
        std::sort(sorted_by_scan.begin(), sorted_by_scan.end(),
                  [](const Sample& a, const Sample& b) { return a.scan_index < b.scan_index; });
        const Sample& first = sorted_by_scan.front();
        const Sample& last = sorted_by_scan.back();
        const Sample& median = sorted_by_scan[sorted_by_scan.size() / 2];
        std::cout << "\t" << name << ":\n"
                  << "\t\tSample count: " << samples.size() << "\n"
                  << "\t\tFirst  -> scan " << first.scan_index << ", active_voxels=" << first.active_voxels
                  << ", stored_points=" << first.stored_points << "\n"
                  << "\t\tMedian -> scan " << median.scan_index << ", active_voxels=" << median.active_voxels
                  << ", stored_points=" << median.stored_points << "\n"
                  << "\t\tLast   -> scan " << last.scan_index << ", active_voxels=" << last.active_voxels
                  << ", stored_points=" << last.stored_points << "\n";
      }
    }
    ~GaugeData() { print_results(); }
  };
  static inline GaugeData data_{};
  static GaugeData& data() { return data_; }
};

/**
 * [instrumentation, additive-only] Histogram of ICP iteration counts per scan,
 * plus the mean per-iteration correspondence count, mirroring MapGrowthGauge's
 * style: call sites push one sample per completed icp() call (see
 * IcpResult::iterations_used / avg_correspondences_per_iteration in lio.cpp),
 * and a summary prints to stdout at program exit alongside the other
 * profiling output. Purely observational -- nothing here is read back into
 * any pose/map computation.
 */
class IcpIterationHistogram {
public:
  /// Record one scan's ICP iteration count and its mean correspondences/iteration.
  static void record(std::size_t iterations, double avg_correspondences_per_iteration) {
    auto& d = data();
    ++d.scan_count;
    d.iteration_sum += iterations;
    d.max_iterations = std::max(d.max_iterations, iterations);
    d.correspondences_sum += avg_correspondences_per_iteration;
    ++d.histogram[bucket_for(iterations)];
  }

  /// Print aggregated histogram results to stdout.
  static void print_results() { data().print_results(); }

private:
  // Buckets: 1, 2, 3, 4, 5, 6-10, 11-20, 21-50, 51-100, 100+
  static std::string bucket_for(std::size_t iterations) {
    if (iterations <= 5) {
      return std::to_string(iterations);
    }
    if (iterations <= 10) {
      return "6-10";
    }
    if (iterations <= 20) {
      return "11-20";
    }
    if (iterations <= 50) {
      return "21-50";
    }
    if (iterations <= 100) {
      return "51-100";
    }
    return "100+";
  }

  static const std::vector<std::string>& bucket_order() {
    static const std::vector<std::string> order = {"1",  "2",     "3",     "4",      "5",
                                                    "6-10", "11-20", "21-50", "51-100", "100+"};
    return order;
  }

  class Data {
  public:
    std::size_t scan_count;
    std::size_t iteration_sum;
    std::size_t max_iterations;
    double correspondences_sum;
    std::unordered_map<std::string, std::size_t> histogram;
    // Not using in-class default member initializers here: nested classes'
    // default member initializers are only usable once the *enclosing*
    // class's definition is complete, which conflicts with instantiating
    // `data_` as an IcpIterationHistogram member below (GCC error: "default
    // member initializer ... required before the end of its enclosing
    // class"). An explicit constructor sidesteps that rule entirely.
    Data() : scan_count(0), iteration_sum(0), max_iterations(0), correspondences_sum(0.0) {}
    Data(const Data&) = delete;
    Data(Data&&) = delete;
    Data& operator=(const Data&) = delete;
    Data& operator=(Data&&) = delete;
    void print_results() const {
      if (scan_count == 0) {
        return;
      }
      std::cout << "ICP Iteration Histogram\n"
                << "\tScans: " << scan_count << "\n"
                << "\tMean iterations/scan: " << std::fixed << std::setprecision(2)
                << (static_cast<double>(iteration_sum) / static_cast<double>(scan_count)) << "\n"
                << "\tMax iterations (any scan): " << max_iterations << "\n"
                << "\tMean correspondences/iteration (avg over scans): " << std::setprecision(1)
                << (correspondences_sum / static_cast<double>(scan_count)) << "\n"
                << "\tHistogram (iterations-to-converge -> scan count):\n";
      for (const auto& bucket : bucket_order()) {
        const auto it = histogram.find(bucket);
        if (it == histogram.end()) {
          continue;
        }
        std::cout << "\t\t" << bucket << ": " << it->second << "\n";
      }
    }
    ~Data() { print_results(); }
  };
  static inline Data data_{};
  static Data& data() { return data_; }
};

} // namespace rko_lio::core

#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)
/**
 * @def SCOPED_PROFILER(name)
 * Macro helper to create a ScopedProfiler instance with an automatic unique name.
 *
 * Use this macro at the start of a scope or function to measure execution time and collect
 * profiling data aggregating count, average, and max times for the named scope.
 *
 * @note This macro and the associated ScopedProfiler class are planned for deprecation.
 */
#define SCOPED_PROFILER(name) rko_lio::core::ScopedProfiler CONCAT(profiler_, __LINE__)(name)
