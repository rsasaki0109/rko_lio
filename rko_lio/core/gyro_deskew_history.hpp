#pragma once
#include "util.hpp"
#include <algorithm>
#include <deque>
#include <optional>

namespace rko_lio::core {

// Independent of ICP pose resets. Each IMU rate applies to the interval ending
// at its timestamp, matching add_imu_measurement's right-sample integration.
class GyroDeskewHistory {
  struct Knot {
    Nsec time;
    Sophus::SO3d rotation;
    Eigen::Vector3d rate;
  };
  std::deque<Knot> knots_;

  void trim() {
    while (knots_.size() > 4096 ||
           (knots_.size() > 2 && knots_.back().time - knots_[1].time > std::chrono::seconds(2))) {
      knots_.pop_front();
    }
  }

 public:
  void clear() { knots_.clear(); }
  std::size_t size() const { return knots_.size(); }

  void add(Nsec begin, Nsec end, const Eigen::Vector3d& rate) {
    if (end <= begin || end - begin > std::chrono::milliseconds(20) || !rate.allFinite()) {
      clear();
      return;
    }
    if (!knots_.empty() && knots_.back().time != begin) clear();
    if (knots_.empty()) knots_.push_back({begin, Sophus::SO3d(), rate});
    const auto rotation = knots_.back().rotation * Sophus::SO3d::exp(rate * to_seconds(end - begin));
    knots_.push_back({end, rotation, rate});
    trim();
  }

  // ROS normally leaves a sub-IMU-period tail before the scan end. Hold the
  // last rate for at most 20 ms; larger gaps invalidate this scan's history.
  bool finish_scan(Nsec end) {
    if (knots_.empty() || end < knots_.back().time) return false;
    const Nsec begin = knots_.back().time;
    if (end - begin > std::chrono::milliseconds(20)) {
      clear();
      return false;
    }
    if (end > begin) add(begin, end, knots_.back().rate);
    return true;
  }

  bool covers(Nsec begin, Nsec end) const {
    return !knots_.empty() && begin <= end && begin >= knots_.front().time && end <= knots_.back().time;
  }

  std::optional<Sophus::SO3d> at(Nsec time) const {
    if (!covers(time, time)) return std::nullopt;
    auto right = std::lower_bound(knots_.begin(), knots_.end(), time,
                                 [](const Knot& knot, Nsec t) { return knot.time < t; });
    if (right->time == time) return right->rotation;
    const auto& left = *std::prev(right);
    return left.rotation * Sophus::SO3d::exp(right->rate * to_seconds(time - left.time));
  }
};
}  // namespace rko_lio::core
