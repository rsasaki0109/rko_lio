"""Experiment (local only): split build_icp_linear_system into an association pass and the
unchanged deterministic reduce, and time both passes. Apply after instrument.py.

Usage: python3 exp_twopass.py <rko_lio source dir> [spatial]
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
spatial = len(sys.argv) > 2 and sys.argv[2] == "spatial"
path = root / "rko_lio/core/lio.cpp"
text = path.read_text()

old = """  using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
  const auto& [H_icp, b_icp, chi_icp, weight_sum, correspondences] = tbb::parallel_deterministic_reduce(
      // Range
      tbb::blocked_range<points_iterator>{frame.cbegin(), frame.cend()},
      // Identity
      WeightedSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero(), 0.0, 0.0, 0),
      // 1st Lambda: Parallel computation
      [&](const tbb::blocked_range<points_iterator>& r, WeightedSystem J) -> WeightedSystem {
        return std::transform_reduce(r.begin(), r.end(), J, linear_system_reduce, [&](const auto& point) {
          // Compute data association and linear system
          const Eigen::Vector3d transformed_point = current_pose * point;
          // neighborhoods holds one cache per point of frame, indexed like frame.
          const auto& [closest_neighbor, distance] =
              neighborhoods != nullptr
                  ? voxel_map.get_closest_neighbor(transformed_point, (*neighborhoods)[&point - frame.data()])
                  : voxel_map.get_closest_neighbor(transformed_point, voxel_search_radius);
          if (distance < max_correspondence_distance) {
            return linear_system_for_one_point(transformed_point, closest_neighbor);
          }
          // TODO (meher): additional 0 add flops, which may hurt single threaded perf slightly
          return WeightedSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero(), 0.0, 0.0, 0);
        });
      },
      // 2nd Lambda: Parallel reduction of the private Jacobians
      linear_system_reduce);
"""
assert text.count(old) == 1, "reduce anchor"
order_code = ""
loop_index = "k"
if spatial:
    order_code = """
  // Visit the keypoints in voxel order so that consecutive queries share map voxels.
  static thread_local std::vector<std::size_t> order;
  static thread_local const Eigen::Vector3d* order_for = nullptr;
  static thread_local std::size_t order_size = 0;
  if (order_for != frame.data() || order_size != frame.size()) {
    order.resize(frame.size());
    std::vector<std::uint64_t> keys(frame.size());
    for (std::size_t i = 0; i < frame.size(); ++i) {
      const Eigen::Vector3d p = current_pose * frame[i];
      const Eigen::Vector3i v = (p * voxel_map.inv_voxel_size_ * 0.5).array().floor().cast<int>();
      std::uint64_t key = 0;
      for (int bit = 0; bit < 21; ++bit) {
        for (int axis = 0; axis < 3; ++axis) {
          key |= ((static_cast<std::uint64_t>(static_cast<std::uint32_t>(v[axis] + (1 << 20))) >> bit) & 1ULL)
                 << (3 * bit + axis);
        }
      }
      keys[i] = key;
      order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });
    order_for = frame.data();
    order_size = frame.size();
  }
  // order is thread_local: the workers below must read the registration thread's copy.
  const std::vector<std::size_t>& visit_order = order;
"""
    loop_index = "visit_order[k]"
new = """  struct Association {
    Eigen::Vector3d source;
    Eigen::Vector3d target;
    bool valid;
  };
  std::vector<Association> associations(frame.size());
""" + order_code + """  {
    SCOPED_PROFILER("prof: icp association pass");
    tbb::parallel_for(tbb::blocked_range<std::size_t>{0, frame.size(), 32}, [&](const tbb::blocked_range<std::size_t>& r) {
      for (std::size_t k = r.begin(); k != r.end(); ++k) {
        const std::size_t i = """ + loop_index + """;
        const Eigen::Vector3d transformed_point = current_pose * frame[i];
        const auto [closest_neighbor, distance] =
            neighborhoods != nullptr ? voxel_map.get_closest_neighbor(transformed_point, (*neighborhoods)[i])
                                     : voxel_map.get_closest_neighbor(transformed_point, voxel_search_radius);
        associations[i] = {transformed_point, closest_neighbor, distance < max_correspondence_distance};
      }
    });
  }
  SCOPED_PROFILER("prof: icp reduce pass");
  using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
  const auto& [H_icp, b_icp, chi_icp, weight_sum, correspondences] = tbb::parallel_deterministic_reduce(
      // Range
      tbb::blocked_range<points_iterator>{frame.cbegin(), frame.cend()},
      // Identity
      WeightedSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero(), 0.0, 0.0, 0),
      // 1st Lambda: Parallel computation
      [&](const tbb::blocked_range<points_iterator>& r, WeightedSystem J) -> WeightedSystem {
        return std::transform_reduce(r.begin(), r.end(), J, linear_system_reduce, [&](const auto& point) {
          const Association& a = associations[&point - frame.data()];
          if (a.valid) {
            return linear_system_for_one_point(a.source, a.target);
          }
          return WeightedSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero(), 0.0, 0.0, 0);
        });
      },
      // 2nd Lambda: Parallel reduction of the private Jacobians
      linear_system_reduce);
"""
text = text.replace(old, new)
if "#include <tbb/parallel_for.h>" not in text:
    raise SystemExit("parallel_for include missing")
path.write_text(text)
print("two-pass experiment applied", root, "spatial" if spatial else "index order")
