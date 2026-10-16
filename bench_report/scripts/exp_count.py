"""Experiment (local only): count the work of the 3x3x3 nearest-neighbour search.

Per query: existing voxels, voxels scanned (not pruned by the cell bound), points scanned, and how
many of the scanned voxels an AABB bound of their points would have skipped at that moment.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / "rko_lio/core/voxel_hash_map.cpp"
text = path.read_text()
old = """  for (std::size_t shift = 0; shift < shifts.size(); ++shift) {
    const Voxel& voxel_shift = shifts[shift];
    const double bound_sq = axis_bounds[0][voxel_shift.x() + 1] +
                            axis_bounds[1][voxel_shift.y() + 1] +
                            axis_bounds[2][voxel_shift.z() + 1];
    if (bound_sq >= closest_squared_distance) {
      continue;
    }
    const auto [points, count] = points_of(shift);
    for (std::size_t i = 0; i < count; ++i) {"""
new = """  long n_scanned = 0, n_points = 0, n_aabb_skip = 0, n_aabb_skip_points = 0, n_pruned = 0;
  for (std::size_t shift = 0; shift < shifts.size(); ++shift) {
    const Voxel& voxel_shift = shifts[shift];
    const double bound_sq = axis_bounds[0][voxel_shift.x() + 1] +
                            axis_bounds[1][voxel_shift.y() + 1] +
                            axis_bounds[2][voxel_shift.z() + 1];
    if (bound_sq >= closest_squared_distance) {
      ++n_pruned;
      continue;
    }
    const auto [points, count] = points_of(shift);
    if (count > 0) {
      ++n_scanned;
      n_points += static_cast<long>(count);
      Eigen::Vector3d lo = points[0], hi = points[0];
      for (std::size_t i = 1; i < count; ++i) {
        lo = lo.cwiseMin(points[i]);
        hi = hi.cwiseMax(points[i]);
      }
      const Eigen::Vector3d clamped = query.cwiseMax(lo).cwiseMin(hi);
      if ((clamped - query).squaredNorm() >= closest_squared_distance) {
        ++n_aabb_skip;
        n_aabb_skip_points += static_cast<long>(count);
      }
    }
    for (std::size_t i = 0; i < count; ++i) {"""
assert text.count(old) == 1, "search anchor"
text = text.replace(old, new)
old_end = """  const double closest_distance = closest_squared_distance == std::numeric_limits<double>::max()
                                      ? std::numeric_limits<double>::max()
                                      : std::sqrt(closest_squared_distance);
  return std::make_tuple(closest_neighbor, closest_distance);
}

} // namespace"""
assert text.count(old_end) == 1, "end anchor"
text = text.replace(old_end, """  search_counters.queries += 1;
  search_counters.scanned += n_scanned;
  search_counters.points += n_points;
  search_counters.aabb_skip += n_aabb_skip;
  search_counters.aabb_skip_points += n_aabb_skip_points;
  search_counters.pruned += n_pruned;
  const double closest_distance = closest_squared_distance == std::numeric_limits<double>::max()
                                      ? std::numeric_limits<double>::max()
                                      : std::sqrt(closest_squared_distance);
  return std::make_tuple(closest_neighbor, closest_distance);
}

} // namespace""")
decl = """
struct SearchCounters {
  std::atomic<long> queries{0}, scanned{0}, points{0}, aabb_skip{0}, aabb_skip_points{0}, pruned{0};
  ~SearchCounters() {
    const double q = static_cast<double>(queries.load());
    std::cout << "search counters: queries=" << queries << " scanned_voxels/q=" << scanned / q
              << " points/q=" << points / q << " pruned_by_cell/q=" << pruned / q
              << " aabb_skippable_voxels/q=" << aabb_skip / q << " aabb_skippable_points/q=" << aabb_skip_points / q
              << "\\n";
  }
};
inline SearchCounters search_counters;
"""
anchor = "namespace {\nusing rko_lio::core::Voxel;\n"
assert text.count(anchor) == 1, "namespace anchor"
text = text.replace(anchor, "#include <atomic>\n#include <iostream>\n" + anchor + decl)
path.write_text(text)
print("count experiment applied", root)
