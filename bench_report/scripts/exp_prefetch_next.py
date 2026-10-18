"""Experiment (local only): prefetch the cached voxel blocks of the next keypoint from ICP's leaf.

Usage: python3 exp_prefetch_next.py <rko_lio source dir> <max_lines> <own:0|1>
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
max_lines = int(sys.argv[2])
own = sys.argv[3] == "1"
hpp = root / "rko_lio/core/voxel_hash_map.hpp"
t = hpp.read_text()
old = """  std::tuple<Eigen::Vector3d, double> get_closest_neighbor(const Eigen::Vector3d& query,
                                                           NeighborhoodCache& cache) const;
"""
assert t.count(old) == 1
t = t.replace(old, old + """  /** Prefetch hint for the points of the voxels `cache` already holds. */
  void prefetch_neighborhood(const NeighborhoodCache& cache) const;
""")
hpp.write_text(t)

cpp = root / "rko_lio/core/voxel_hash_map.cpp"
t = cpp.read_text()
old = """std::tuple<Eigen::Vector3d, double> VoxelHashMap::get_closest_neighbor(const Eigen::Vector3d& query,
                                                                        NeighborhoodCache& cache) const {
  const Voxel voxel = point_to_voxel(query, inv_voxel_size_);
  if (voxel != cache.voxel) {
    cache.voxel = voxel;
    cache.looked_up = 0;
  }
"""
assert t.count(old) == 1
own_code = "  } else {\n    prefetch_neighborhood(cache);\n  }\n" if own else "  }\n"
new = f"""void VoxelHashMap::prefetch_neighborhood(const NeighborhoodCache& cache) const {{
  for (std::uint32_t pending = cache.looked_up; pending != 0; pending &= pending - 1) {{
    const int shift = __builtin_ctz(pending);
    const char* bytes = reinterpret_cast<const char*>(cache.points[shift]);
    if (bytes == nullptr) continue;
    const std::uint32_t lines = std::min<std::uint32_t>({max_lines}, (cache.sizes[shift] * 24 + 63) / 64);
    for (std::uint32_t line = 0; line < lines; ++line) __builtin_prefetch(bytes + 64 * line);
  }}
}}

std::tuple<Eigen::Vector3d, double> VoxelHashMap::get_closest_neighbor(const Eigen::Vector3d& query,
                                                                        NeighborhoodCache& cache) const {{
  const Voxel voxel = point_to_voxel(query, inv_voxel_size_);
  if (voxel != cache.voxel) {{
    cache.voxel = voxel;
    cache.looked_up = 0;
{own_code}"""
t = t.replace(old, new)
cpp.write_text(t)

lio = root / "rko_lio/core/lio.cpp"
t = lio.read_text()
old = """          // neighborhoods holds one cache per point of frame, indexed like frame.
          const auto& [closest_neighbor, distance] ="""
assert t.count(old) == 1
new = """          // neighborhoods holds one cache per point of frame, indexed like frame.
          if (neighborhoods != nullptr) {
            const std::size_t next = static_cast<std::size_t>(&point - frame.data()) + 1;
            if (next < frame.size()) voxel_map.prefetch_neighborhood((*neighborhoods)[next]);
          }
          const auto& [closest_neighbor, distance] ="""
lio.write_text(t.replace(old, new))
print("prefetch-next experiment applied", root, max_lines, own)
