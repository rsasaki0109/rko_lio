"""Experiment (local only): prefetch only the voxels the previous cached search actually read.
Apply on top of 734d54f. Usage: python3 exp_prefetch_read.py <dir>"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
hpp = root / "rko_lio/core/voxel_hash_map.hpp"
t = hpp.read_text()
old = "    std::uint32_t looked_up = 0; // bit i: shift i has been looked up\n"
assert t.count(old) == 1
hpp.write_text(t.replace(old, old + "    std::uint32_t read = 0;      // bit i: the last search read shift i\n"))
cpp = root / "rko_lio/core/voxel_hash_map.cpp"
t = cpp.read_text()
old1 = """    cache.voxel = voxel;
    cache.looked_up = 0;
  } else {"""
new1 = """    cache.voxel = voxel;
    cache.looked_up = 0;
    cache.read = 0;
  } else {"""
old2 = """    for (std::uint32_t pending = cache.looked_up; pending != 0; pending &= pending - 1) {
      const auto shift = static_cast<std::size_t>(std::countr_zero(pending));
      prefetch_points(cache.points[shift], cache.sizes[shift]);
    }
  }
  return closest_neighbor_around(query, voxel, voxel_size_, [&](const std::size_t shift) {
    const std::uint32_t bit = std::uint32_t{1} << shift;"""
new2 = """    for (std::uint32_t pending = cache.read; pending != 0; pending &= pending - 1) {
      const auto shift = static_cast<std::size_t>(std::countr_zero(pending));
      prefetch_points(cache.points[shift], cache.sizes[shift]);
    }
  }
  cache.read = 0;
  return closest_neighbor_around(query, voxel, voxel_size_, [&](const std::size_t shift) {
    const std::uint32_t bit = std::uint32_t{1} << shift;
    cache.read |= bit;"""
for o, n in ((old1, new1), (old2, new2)):
    assert t.count(o) == 1, o[:60]
    t = t.replace(o, n)
cpp.write_text(t)
print("prefetch-read experiment applied", root)
