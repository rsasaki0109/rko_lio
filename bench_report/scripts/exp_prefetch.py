"""Experiment (local only): prefetch the cached voxel blocks at the start of every cached query.

Usage: python3 exp_prefetch.py <rko_lio source dir> <lines>
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
lines = int(sys.argv[2])
path = root / "rko_lio/core/voxel_hash_map.cpp"
text = path.read_text()
old = """  if (voxel != cache.voxel) {
    cache.voxel = voxel;
    cache.looked_up = 0;
  }
  return closest_neighbor_around(query, voxel, voxel_size_, [&](const std::size_t shift) {"""
prefetch = "".join(
    f"        __builtin_prefetch(reinterpret_cast<const char*>(cache.points[shift]) + {64 * k});\n" for k in range(lines))
new = """  if (voxel != cache.voxel) {
    cache.voxel = voxel;
    cache.looked_up = 0;
  } else {
    for (std::uint32_t pending = cache.looked_up; pending != 0; pending &= pending - 1) {
      const int shift = __builtin_ctz(pending);
      if (cache.points[shift] != nullptr) {
""" + prefetch + """      }
    }
  }
  return closest_neighbor_around(query, voxel, voxel_size_, [&](const std::size_t shift) {"""
assert text.count(old) == 1
path.write_text(text.replace(old, new))
print("prefetch experiment applied", root, lines)
