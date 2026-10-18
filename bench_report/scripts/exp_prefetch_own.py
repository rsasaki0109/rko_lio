"""Experiment (local only): at the start of a cached query, prefetch up to <max_lines> lines of every
cached voxel block, sized by its point count. Usage: python3 exp_prefetch_own.py <dir> <max_lines>"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
max_lines = int(sys.argv[2])
path = root / "rko_lio/core/voxel_hash_map.cpp"
text = path.read_text()
old = """  if (voxel != cache.voxel) {
    cache.voxel = voxel;
    cache.looked_up = 0;
  }
  return closest_neighbor_around(query, voxel, voxel_size_, [&](const std::size_t shift) {"""
new = f"""  if (voxel != cache.voxel) {{
    cache.voxel = voxel;
    cache.looked_up = 0;
  }} else {{
    for (std::uint32_t pending = cache.looked_up; pending != 0; pending &= pending - 1) {{
      const int shift = __builtin_ctz(pending);
      const char* bytes = reinterpret_cast<const char*>(cache.points[shift]);
      if (bytes == nullptr) continue;
      const std::uint32_t lines = std::min<std::uint32_t>({max_lines}, (cache.sizes[shift] * 24 + 63) / 64);
      for (std::uint32_t line = 0; line < lines; ++line) __builtin_prefetch(bytes + 64 * line);
    }}
  }}
  return closest_neighbor_around(query, voxel, voxel_size_, [&](const std::size_t shift) {{"""
assert text.count(old) == 1
path.write_text(text.replace(old, new))
print("own-prefetch experiment applied", root, max_lines)
