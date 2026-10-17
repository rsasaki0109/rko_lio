"""Count the cached ICP neighbour searches of an rko_lio source tree (local measurement only, never committed).

Usage: python3 instrument_nn_reuse.py <rko_lio source dir>
Counts, over the whole run, the calls of VoxelHashMap::get_closest_neighbor(query, cache), how many
started in a new voxel, how many were answered from the cached neighbour, and how many ICP linear
systems were built with the cache (ICP iterations plus Levenberg-Marquardt evaluations). Prints them
to stderr at exit. Every replacement must match exactly once, otherwise the script fails.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def patch(rel, pairs):
    path = root / rel
    text = path.read_text()
    for old, new in pairs:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{rel}: anchor found {count} times: {old[:80]!r}")
        text = text.replace(old, new)
    path.write_text(text)


patch("rko_lio/core/voxel_hash_map.cpp", [
    ("""namespace rko_lio::core {

VoxelHashMap::VoxelHashMap(""",
     """#include <atomic>
#include <cstdio>
namespace rko_lio::core {
struct NnReuseCounters {
  std::atomic<unsigned long long> calls{0}, new_voxel{0}, skipped{0}, systems{0};
  ~NnReuseCounters() {
    std::fprintf(stderr, "NN_REUSE calls=%llu new_voxel=%llu skipped=%llu full_same_voxel=%llu systems=%llu\\n",
                 calls.load(), new_voxel.load(), skipped.load(), calls.load() - new_voxel.load() - skipped.load(),
                 systems.load());
  }
};
NnReuseCounters nn_reuse_counters;

VoxelHashMap::VoxelHashMap("""),
    ("""  const Voxel voxel = point_to_voxel(query, inv_voxel_size_);
  if (voxel != cache.voxel) {
    cache.voxel = voxel;
    cache.looked_up = 0;
    cache.has_reference = false;
  } else {""",
     """  const Voxel voxel = point_to_voxel(query, inv_voxel_size_);
  nn_reuse_counters.calls.fetch_add(1, std::memory_order_relaxed);
  if (voxel != cache.voxel) {
    nn_reuse_counters.new_voxel.fetch_add(1, std::memory_order_relaxed);
    cache.voxel = voxel;
    cache.looked_up = 0;
    cache.has_reference = false;
  } else {"""),
    ("""      if (distance + moved + reuse_margin(query) < cache.reference_others_distance) {
        return std::make_tuple(cache.reference_neighbor, distance);""",
     """      if (distance + moved + reuse_margin(query) < cache.reference_others_distance) {
        nn_reuse_counters.skipped.fetch_add(1, std::memory_order_relaxed);
        return std::make_tuple(cache.reference_neighbor, distance);"""),
])

patch("rko_lio/core/lio.cpp", [
    ("""#include "lio.hpp"
""",
     """#include "lio.hpp"
void rko_lio_count_cached_system();
"""),
    ("""  SystemSumBody body(add_point_system);
  tbb::parallel_deterministic_reduce(""",
     """  if (neighborhoods != nullptr) {
    rko_lio_count_cached_system();
  }
  SystemSumBody body(add_point_system);
  tbb::parallel_deterministic_reduce("""),
])

path = root / "rko_lio/core/voxel_hash_map.cpp"
path.write_text(path.read_text() + """
void rko_lio_count_cached_system() {
  rko_lio::core::nn_reuse_counters.systems.fetch_add(1, std::memory_order_relaxed);
}
""")
