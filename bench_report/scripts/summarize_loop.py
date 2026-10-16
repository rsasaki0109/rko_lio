"""Summarize loop-closure pipeline ABAB CSVs (+ per-run CPU samples) and drain checks."""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def cpu_summary(path):
    rows = list(csv.DictReader(open(path)))
    series = defaultdict(dict)
    for r in rows:
        series[r["proc"]][int(r["t"])] = int(r["cpu_ticks"])
    out = {}
    for proc, s in series.items():
        ts = sorted(s)
        if len(ts) < 2:
            continue
        out[proc] = ((s[ts[-1]] - s[ts[0]]) / 100.0, ts[-1] - ts[0])
    return out


for path in sys.argv[1:]:
    p = Path(path)
    rows = list(csv.reader(open(p)))
    if not rows:
        continue
    if p.name.startswith("drain"):
        print(f"### {p.name}")
        print("| run | frontend s | frontend end (t) | backend idle (t) | backend left after frontend s | loop candidates at frontend end | loop candidates total | md5 |")
        print("|---|---:|---:|---:|---:|---:|---:|---|")
        for r in rows:
            print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7][:8]} |")
        print()
        continue
    labels = []
    for r in rows:
        if r[0] not in labels:
            labels.append(r[0])
    print(f"### {p.name}")
    print("| run | label | pair | wall s | RTF | offline node s | loop candidates | load1 | frontend CPU s (cores) | backend CPU s (cores) | md5 |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(rows, 1):
        cpu_path = p.parent / f"{p.stem}_{r[0]}_{r[1]}_cpu.csv"
        cpu = cpu_summary(cpu_path) if cpu_path.exists() else {}
        f = cpu.get("offline_node", (float("nan"), 1))
        g = cpu.get("graph_based_slam_node", (float("nan"), 1))
        print(f"| {i} | {r[0]} | {r[1]} | {float(r[3]):.2f} | {float(r[4]):.4f} | {r[11]} | {r[12]} | {r[5]} | "
              f"{f[0]:.1f} ({f[0] / max(1, f[1]):.2f}) | {g[0]:.1f} ({g[0] / max(1, g[1]):.2f}) | {r[9][:8]} |")
    med = {}
    for lab in labels:
        w = [float(r[3]) for r in rows if r[0] == lab]
        med[lab] = statistics.median(w)
        print(f"- {lab}: median wall {med[lab]:.2f} s, median RTF {med[lab] / 125.814:.4f}")
    if len(labels) == 2:
        a, b = labels
        print(f"- ratio {b}/{a} = {med[b] / med[a]:.4f}")
    print()
