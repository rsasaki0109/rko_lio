"""Summarize ABAB CSVs: per-run table, medians per label, and ratio B/A of medians.

Usage: python3 summarize_abab.py <csv> [<csv> ...]
CSV columns: label,pair,rc,node_s,wall_s,load1,load5,load15,start_utc,md5,lines,icp_ms,reg_ms
"""
import csv
import statistics
import sys

for path in sys.argv[1:]:
    rows = list(csv.reader(open(path)))
    labels = []
    for r in rows:
        if r[0] not in labels:
            labels.append(r[0])
    print(f"## {path}")
    print("| run | label | pair | node s | wall s | load1 at start | md5 | ICP ms/scan | reg loop ms/scan |")
    print("|---:|---|---:|---:|---:|---:|---|---:|---:|")
    for i, r in enumerate(rows, 1):
        print(f"| {i} | {r[0]} | {r[1]} | {float(r[3]):.2f} | {float(r[4]):.2f} | {r[5]} | {r[9][:8]} ({r[10]}) | {r[11]} | {r[12]} |")
    med = {}
    for lab in labels:
        xs = [float(r[3]) for r in rows if r[0] == lab]
        loads = [float(r[5]) for r in rows if r[0] == lab]
        med[lab] = statistics.median(xs)
        md5s = sorted({r[9] for r in rows if r[0] == lab})
        print(f"- {lab}: n={len(xs)} median={med[lab]:.2f} s min={min(xs):.2f} max={max(xs):.2f} "
              f"median load1={statistics.median(loads):.2f} md5={','.join(m[:8] for m in md5s)}")
    if len(labels) == 2:
        a, b = labels
        print(f"- ratio {b}/{a} = {med[b] / med[a]:.4f} (speedup {med[a] / med[b]:.3f}x)")
        pa = {r[1]: float(r[3]) for r in rows if r[0] == a}
        pb = {r[1]: float(r[3]) for r in rows if r[0] == b}
        pr = [pb[k] / pa[k] for k in pa if k in pb]
        print(f"- per-pair ratios: {', '.join(f'{x:.4f}' for x in pr)}; median {statistics.median(pr):.4f}")
    print()
