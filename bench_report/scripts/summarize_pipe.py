"""Summarize full-pipeline ABAB CSVs.

CSV columns: label,pair,rc,wall_s,rtf,load1,load5,load15,start_utc,front_md5,lines,node_took_s
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
    print("| run | label | pair | wall s | RTF | offline node s | load1 at start | front md5 |")
    print("|---:|---|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(rows, 1):
        print(f"| {i} | {r[0]} | {r[1]} | {float(r[3]):.2f} | {float(r[4]):.4f} | {r[11]} | {r[5]} | {r[9][:8]} ({r[10]}) |")
    med = {}
    for lab in labels:
        w = [float(r[3]) for r in rows if r[0] == lab]
        med[lab] = statistics.median(w)
        print(f"- {lab}: n={len(w)} median wall={med[lab]:.2f} s, median RTF={med[lab] / 125.814:.4f}, "
              f"min={min(w):.2f} max={max(w):.2f}")
    if len(labels) == 2:
        a, b = labels
        print(f"- ratio {b}/{a} = {med[b] / med[a]:.4f}")
    print()
