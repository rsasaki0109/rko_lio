"""Turn the profiler block of one or more offline_node stdout logs into a table (mean ms, count, total s)."""
import re
import sys

def parse(path):
    text = open(path).read()
    out = {}
    for m in re.finditer(r"\t([^\n]+?):\n\t\tExecution count: (\d+)\n\t\tAverage time: ([0-9.]+) ms", text):
        name, count, mean = m.group(1), int(m.group(2)), float(m.group(3))
        out[name] = (count, mean)
    counters = re.search(r"prof counters: (.*)", text)
    return out, counters.group(1) if counters else ""

paths = sys.argv[1:]
data = [parse(p) for p in paths]
names = sorted({n for d, _ in data for n in d}, key=lambda n: -max(d.get(n, (0, 0))[0] * d.get(n, (0, 0))[1] for d, _ in data))
print("| stage | " + " | ".join(f"{p.split('/')[-2]} mean ms (count)" for p in paths) + " |")
print("|---|" + "---:|" * len(paths))
for n in names:
    cells = []
    for d, _ in data:
        if n in d:
            c, m = d[n]
            cells.append(f"{m:.2f} ({c}) = {c * m / 1000:.1f} s")
        else:
            cells.append("-")
    print(f"| {n} | " + " | ".join(cells) + " |")
for p, (_, c) in zip(paths, data):
    print(f"- {p.split('/')[-2]}: {c}")
