"""md5 table (branch x bag) from results_md5.csv plus every ABAB/pipeline run's trajectory md5."""
import csv
import glob
from collections import defaultdict

EXP04 = "bc23199243971466e1645e4eed7640ca"
MID = "2a1c810771773f2ed719735cdf9c11fd"
rows = list(csv.reader(open("/home/user/bench/runs/md5/results_md5.csv")))
table = defaultdict(dict)
for r in rows:
    label, md5, n = r[0], r[8], r[9]
    bag = "exp04" if label.endswith("_exp04") or "_exp04_" in label else "mid360"
    key = label.rsplit("_", 1)[0] if label.endswith(("_exp04", "_mid360")) else label
    key = key.replace("_exp04", "").replace("_mid360", "")
    table[key][bag] = (md5, n)
print("| build | exp04 md5 (poses) | MID-360 md5 (poses) |")
print("|---|---|---|")
for key, bags in table.items():
    cells = []
    for bag, ref in (("exp04", EXP04), ("mid360", MID)):
        if bag in bags:
            md5, n = bags[bag]
            cells.append(f"`{md5}` ({n}) {'一致' if md5 == ref else '**不一致**'}")
        else:
            cells.append("-")
    print(f"| {key} | {cells[0]} | {cells[1]} |")
# every timed run also produced a trajectory
md5s = defaultdict(set)
count = 0
for path in glob.glob("/home/user/bench/abab/*.csv") + glob.glob("/home/user/bench/prof/*.csv"):
    for r in csv.reader(open(path)):
        md5s[r[0]].add(r[9] if len(r) > 9 and len(r[9]) == 32 else r[8])
        count += 1
print()
print(f"timed frontend runs checked: {count}; distinct md5 over them: "
      f"{sorted({m for s in md5s.values() for m in s})}")
