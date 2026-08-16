# build_zh_side.py
import os

SRC = "data/dataset/tgredial/pkuseg/hownet.txt"
DST = "data/conceptnet/zh_side.txt"

os.makedirs(os.path.dirname(DST), exist_ok=True)

pairs = set()
with open(SRC, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        w1 = parts[1].split("/")[0]
        w2 = parts[2].split("/")[0]
        if w1 and w2 and w1 != w2:
            pairs.add((w1, w2))

with open(DST, "w", encoding="utf-8") as f:
    for a, b in pairs:
        f.write(f"{a} {b}\n")

print(f"Wrote {len(pairs)} pairs to {DST}")
