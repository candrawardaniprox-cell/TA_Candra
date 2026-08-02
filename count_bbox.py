"""Hitung jumlah bbox per kelas di Dataset2026."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

path = Path("data/Dataset2026/_annotations.coco.json")
coco = json.load(open(path, "r", encoding="utf-8"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
counts = Counter(a["category_id"] for a in coco["annotations"])
img_count = len(coco["images"])

print(f"=== Dataset2026 ({img_count} images) ===")
print(f"Categories: {cats}")
for k in sorted(counts):
    print(f"  {cats[k]}: {counts[k]} bbox")
total = sum(counts.values())
print(f"  TOTAL: {total} bbox")
