"""Generate a reproducible *synthetic* ratings file (data/sample_data.json).

Users belong to one of three taste groups; each group prefers a different third
of the items. This is toy data for demonstrating and testing the pipeline,
not a real-world benchmark.
"""
import json
import random
from pathlib import Path

random.seed(42)
N_USERS, N_ITEMS, N_GROUPS = 60, 45, 3
group_size = N_ITEMS // N_GROUPS

ratings = []
for u in range(N_USERS):
    g = u % N_GROUPS
    liked = range(g * group_size, (g + 1) * group_size)
    for i in range(N_ITEMS):
        in_group = i in liked
        if random.random() < (0.55 if in_group else 0.08):
            base = 4.5 if in_group else 2.0
            rating = max(1, min(5, round(random.gauss(base, 0.7))))
            ratings.append({"user": f"user_{u}", "item": f"item_{i}", "rating": rating})

data = {"users": [f"user_{u}" for u in range(N_USERS)],
        "items": [f"item_{i}" for i in range(N_ITEMS)],
        "ratings": ratings}
out = Path(__file__).resolve().parents[1] / "data" / "sample_data.json"
out.write_text(json.dumps(data))
print(f"wrote {len(ratings)} ratings to {out}")
