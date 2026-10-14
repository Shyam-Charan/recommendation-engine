"""Runnable recommendation engine demonstration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.content_based import recommend as content_recommend
from src.collaborative import recommend as collaborative_recommend
from src.data import load_ratings
from src.evaluate import evaluate

DATA = Path(__file__).resolve().parents[1] / "data" / "sample_data.json"


def main():
    # 1. Tiny hand-made examples
    item_features = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=float)
    print("Content-based (items similar to item 0):", content_recommend(item_features, 0, 3))

    # 2. Collaborative filtering on the sample ratings file
    users, items, matrix = load_ratings(DATA)
    print(f"\nLoaded {len(users)} users x {len(items)} items, {int((matrix > 0).sum())} ratings")
    recs = collaborative_recommend(matrix, 0, 5)
    print(f"Top-5 for {users[0]}:", [(items[i], round(s, 2)) for i, s in recs])

    # 3. Offline evaluation
    scores = evaluate(matrix, top_k=5)
    print("\nHit rate@5 (leave-one-out):")
    for name, value in scores.items():
        print(f"  {name:20s} {value:.2f}")


if __name__ == "__main__":
    main()
