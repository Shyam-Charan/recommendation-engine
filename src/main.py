"""Runnable recommendation engine demonstration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.content_based import recommend as content_recommend
from src.collaborative import recommend as collaborative_recommend


def main():
    item_features = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=float)
    print("Content-based:", content_recommend(item_features, 0, 3))

    user_item = np.array([[5, 4, 0, 0], [5, 0, 3, 1], [0, 4, 5, 2]], dtype=float)
    print("Collaborative:", collaborative_recommend(user_item, 0, 3))


if __name__ == "__main__":
    main()
