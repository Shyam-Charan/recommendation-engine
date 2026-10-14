"""Load a ratings JSON file into a user-item matrix."""
import json

import numpy as np


def load_ratings(path):
    """Return (users, items, matrix) where matrix[u, i] is the rating (0 = not rated)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    users, items = data["users"], data["items"]
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {it: i for i, it in enumerate(items)}
    matrix = np.zeros((len(users), len(items)))
    for r in data["ratings"]:
        matrix[u_idx[r["user"]], i_idx[r["item"]]] = r["rating"]
    return users, items, matrix
