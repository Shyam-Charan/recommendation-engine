"""Offline evaluation by hiding one rating per user (leave-one-out).

For every user with at least two ratings, one rated item is hidden. The
recommender is asked for the top-k unseen items; it scores a *hit* if the
hidden item is among them. We report hit rate@k and compare against a
popularity baseline (recommend the most-rated items the user has not seen).
"""
import numpy as np

try:
    from .collaborative import recommend as collaborative_recommend
except ImportError:
    from collaborative import recommend as collaborative_recommend


def leave_one_out(matrix, seed=0):
    """Return (train_matrix, {user_index: hidden_item_index})."""
    rng = np.random.default_rng(seed)
    train = matrix.copy()
    hidden = {}
    for u in range(matrix.shape[0]):
        rated = np.flatnonzero(matrix[u] > 0)
        if len(rated) >= 2:
            item = int(rng.choice(rated))
            train[u, item] = 0
            hidden[u] = item
    return train, hidden


def popularity_recommend(train, user_index, top_k=5):
    counts = (train > 0).sum(axis=0).astype(float)
    counts[train[user_index] > 0] = -np.inf
    order = np.argsort(counts)[::-1][:top_k]
    return [(int(i), float(counts[i])) for i in order if np.isfinite(counts[i])]


def hit_rate_at_k(matrix, recommender, top_k=5, seed=0):
    train, hidden = leave_one_out(matrix, seed)
    hits = sum(1 for u, item in hidden.items()
               if item in [i for i, _ in recommender(train, u, top_k)])
    return hits / len(hidden) if hidden else 0.0


def evaluate(matrix, top_k=5, seed=0):
    return {
        "collaborative": hit_rate_at_k(matrix, collaborative_recommend, top_k, seed),
        "popularity_baseline": hit_rate_at_k(matrix, popularity_recommend, top_k, seed),
    }
