"""User-item collaborative filtering using cosine similarity."""
import numpy as np
from .similarity import cosine_similarity


def recommend(user_item_matrix, user_index, top_k=5):
    matrix = np.asarray(user_item_matrix, dtype=float)
    similarities = cosine_similarity(matrix)[user_index]
    scores = similarities @ matrix
    seen = matrix[user_index] > 0
    scores = scores.copy()
    scores[seen] = -np.inf
    indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in indices if np.isfinite(scores[i])]
