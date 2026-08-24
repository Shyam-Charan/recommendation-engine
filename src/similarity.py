"""Similarity utilities for recommendation systems."""
import numpy as np


def cosine_similarity(matrix):
    x = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms) @ (x / norms).T
