"""Content-based recommendations using cosine similarity."""
from .similarity import cosine_similarity


def recommend(item_features, item_index, top_k=5):
    scores = cosine_similarity(item_features)[item_index].copy()
    scores[item_index] = -1
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(int(i), float(scores[i])) for i in indices]
