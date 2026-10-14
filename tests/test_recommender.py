import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.similarity import cosine_similarity
from src.collaborative import recommend as collab
from src.content_based import recommend as content
from src.data import load_ratings
from src.evaluate import leave_one_out, evaluate

DATA = Path(__file__).resolve().parents[1] / "data" / "sample_data.json"


def test_cosine_similarity_identical_rows_is_one():
    sim = cosine_similarity([[1, 2, 3], [2, 4, 6], [0, 0, 0]])
    assert sim[0, 1] == pytest.approx(1.0)
    assert sim[0, 2] == 0.0  # zero vector handled without NaN


def test_collaborative_never_recommends_seen_items():
    m = np.array([[5, 4, 0, 0], [5, 0, 3, 1], [0, 4, 5, 2]], dtype=float)
    recs = collab(m, 0, 4)
    assert all(m[0, i] == 0 for i, _ in recs)


def test_content_based_excludes_query_item():
    feats = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=float)
    assert content(feats, 0, 1)[0][0] != 0


def test_loader_shapes():
    users, items, matrix = load_ratings(DATA)
    assert matrix.shape == (len(users), len(items))
    assert (matrix > 0).sum() > 0


def test_leave_one_out_hides_exactly_one_rating_per_user():
    _, _, matrix = load_ratings(DATA)
    train, hidden = leave_one_out(matrix)
    for u, item in hidden.items():
        assert matrix[u, item] > 0 and train[u, item] == 0
        assert (matrix[u] > 0).sum() - (train[u] > 0).sum() == 1


def test_collaborative_beats_popularity_on_grouped_data():
    _, _, matrix = load_ratings(DATA)
    scores = evaluate(matrix, top_k=5)
    assert scores["collaborative"] > scores["popularity_baseline"]
