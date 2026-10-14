# recommendation-engine

Content-based and user-based collaborative filtering recommenders in Python/NumPy,
with an offline evaluation against a popularity baseline.

## Components

| File | Role |
|---|---|
| `src/similarity.py` | Vectorized cosine similarity matrix (zero vectors handled) |
| `src/content_based.py` | Recommends items whose feature vectors are most similar to a given item |
| `src/collaborative.py` | User-based CF: score = sum of (user similarity x other users' ratings); already-rated items are excluded; configurable `top_k` |
| `src/data.py` | Loads a ratings JSON file into a user-item matrix |
| `src/evaluate.py` | Leave-one-out hit rate@k, compared with a popularity baseline |

## Run

```bash
pip install -r requirements.txt
python src/main.py
python -m pytest -q
```

## Evaluation

For each user one rating is hidden; the recommender returns its top-k unseen items and
scores a hit if the hidden item is among them.

On the bundled **synthetic** dataset (60 users, 45 items, 631 ratings, three taste groups,
regenerate with `python scripts/generate_sample_data.py`), hit rate@5 is about 0.62 for
collaborative filtering versus 0.08 for the popularity baseline. The data is built so that
users in a group share tastes, so this shows the pipeline works, not real-world performance.

## Limitations

- Synthetic toy data only; no real dataset benchmark yet.
- No rating normalization or similarity-weighted averaging; no cold-start handling.
- Matrix factorization is not implemented.
- Evaluation covers collaborative filtering only.
