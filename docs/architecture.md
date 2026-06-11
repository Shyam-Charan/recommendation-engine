# Architecture

## Components

### Data Ingestion Layer
Loads and preprocesses user-item interaction data.

### Content-Based Filter
Recommends items based on item feature similarity.

### Collaborative Filter
Recommends items based on user behaviour patterns.

### Similarity Engine
Computes cosine similarity and matrix factorization.

## Data Flow

User Input → Filter Selection → Similarity Engine → Ranked Recommendations
