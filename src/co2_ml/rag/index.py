"""Step 3: the vector index - store chunk vectors, find the nearest ones to a question (cosine similarity)."""
import joblib, numpy as np
from pathlib import Path
from scipy import sparse
from sklearn.preprocessing import normalize

class VectorIndex:
    def __init__(self, embedder, chunks=None, matrix=None):
        self.embedder, self.chunks, self.matrix = embedder, chunks or [], matrix

    def build(self, chunks):
        self.chunks = chunks
        self.matrix = normalize(self.embedder.fit_transform([f"{c['title']}. {c['text']}" for c in chunks]))  # title helps retrieval
        return self

    def search(self, query, k=4):
        q = normalize(self.embedder.transform([query]))
        scores = q @ self.matrix.T
        scores = scores.toarray().ravel() if sparse.issparse(scores) else np.asarray(scores).ravel()
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True); joblib.dump(self, path)

    @staticmethod
    def load(path): return joblib.load(path)
