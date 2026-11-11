"""Step 2: embeddings = turning text into vectors so 'similar meaning' becomes 'close in space'."""
import numpy as np

class TfidfEmbedder:
    """Offline, no downloads. Matches on shared words (lexical). Good baseline, fast, easy to explain."""
    name = "tfidf"
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", token_pattern=r"(?u)\b\w+\b")   # keep 1-letter tokens like fuel code "Z"
    def fit_transform(self, texts): return self.v.fit_transform(texts)
    def transform(self, texts): return self.v.transform(texts)

class SentenceTransformerEmbedder:
    """Semantic embeddings (matches on meaning, e.g. 'pollution' ~ 'emissions'). Needs `pip install sentence-transformers`."""
    name = "sentence-transformers"
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model_name)
    def fit_transform(self, texts): return self.transform(texts)
    def transform(self, texts): return np.asarray(self.m.encode(list(texts), normalize_embeddings=True))

def make_embedder(cfg):
    rc = cfg["rag"]
    if rc["embedder"] == "sentence-transformers":
        return SentenceTransformerEmbedder(rc["st_model"])
    return TfidfEmbedder()
