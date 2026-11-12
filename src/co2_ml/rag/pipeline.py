import re
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from .kb import build_chunks
from .embed import make_embedder
from .index import VectorIndex
from .generate import make_generator

class RAGPipeline:
    """question -> retrieve top chunks -> (guard: relevant enough?) -> generate cited answer."""
    def __init__(self, cfg):
        self.cfg, self.rc = cfg, cfg["rag"]
        self.path = Path(cfg["registry_dir"]) / "rag_index.joblib"
        self.gen = make_generator(cfg)
        self.index = VectorIndex.load(self.path) if self.path.exists() else self.reindex()

    def reindex(self):
        """Rebuild after new docs, a new model version, or a new monitoring report."""
        self.index = VectorIndex(make_embedder(self.cfg)).build(build_chunks(self.cfg))
        self.index.save(self.path); return self.index

    def unknown_codes(self, question):
        """Short code-like words (<=3 letters, e.g. a fuel code 'Zz') that appear nowhere in the knowledge base.
        TF-IDF silently drops unknown words, so without this check the assistant would answer with the *nearest* facts."""
        vec = getattr(self.index.embedder, "v", None)       # only meaningful for the word-based TF-IDF embedder
        if vec is None: return []
        return [t for t in re.findall(r"[a-z0-9]+", question.lower())
                if len(t) <= 3 and t not in vec.vocabulary_ and t not in ENGLISH_STOP_WORDS]

    def ask(self, question, live_context=None):
        hits = [h for h in self.index.search(question, self.rc["top_k"]) if h["score"] >= self.rc["min_score"]]
        passages = ([{"source": "live-model-output", "title": "Live prediction", "text": live_context, "score": 1.0}] if live_context else []) + hits
        if not passages:
            return {"answer": "I don't know - nothing in the knowledge base is relevant to that.", "sources": [], "generator": self.gen.name}
        unknown = self.unknown_codes(question)
        note = (f"Note: I found nothing about {', '.join(repr(u) for u in unknown)} in the knowledge base, "
                f"so the facts below may not answer your question. ") if unknown else ""
        return {"answer": note + self.gen.answer(question, passages), "generator": self.gen.name, "unknown_terms": unknown,
                "sources": [{"n": i, "source": p["source"], "title": p["title"], "score": round(p["score"], 3)} for i, p in enumerate(passages, 1)]}
