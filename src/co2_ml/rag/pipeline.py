from pathlib import Path
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

    def ask(self, question, live_context=None):
        hits = [h for h in self.index.search(question, self.rc["top_k"]) if h["score"] >= self.rc["min_score"]]
        passages = ([{"source": "live-model-output", "title": "Live prediction", "text": live_context, "score": 1.0}] if live_context else []) + hits
        if not passages:
            return {"answer": "I don't know - nothing in the knowledge base is relevant to that.", "sources": [], "generator": self.gen.name}
        return {"answer": self.gen.answer(question, passages), "generator": self.gen.name,
                "sources": [{"n": i, "source": p["source"], "title": p["title"], "score": round(p["score"], 3)} for i, p in enumerate(passages, 1)]}
