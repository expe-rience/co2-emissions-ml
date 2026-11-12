"""Step 5: generation. Two interchangeable generators. Both ONLY see the retrieved context (that is what 'grounded' means)."""
import os, re

SYSTEM = ("You are an assistant for a vehicle CO2 emissions ML system. Answer ONLY from the numbered context. "
          "Cite sources like [1]. If the context does not contain the answer, say you don't know. "
          "Never invent numbers; never claim to have made the prediction yourself - the regression model did. Be concise.")

def _sentences(t): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) > 2]
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# question filler words that would otherwise match unrelated sentences ("does", "mean"...)
_STOP = set(ENGLISH_STOP_WORDS) | {"does", "did", "do", "mean", "means", "meaning", "stand", "stands", "tell", "explain"}
def _tok(t): return set(re.findall(r"[a-z0-9]+", t.lower())) - _STOP

class ExtractiveGenerator:
    """No LLM: picks the context sentences that best overlap the question. Cheap, offline, can't hallucinate."""
    name = "extractive"
    def answer(self, question, passages):
        q = _tok(question); scored = []
        if passages and passages[0]["source"] == "live-model-output":      # short, already the answer: return it whole
            return passages[0]["text"] + " [1]"
        for n, p in enumerate(passages, 1):
            for s in _sentences(p["text"]):
                scored.append((len(q & _tok(s)) / (1 + 0.02 * len(s.split())), n, s))
        # trust the retriever: take the 2 best sentences from the top-ranked passage, then 1 more from the rest
        top = sorted([x for x in scored if x[1] == 1], key=lambda x: -x[0])[:2]
        rest = sorted([x for x in scored if x[1] != 1], key=lambda x: -x[0])[:1]
        best = [x for x in top + rest if x[0] > 0]
        return " ".join(f"{s} [{n}]" for _, n, s in best) if best else "I don't know."

class ClaudeGenerator:
    """LLM: writes fluent answers from the same context. Needs ANTHROPIC_API_KEY."""
    name = "claude"
    def __init__(self, model):
        import anthropic; self.c = anthropic.Anthropic(); self.model = model
    def answer(self, question, passages):
        ctx = "\n\n".join(f"[{i}] ({p['source']}) {p['text']}" for i, p in enumerate(passages, 1))
        r = self.c.messages.create(model=self.model, max_tokens=500, system=SYSTEM,
                                   messages=[{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}])
        return r.content[0].text

def make_generator(cfg):
    mode = cfg["rag"]["generator"]
    if mode == "claude" or (mode == "auto" and os.getenv("ANTHROPIC_API_KEY")):
        try: return ClaudeGenerator(cfg["rag"]["llm_model"])
        except Exception: pass
    return ExtractiveGenerator()
