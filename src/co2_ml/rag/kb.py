"""Step 1 of RAG: turn every source of knowledge into small, labelled text chunks."""
import json, re
from pathlib import Path

def chunk_text(text, words=90, overlap=20):
    """Split on headings, then slide a window of ~`words` words so each chunk fits one idea."""
    out = []
    for sec in re.split(r"\n(?=#)", text):
        title = sec.strip().splitlines()[0].lstrip("# ").strip() if sec.strip() else ""
        lines = sec.strip().splitlines()
        toks = ("\n".join(lines[1:]) if lines and lines[0].startswith("#") else sec).split()   # heading is stored as title, not in the text
        step = max(1, words - overlap)
        for i in range(0, max(len(toks), 1), step):
            piece = " ".join(toks[i:i + words])
            if len(piece.split()) >= 8: out.append((title, piece))
            if i + words >= len(toks): break
    return out

def _registry_docs(cfg):
    root = Path(cfg["registry_dir"]); docs = []
    for scen_dir in sorted(p for p in root.glob("*") if p.is_dir() and p.name != "reports"):
        champ = (scen_dir / "champion.txt")
        champ_v = champ.read_text().strip() if champ.exists() else None
        for meta_f in sorted(scen_dir.glob("v*/metadata.json")):
            m = json.loads(meta_f.read_text()); mt = m["metrics"]
            status = "CURRENT CHAMPION" if m["version"] == champ_v else "older version"
            txt = (f"Model card for scenario {m['scenario']} version {m['version']} ({status}). Algorithm: {m['model']}. "
                   f"Created {m['created']}. Why it was created (training or retrain trigger): {m['trigger']}. Test metrics: RMSE {mt['rmse']:.2f} g/km, MAE {mt['mae']:.2f} g/km, "
                   f"R-squared {mt['r2']:.4f}. Features used: {', '.join(m['features'])}. Trained on {m['n_train']} rows. "
                   f"Tuned hyperparameters: {m['params'] or 'defaults'}.")
            docs.append((f"registry/{m['scenario']}/{m['version']}", "Model card", txt))
        for lb in sorted(scen_dir.glob("leaderboard_*.csv")):
            import pandas as pd
            d = pd.read_csv(lb).round(3)
            rows = "; ".join(f"{r.model}: test RMSE {r.test_rmse}, R-squared {r.test_r2}" for r in d.itertuples())
            docs.append((f"registry/{scen_dir.name}/{lb.stem}", "Leaderboard",
                         f"Algorithm leaderboard for scenario {scen_dir.name}, sorted best first. {rows}."))
    for rep in sorted((root / "reports").glob("*.json")) if (root / "reports").exists() else []:
        r = json.loads(rep.read_text()); lv, bs = r["live"], r["baseline"]
        verdict = "DEGRADED: " + " | ".join(r["reasons"]) if r["degraded"] else "healthy, no degradation detected"
        docs.append((f"reports/{rep.name}", "Monitoring report",
                     f"Monitoring report for scenario {r['scenario']} on champion {r['champion']} ({r['model']}). Status: {verdict}. "
                     f"Live RMSE {lv['rmse']:.2f} vs baseline {bs['rmse']:.2f}; live R-squared {lv['r2']:.4f} vs baseline {bs['r2']:.4f}. "
                     f"Drift PSI by feature: {json.dumps({k: round(v, 3) for k, v in r['drift_psi'].items()})}."))
    return docs

def build_chunks(cfg):
    rc = cfg["rag"]; chunks = []
    for f in sorted(Path(rc["knowledge_dir"]).glob("*.md")):
        for title, piece in chunk_text(f.read_text(), rc["chunk_words"], rc["chunk_overlap"]):
            chunks.append({"source": f.name, "title": title, "text": piece})
    for src, title, text in _registry_docs(cfg):
        chunks.append({"source": src, "title": title, "text": text})
    return chunks
