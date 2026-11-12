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


# ---------- PDF support: PDF -> text -> structured chunks ----------
_BULLET = re.compile(r"^(?:[\u25cf\u2022\u25aa*\-]|\d{1,2}[.)])\s+")

def pdf_to_text(path):
    """Read a PDF with pypdf in 'layout' mode (keeps headings and bullets on their own lines).
    All pages are joined BEFORE chunking so a list that runs across a page break stays together."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("PDF knowledge files need pypdf:  pip install pypdf") from e
    pages = [p.extract_text(extraction_mode="layout") or "" for p in PdfReader(str(path)).pages]
    return "\n\n".join(pages).replace("\u200b", "")

def _is_header(line, nxt):
    """A short line with no number, colon or equals sign, followed by more text = a section header."""
    w = line.split()
    return bool(nxt) and 0 < len(w) <= 4 and not _BULLET.match(line) and not re.search(r"[:=.]", line)

def _end(t):
    t = t.strip()
    return t if t[-1:] in ".!?" else t + "."

def chunk_structured(text, words=90):
    """Chunk list-like documents (data dictionaries, glossaries). Rules:
    - a header line starts a new section; every bullet is one unit and is never split
    - bullet units are prefixed with their header ('Fuel type: Z = Premium gasoline.') so each one stands alone
    - units are packed into chunks of about `words` words, inside one section."""
    lines = [l.rstrip() for l in text.splitlines()]
    sections, header, units = [], "Document", []
    def flush():
        nonlocal units
        if units: sections.append((header, units)); units = []
    i = 0
    while i < len(lines):
        raw = lines[i].strip(); i += 1
        if not raw: continue
        nxt = next((l.strip() for l in lines[i:] if l.strip()), "")
        m = _BULLET.match(raw)
        # a lowercase or deeply indented line right after an item is that item's wrapped text, never a new header
        if units and not m and (not raw[0].isupper() or lines[i-1].startswith(" " * 8)):
            units[-1][0] += " " + raw
            continue
        if _is_header(raw, nxt):
            flush(); header = raw; continue
        if m:
            body = raw[m.end():]
            numbered = raw[:m.end()].strip()[0].isdigit()
            units.append([body, not numbered])          # [text, needs header prefix]
        elif units and (not raw[0].isupper() or lines[i-1].startswith(" " * 8)):   # wrapped line of the previous item
            units[-1][0] += " " + raw
        else:
            units.append([raw, False])                   # plain paragraph line
    flush()
    out = []
    for head, us in sections:
        cur, n = [], 0
        for body, prefix in us:
            u = _end(f"{head}: {body}" if prefix else body)
            if cur and n + len(u.split()) > words:
                out.append((head, " ".join(cur))); cur, n = [], 0
            cur.append(u); n += len(u.split())
        if cur: out.append((head, " ".join(cur)))
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
    kdir = Path(rc["knowledge_dir"])
    for f in sorted(kdir.glob("*.md")):
        for title, piece in chunk_text(f.read_text(), rc["chunk_words"], rc["chunk_overlap"]):
            chunks.append({"source": f.name, "title": title, "text": piece})
    for f in sorted(kdir.glob("*.pdf")):                      # drop a PDF in knowledge/ and re-index
        for title, piece in chunk_structured(pdf_to_text(f), rc["chunk_words"]):
            chunks.append({"source": f.name, "title": title, "text": piece})
    for src, title, text in _registry_docs(cfg):
        chunks.append({"source": src, "title": title, "text": text})
    return chunks
