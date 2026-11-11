# Learn-by-building guide: ML model + RAG, end to end

Read this alongside the code. Each section says **what** we built, **why**, and **how** to see it run.

## 0. The big picture
```
 CSV ──► train.py ──► REGISTRY (models/) ──► service.py ──► FastAPI (/predict /ask)
          13 models      versions, metrics,       │               │
          tune + CV      champion, SHAP bg        │               └─ Docker / cloud
                              ▲                   ▼
 monitor.py ── degraded? ─► retrain ─► promote   explain.py (SHAP: "why this number?")
                              │
                              └──► reports/ ──► rag/ (knowledge base + reports + model cards)
                                                   chunk → embed → index → retrieve → generate
```
**Golden rule:** the *regression model* makes the number. The *RAG layer* explains and answers questions around it. Never let an LLM produce the prediction.

## 1. The ML core (src/co2_ml/{data,models,pipeline,train}.py)
* **Pipeline** (`pipeline.py`): one object = preprocessing + model. *Why:* training and serving can never disagree on encoding (the #1 cause of silent production bugs).
* **Model zoo + CV** (`models.py`, `train.py`): try many algorithms, compare with cross-validation, tune the best three. *Why:* no algorithm wins everywhere; on near-linear data (fuel consumption → CO2) plain Ridge often wins, and that is a valid result.
* **Two scenarios**: with and without fuel consumption (the notebook's Model A vs B). *Why:* a feature you won't have at prediction time is a trap.
Try: `co2-ml train` and read the leaderboard.

## 2. The registry (registry.py)
Every trained model is saved as `models/<scenario>/vNNN/` with its metrics, params, features and a 100-row `background.csv`. `champion.txt` points to the live one. *Why:* reproducibility, audit, one-line rollback.

## 3. Monitoring and retraining (monitor.py)
* **Performance drift**: needs true labels (RMSE up >10% etc.). **Data drift** (PSI): needs no labels, an early warning.
* **Champion/challenger**: a retrained model is promoted only if it beats the champion on held-out data.
* Every monitoring report is saved to `models/reports/`, which feeds the RAG layer.
Try: `co2-ml retrain --new-data data/synthetic_drifted.csv --reference-data data/synthetic_train.csv`

## 4. Explainability (explain.py)
**SHAP** splits one prediction into per-feature contributions that add up to (prediction − baseline). One-hot columns are summed back to the original feature. *Why:* users trust "fuel consumption added +29 g/km" more than a bare number, and it gives the RAG layer *facts* to talk about instead of inventing reasons.

## 5. The RAG pipeline (src/co2_ml/rag/)
RAG = **R**etrieve relevant text, **A**ugment the prompt with it, **G**enerate an answer from it. Five steps, one file each:
| Step | File | What happens | Why it matters |
|---|---|---|---|
| 1 Chunk | `kb.py` | Docs, model cards, leaderboards, monitoring reports → ~90-word pieces | Small chunks retrieve precisely; too big dilutes the match |
| 2 Embed | `embed.py` | Text → vector. Default TF-IDF (word overlap, offline). Option: sentence-transformers (meaning) | Vectors let us compute "similarity" |
| 3 Index | `index.py` | Normalised vectors; cosine similarity search | Finds the top-k relevant chunks fast |
| 4 Guard | `pipeline.py` | If best score < `min_score` → "I don't know" | Prevents confident nonsense |
| 5 Generate | `generate.py` | Answer **only** from retrieved chunks, with [n] citations | Grounding = fewer hallucinations, verifiable answers |

**Live context:** `/ask` with a `vehicle` runs predict + SHAP and injects the result as source `[1] live-model-output`. That is how the assistant explains *this* estimate truthfully.

### Lessons visible in this repo (try them)
1. `co2-ml ask "What does fuel type code Z mean?"` — works because the answer is literally in the text. Delete `knowledge/vehicle_co2_basics.md`, re-index, and ask again: it should say it doesn't know. **RAG is only as good as its knowledge base.**
2. TF-IDF matches *words*, not *meaning*. Ask "what is pollution per kilometre?" (the docs say "CO2 g/km") and it can miss. Set `rag.embedder: sentence-transformers` and compare. That is the lexical vs semantic gap.
3. The offline *extractive* generator only copies sentences, so answers can be clunky. Set `ANTHROPIC_API_KEY` and it switches to Claude, which writes fluent cited answers from the same chunks (the system prompt forces "answer only from context"). Note: the Claude path needs your key and was not run in the build environment.
4. Re-index after every retrain (`co2-ml index` or `POST /reindex`) so "which model is champion?" stays true. Stale index = confidently wrong answers.

## 6. Serving (service.py, api.py, Dockerfile)
* `service.py` caches the champion and checks `champion.txt` each request, so a promotion goes live **without a restart**.
* `api.py`: FastAPI + pydantic validation (bad input → 422, not a wrong prediction). Docs at `/docs`.
* `Dockerfile`: container = same environment everywhere. Mount `models/` as a volume (or bake it in during CI).
Run: `co2-ml serve` → http://localhost:8000/docs

## 7. Production architecture (what you'd add around this)
| Need | Typical choice |
|---|---|
| Scheduled retrain/monitor | Airflow, Step Functions, GitHub Actions cron |
| Registry & experiments | MLflow / SageMaker / Vertex (replace `registry.py`) |
| Vector store at scale | pgvector, OpenSearch, Pinecone, Qdrant (replace `index.py`) |
| Serving | ECS/Cloud Run/K8s behind a load balancer; batch scoring if latency isn't needed |
| Observability | Log every prediction + later true value; Prometheus/Grafana alerts |
| LLM guardrails | Keep the "answer from context only" prompt, log questions, evaluate with a small Q&A test set |
| Cost (FinOps view) | Retrain on spot/serverless; metric-triggered retraining; cache RAG answers; batch over always-on endpoints |

## 8. Exercises
1. Add a new knowledge file about emissions regulations; re-index; ask about it.
2. Swap to `sentence-transformers` and compare retrieval scores.
3. Add a Prometheus `/metrics` endpoint counting predictions and `/ask` calls.
4. Build a 10-question evaluation set for the RAG assistant and measure how many answers cite the right source.
