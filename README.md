# CO2 Emissions – production-style ML project

Turns the Colab case study into a reusable package: **model zoo → CV comparison → tuning → versioned registry → monitoring → auto retrain**.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . shap fastapi uvicorn anthropic   # add xgboost lightgbm for optional boosters
curl -L -o data/CO2_Emissions.csv "https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/158/208/original/CO2_Emissions.csv?1760012266"
```

## Use
| Command | What it does |
|---|---|
| `co2-ml train` | Trains 13 regressors (Linear, Ridge, Lasso, ElasticNet, KNN, Tree, RandomForest, ExtraTrees, GradientBoosting, HistGB, SVR, XGBoost, LightGBM) for two scenarios (`spec_only`, `spec_plus_consumption`), 5-fold CV + hold-out test, tunes the top-3 (RandomizedSearchCV), saves the champion to `models/<scenario>/vNNN/` with metrics. |
| `co2-ml predict --input new.csv --output out.csv` | Batch predictions from the champion. |
| `co2-ml monitor --new-data fresh.csv --reference-data data/CO2_Emissions.csv` | Scores the champion on new *labelled* data; flags degradation (RMSE/MAE up >10%, R² down >0.02) and feature drift (PSI > 0.25). |
| `co2-ml retrain --new-data fresh.csv --reference-data ...` | Same check; if degraded, re-tunes candidates on reference + recent data, and **promotes only if it beats the champion by ≥1% RMSE on a hold-out**. Otherwise the old champion stays. |
| `co2-ml history` | Version history of the champion. |

Add `--scenario spec_only` for the spec-only model. All thresholds, models, features and paths live in `configs/config.yaml`.

## RAG assistant + API (see LEARN.md for the full walkthrough)
| Command | What it does |
|---|---|
| `co2-ml index` | Build the vector index from `knowledge/`, model cards, leaderboards and monitoring reports |
| `co2-ml ask "why did we retrain?"` | Cited answer from the knowledge base (set `ANTHROPIC_API_KEY` to use Claude as the writer) |
| `co2-ml serve` | FastAPI on :8000 - `/predict?explain=true`, `/ask`, `/reindex`, `/health`, docs at `/docs` |
| `docker build -t co2-ml . && docker run -p 8000:8000 -v $PWD/models:/app/models co2-ml` | Containerised service |

## Schedule monitoring (cron / Airflow / GitHub Actions)
```
0 2 * * 1  cd /path/co2_ml && co2-ml retrain --new-data /data/last_week.csv --reference-data data/CO2_Emissions.csv
```

## Notes
* Fuel consumption is a near-direct predictor of CO2 (notebook R² 0.86 → 0.99), so both scenarios are kept: use `spec_only` when consumption is unknown at prediction time.
* Tests: `pip install pytest && pytest` (uses synthetic data, includes a simulated drift → retrain → promote check).
* Registry is file-based; swap `registry.py` for MLflow/S3 in a larger deployment.
