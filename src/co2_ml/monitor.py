"""Performance + drift monitoring, and automatic tune/retrain/promote."""
import numpy as np, pandas as pd
from pathlib import Path
from .config import features_for
from .data import validate
from .metrics import evaluate
from .registry import Registry
from .train import tune, scenarios

def psi(expected, actual, bins=10):
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3: return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))

def cat_psi(expected, actual):
    cats = set(expected) | set(actual)
    e = expected.value_counts(normalize=True); a = actual.value_counts(normalize=True)
    s = 0.0
    for c in cats:
        pe, pa = max(e.get(c, 0), 1e-4), max(a.get(c, 0), 1e-4)
        s += (pa - pe) * np.log(pa / pe)
    return float(s)

def check(cfg, scenario, new_df, reference_df=None):
    """Score champion on labeled new data; return report with degradation flags."""
    reg = Registry(cfg["registry_dir"]); model, meta = reg.load_champion(scenario)
    t = cfg["data"]["target"]; feats = meta["features"]; base = meta["metrics"]; mc = cfg["monitoring"]
    live = evaluate(new_df[t], model.predict(new_df[feats]))
    reasons = []
    if live["rmse"] > base["rmse"] * (1 + mc["max_rmse_increase_pct"] / 100):
        reasons.append(f"RMSE {live['rmse']:.2f} vs baseline {base['rmse']:.2f} (+{(live['rmse']/base['rmse']-1)*100:.1f}%)")
    if live["mae"] > base["mae"] * (1 + mc["max_mae_increase_pct"] / 100):
        reasons.append(f"MAE {live['mae']:.2f} vs baseline {base['mae']:.2f}")
    if live["r2"] < base["r2"] - mc["min_r2_drop"]:
        reasons.append(f"R2 {live['r2']:.4f} vs baseline {base['r2']:.4f}")
    drift = {}
    if reference_df is not None:
        cat, num = features_for(cfg, meta["with_consumption"])
        for c in num: drift[c] = psi(reference_df[c], new_df[c])
        for c in cat: drift[c] = cat_psi(reference_df[c], new_df[c])
        bad = {k: round(v, 3) for k, v in drift.items() if v > mc["psi_threshold"]}
        if bad: reasons.append(f"Feature drift (PSI>{mc['psi_threshold']}): {bad}")
    return {"scenario": scenario, "champion": meta["version"], "model": meta["model"], "baseline": base,
            "live": live, "drift_psi": drift, "degraded": bool(reasons), "reasons": reasons}

def retrain(cfg, scenario, train_df, holdout_df, reason="metric degradation"):
    """Re-tune top models on fresh data; promote only if it beats the champion on the holdout."""
    reg = Registry(cfg["registry_dir"]); champ, meta = reg.load_champion(scenario)
    t = cfg["data"]["target"]; feats = meta["features"]; with_cons = meta["with_consumption"]
    champ_m = evaluate(holdout_df[t], champ.predict(holdout_df[feats]))
    cands = []
    for name in dict.fromkeys([meta["model"], "hist_gradient_boosting", "random_forest", "ridge"]):
        mdl, params = tune(cfg, name, train_df, with_cons)
        m = evaluate(holdout_df[t], mdl.predict(holdout_df[feats]))
        cands.append((m["rmse"], name, mdl, params, m))
    rm, name, mdl, params, m = min(cands, key=lambda c: c[0])
    gain = (champ_m["rmse"] - m["rmse"]) / champ_m["rmse"] * 100
    promote = gain >= cfg["monitoring"]["min_improvement_pct"]
    new_meta = {"scenario": scenario, "with_consumption": with_cons, "model": name,
                "params": {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()},
                "metrics": m, "features": feats, "n_train": len(train_df), "trigger": f"retrain: {reason}",
                "previous_champion": meta["version"], "previous_champion_metrics_on_holdout": champ_m}
    v = reg.save(scenario, mdl, new_meta, promote=promote, background=train_df[feats].sample(min(100, len(train_df)), random_state=42))
    return {"scenario": scenario, "new_version": v, "model": name, "promoted": promote,
            "champion_rmse_on_holdout": champ_m["rmse"], "candidate_rmse": m["rmse"], "improvement_pct": gain}

def monitor_and_retrain(cfg, scenario, new_df, reference_df=None, auto_retrain=True):
    """Full loop: check -> (if degraded) retrain on reference+new data -> promote if better."""
    report = check(cfg, scenario, new_df, reference_df)
    out = {"report": report, "retrain": None}
    _save_report(cfg, report)
    if report["degraded"] and auto_retrain:
        new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)
        k = len(new_df) // 2
        recent_train, holdout = new_df.iloc[:k], new_df.iloc[k:]
        train = pd.concat([reference_df, recent_train]) if reference_df is not None else recent_train
        out["retrain"] = retrain(cfg, scenario, train, holdout, "; ".join(report["reasons"]))
    return out


def _save_report(cfg, report):
    """Persist every monitoring report - the RAG layer indexes these so you can ask 'why did we retrain?'."""
    import json, time
    d = Path(cfg["registry_dir"]) / "reports"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{time.strftime('%Y%m%d_%H%M%S')}_{report['scenario']}.json").write_text(json.dumps(report, indent=2, default=float))
