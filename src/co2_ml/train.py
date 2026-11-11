import numpy as np, pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, KFold
from .config import features_for
from .data import load_data, split, validate
from .models import ZOO, available
from .pipeline import build_pipeline
from .metrics import evaluate
from .registry import Registry

def scenarios(cfg):
    return {n: bool(v) for n, v in cfg["scenarios"].items()}

def leaderboard(cfg, train, test, with_cons, verbose=True):
    cat, num = features_for(cfg, with_cons); feats = cat + num; t = cfg["data"]["target"]
    Xtr, ytr, Xte, yte = train[feats], train[t], test[feats], test[t]
    cv = KFold(cfg["cv_folds"], shuffle=True, random_state=42)
    rows = []
    for name in available(cfg["models"]):
        pipe = build_pipeline(name, cat, num)
        cvr = -cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
        pipe.fit(Xtr, ytr)
        m = evaluate(yte, pipe.predict(Xte), n_features=pipe[:-1].transform(Xte.head(1)).shape[1])
        rows.append({"model": name, "cv_rmse": cvr.mean(), "cv_std": cvr.std(), **{f"test_{k}": v for k, v in m.items() if k != 'n'}})
        if verbose: print(f"  {name:24s} cv_rmse={cvr.mean():7.3f}  test_rmse={m['rmse']:7.3f}  r2={m['r2']:.4f}")
    return pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)

def tune(cfg, name, train, with_cons):
    cat, num = features_for(cfg, with_cons); feats = cat + num; t = cfg["data"]["target"]
    space = ZOO[name][2]
    pipe = build_pipeline(name, cat, num)
    if not space:
        return pipe.fit(train[feats], train[t]), {}
    grid = {f"regressor__{k}": v for k, v in space.items()}
    tc = cfg["tuning"]
    rs = RandomizedSearchCV(pipe, grid, n_iter=tc["n_iter"], cv=tc["cv_folds"],
                            scoring="neg_root_mean_squared_error", random_state=42, n_jobs=1, refit=True)
    rs.fit(train[feats], train[t])
    return rs.best_estimator_, {k.replace("regressor__", ""): v for k, v in rs.best_params_.items()}

def run_training(cfg, data_path=None, verbose=True):
    df = load_data(cfg, data_path); validate(cfg, df)
    train, test = split(cfg, df)
    reg = Registry(cfg["registry_dir"]); results = {}
    for scen, with_cons in scenarios(cfg).items():
        if verbose: print(f"\n=== Scenario: {scen} (consumption={with_cons}) | train={len(train)} test={len(test)} ===")
        lb = leaderboard(cfg, train, test, with_cons, verbose)
        cat, num = features_for(cfg, with_cons); feats = cat + num; t = cfg["data"]["target"]
        best_name, best_model, best_params = lb.loc[0, "model"], None, {}
        best_metrics = None
        candidates = []
        if cfg["tuning"]["enabled"]:
            for name in lb["model"].head(cfg["tuning"]["top_k"]):
                if verbose: print(f"  tuning {name} ...")
                mdl, params = tune(cfg, name, train, with_cons)
                m = evaluate(test[t], mdl.predict(test[feats]))
                candidates.append((m["rmse"], name, mdl, params, m))
        base = lb.loc[0]
        m0 = {"mae": base.test_mae, "rmse": base.test_rmse, "r2": base.test_r2}
        if not candidates or min(c[0] for c in candidates) >= base.test_rmse:
            mdl = build_pipeline(best_name, cat, num).fit(train[feats], train[t])
            best_model, best_params, best_metrics = mdl, {}, evaluate(test[t], mdl.predict(test[feats]))
        else:
            _, best_name, best_model, best_params, best_metrics = min(candidates, key=lambda c: c[0])
        if verbose: print(f"  >> champion: {best_name}  test={ {k: round(v,4) for k,v in best_metrics.items()} }")
        meta = {"scenario": scen, "with_consumption": with_cons, "model": best_name, "params": {k: (v.item() if hasattr(v,'item') else v) for k,v in best_params.items()},
                "metrics": best_metrics, "features": feats, "n_train": len(train), "trigger": "train"}
        v = reg.save(scen, best_model, meta, background=train[feats].sample(min(100, len(train)), random_state=42))
        lb.to_csv(f"{cfg['registry_dir']}/{scen}/leaderboard_{v}.csv", index=False)
        results[scen] = {"version": v, "model": best_name, "metrics": best_metrics, "leaderboard": lb}
    return results
