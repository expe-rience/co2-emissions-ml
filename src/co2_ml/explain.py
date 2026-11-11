"""Local explanations: *why* did the model predict this number?  Uses SHAP, with an ablation fallback."""
import numpy as np, pandas as pd

def _group(name, features):
    n = name.split("__", 1)[-1]
    for f in sorted(features, key=len, reverse=True):
        if n == f or n.startswith(f + "_"): return f
    return n

def explain_one(model, meta, row: pd.DataFrame, background: pd.DataFrame, top_n=6):
    feats = meta["features"]; pre, reg = model[:-1], model[-1]
    pred = float(model.predict(row[feats])[0])
    try:
        import shap
        names = list(pre.get_feature_names_out())
        Xb, Xr = pre.transform(background[feats]), pre.transform(row[feats])
        sv = shap.Explainer(reg, Xb, feature_names=names)(Xr)
        base = float(np.ravel(sv.base_values)[0]); agg = {}
        for n, v in zip(names, sv.values[0]):
            g = _group(n, feats); agg[g] = agg.get(g, 0.0) + float(v)
        method = "shap"
    except Exception:                       # ablation: swap each feature for a typical value, see how prediction moves
        base = float(model.predict(background[feats]).mean()); agg = {}
        for f in feats:
            alt = row[feats].copy()
            alt[f] = background[f].mode().iloc[0] if background[f].dtype == object else background[f].median()
            agg[f] = pred - float(model.predict(alt)[0])
        method = "ablation"
    contrib = sorted(({"feature": f, "value": row[f].iloc[0].item() if hasattr(row[f].iloc[0], "item") else row[f].iloc[0],
                       "impact_g_per_km": round(v, 2)} for f, v in agg.items()), key=lambda d: -abs(d["impact_g_per_km"]))
    return {"prediction_g_per_km": round(pred, 1), "baseline_g_per_km": round(base, 1), "method": method,
            "contributions": contrib[:top_n]}

def to_text(exp, meta):
    lines = [f"The model ({meta['model']} {meta['version']}) predicts {exp['prediction_g_per_km']} g/km CO2.",
             f"The average vehicle in the training sample is about {exp['baseline_g_per_km']} g/km."]
    for c in exp["contributions"]:
        d = "raises" if c["impact_g_per_km"] > 0 else "lowers"
        lines.append(f"{c['feature']} = {c['value']} {d} the estimate by {abs(c['impact_g_per_km'])} g/km.")
    return " ".join(lines)
