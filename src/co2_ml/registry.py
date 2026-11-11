"""File-based model registry with versioning and a champion pointer."""
import json, time, joblib
from pathlib import Path

class Registry:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def _scen(self, scenario):
        p = self.root / scenario; p.mkdir(exist_ok=True); return p

    def save(self, scenario, model, meta, promote=True, background=None):
        p = self._scen(scenario)
        version = f"v{len(list(p.glob('v*'))) + 1:03d}"
        vd = p / version; vd.mkdir()
        joblib.dump(model, vd / "model.joblib")
        if background is not None: background.to_csv(vd / "background.csv", index=False)  # sample of training rows, used by SHAP
        meta = {**meta, "version": version, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        (vd / "metadata.json").write_text(json.dumps(meta, indent=2))
        if promote: self.promote(scenario, version)
        return version

    def promote(self, scenario, version):
        (self._scen(scenario) / "champion.txt").write_text(version)

    def champion_version(self, scenario):
        f = self._scen(scenario) / "champion.txt"
        return f.read_text().strip() if f.exists() else None

    def load_champion(self, scenario):
        v = self.champion_version(scenario)
        if v is None: raise FileNotFoundError(f"No champion for '{scenario}'. Run `co2-ml train` first.")
        vd = self.root / scenario / v
        return joblib.load(vd / "model.joblib"), json.loads((vd / "metadata.json").read_text())

    def history(self, scenario):
        p = self._scen(scenario)
        return [json.loads((d / "metadata.json").read_text()) for d in sorted(p.glob("v*"))]

    def load_background(self, scenario):
        import pandas as pd
        f = self.root / scenario / self.champion_version(scenario) / "background.csv"
        return pd.read_csv(f) if f.exists() else None
