import sys, subprocess, yaml, pandas as pd
from pathlib import Path
sys.path.insert(0, "scripts")
from make_synthetic_data import make
from co2_ml.config import load_config
from co2_ml.train import run_training
from co2_ml.monitor import check, monitor_and_retrain
from co2_ml.registry import Registry

def test_end_to_end(tmp_path):
    cfg = load_config("configs/config.yaml")
    cfg["registry_dir"] = str(tmp_path / "models")
    cfg["models"] = ["linear", "ridge", "random_forest", "hist_gradient_boosting"]
    cfg["tuning"].update(top_k=1, n_iter=3, cv_folds=2); cfg["cv_folds"] = 2
    base = make(1500); base.to_csv(tmp_path / "base.csv", index=False)
    run_training(cfg, str(tmp_path / "base.csv"))
    assert Registry(cfg["registry_dir"]).champion_version("spec_plus_consumption") == "v001"
    ok = check(cfg, "spec_plus_consumption", make(800, seed=3), base)
    assert not ok["degraded"]
    drifted = make(1600, seed=7, drift=True)
    res = monitor_and_retrain(cfg, "spec_plus_consumption", drifted, base)
    assert res["report"]["degraded"] and res["retrain"]["promoted"]
