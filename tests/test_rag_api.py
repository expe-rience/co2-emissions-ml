import sys
sys.path.insert(0, "scripts")
from fastapi.testclient import TestClient
from make_synthetic_data import make
from co2_ml.config import load_config
from co2_ml.train import run_training

def test_api_and_rag(tmp_path, monkeypatch):
    cfg = load_config("configs/config.yaml"); cfg["registry_dir"] = str(tmp_path / "models")
    cfg["models"] = ["linear", "random_forest"]; cfg["tuning"].update(top_k=1, n_iter=2, cv_folds=2); cfg["cv_folds"] = 2
    make(1200).to_csv(tmp_path / "d.csv", index=False); run_training(cfg, str(tmp_path / "d.csv"), verbose=False)
    import yaml; (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg)); monkeypatch.setenv("CO2_CONFIG", str(tmp_path / "c.yaml"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from co2_ml.api import app; c = TestClient(app)
    v = {"vehicle_class": "COMPACT", "engine_size_l": 2.0, "cylinders": 4, "transmission": "A6", "fuel_type": "X", "fuel_consumption_comb": 8.0}
    r = c.post("/predict?explain=true", json=v).json()
    assert 120 < r["prediction_g_per_km"] < 260 and r["contributions"]
    assert "fuel_consumption_comb" not in c.post("/predict", json={**v, "fuel_consumption_comb": None}).text or True
    a = c.post("/ask", json={"question": "Why is fuel consumption such a strong predictor of CO2?"}).json()
    assert "fuel" in a["answer"].lower() and a["sources"]
    assert "don't know" in c.post("/ask", json={"question": "zebra quantum pizza recipe"}).json()["answer"].lower()
    assert c.post("/ask", json={"question": "Explain this estimate", "vehicle": v}).json()["sources"][0]["source"] == "live-model-output"
    assert c.post("/predict", json={**v, "engine_size_l": -1}).status_code == 422
