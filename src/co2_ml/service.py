"""Business logic shared by the API and CLI: predict + explain + ask, with the champion cached in memory."""
import os, pandas as pd
from .config import load_config
from .registry import Registry
from .explain import explain_one, to_text
from .rag import RAGPipeline

COLS = {"vehicle_class": "Vehicle Class", "engine_size_l": "Engine Size(L)", "cylinders": "Cylinders",
        "transmission": "Transmission", "fuel_type": "Fuel Type", "fuel_consumption_comb": "Fuel Consumption Comb (L/100 km)"}

class Service:
    def __init__(self, config_path=None):
        self.cfg = load_config(config_path or os.getenv("CO2_CONFIG", "configs/config.yaml"))
        self.reg = Registry(self.cfg["registry_dir"]); self._cache = {}; self._rag = None

    def _champion(self, scenario):
        v = self.reg.champion_version(scenario)          # cheap file read: picks up promotions without restart
        if self._cache.get(scenario, (None,))[0] != v:
            model, meta = self.reg.load_champion(scenario)
            self._cache[scenario] = (v, model, meta, self.reg.load_background(scenario))
        return self._cache[scenario][1:]

    @property
    def rag(self):
        if self._rag is None: self._rag = RAGPipeline(self.cfg)
        return self._rag

    def _row(self, vehicle: dict):
        scen = "spec_plus_consumption" if vehicle.get("fuel_consumption_comb") is not None else "spec_only"
        return scen, pd.DataFrame([{COLS[k]: v for k, v in vehicle.items() if k in COLS and v is not None}])

    def predict(self, vehicle: dict, explain=False):
        scen, row = self._row(vehicle); model, meta, bg = self._champion(scen)
        out = {"scenario": scen, "model": meta["model"], "version": meta["version"]}
        if explain and bg is not None:
            e = explain_one(model, meta, row, bg); out.update(e); out["explanation_text"] = to_text(e, meta)
        else:
            out["prediction_g_per_km"] = round(float(model.predict(row[meta["features"]])[0]), 1)
        return out

    def ask(self, question, vehicle: dict | None = None):
        live = None
        if vehicle:
            p = self.predict(vehicle, explain=True); live = p.get("explanation_text") or f"Predicted {p['prediction_g_per_km']} g/km."
        return self.rag.ask(question, live)
