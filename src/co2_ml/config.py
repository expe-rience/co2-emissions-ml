import yaml
from pathlib import Path

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def features_for(cfg, with_consumption: bool):
    d = cfg["data"]
    num = list(d["numeric_spec"]) + ([d["consumption_col"]] if with_consumption else [])
    return d["categorical"], num
