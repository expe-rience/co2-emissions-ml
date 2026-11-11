import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(cfg, path=None):
    d = cfg["data"]
    df = pd.read_csv(path or d["path"])
    if d.get("drop_duplicates", True):
        df = df.drop_duplicates().reset_index(drop=True)
    return df

def split(cfg, df):
    s = cfg["split"]
    return train_test_split(df, test_size=s["test_size"], random_state=s["random_state"])

def validate(cfg, df):
    d = cfg["data"]
    need = d["categorical"] + d["numeric_spec"] + [d["consumption_col"], d["target"]]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing columns: {missing}")
