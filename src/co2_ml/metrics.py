import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(y_true, y_pred, n_features=None):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    out = {"mae": float(mean_absolute_error(y_true, y_pred)),
           "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
           "r2": float(r2), "n": int(n)}
    if n_features and n > n_features + 1:
        out["adj_r2"] = float(1 - (1 - r2) * (n - 1) / (n - n_features - 1))
    return out
