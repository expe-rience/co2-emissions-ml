from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .models import ZOO

def build_pipeline(name, cat, num, estimator=None):
    factory, scale, _ = ZOO[name]
    num_t = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("numerical", num_t, num)])
    return Pipeline([("preprocessor", pre), ("regressor", estimator or factory())])
