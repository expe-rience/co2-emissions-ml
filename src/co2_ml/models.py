"""Model zoo: name -> (estimator factory, needs_scaling, tuning search space)."""
from scipy.stats import randint, uniform, loguniform
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.svm import SVR

SEED = 42
ZOO = {
 "linear": (lambda: LinearRegression(), True, {}),
 "ridge": (lambda: Ridge(random_state=SEED), True, {"alpha": loguniform(1e-2, 1e2)}),
 "lasso": (lambda: Lasso(random_state=SEED, max_iter=20000), True, {"alpha": loguniform(1e-3, 1)}),
 "elasticnet": (lambda: ElasticNet(random_state=SEED, max_iter=20000), True,
                {"alpha": loguniform(1e-3, 1), "l1_ratio": uniform(0.1, 0.8)}),
 "knn": (lambda: KNeighborsRegressor(), True,
         {"n_neighbors": randint(3, 25), "weights": ["uniform", "distance"]}),
 "decision_tree": (lambda: DecisionTreeRegressor(random_state=SEED), False,
                   {"max_depth": randint(3, 20), "min_samples_leaf": randint(1, 20)}),
 "random_forest": (lambda: RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1), False,
                   {"max_depth": [None, 10, 20], "min_samples_leaf": randint(1, 8),
                    "max_features": [1.0, 0.7, 0.5]}),
 "extra_trees": (lambda: ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=-1), False,
                 {"max_depth": [None, 10, 20], "min_samples_leaf": randint(1, 8)}),
 "gradient_boosting": (lambda: GradientBoostingRegressor(random_state=SEED), False,
                       {"n_estimators": randint(100, 400), "learning_rate": loguniform(0.02, 0.3),
                        "max_depth": randint(2, 6), "subsample": uniform(0.6, 0.4)}),
 "hist_gradient_boosting": (lambda: HistGradientBoostingRegressor(random_state=SEED), False,
                            {"learning_rate": loguniform(0.02, 0.3), "max_iter": randint(100, 500),
                             "max_leaf_nodes": randint(8, 64), "l2_regularization": loguniform(1e-3, 10)}),
 "svr": (lambda: SVR(), True,
         {"C": loguniform(1, 1000), "epsilon": loguniform(0.01, 1), "gamma": ["scale", "auto"]}),
}

try:
    from xgboost import XGBRegressor
    ZOO["xgboost"] = (lambda: XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0), False,
                      {"n_estimators": randint(100, 500), "learning_rate": loguniform(0.02, 0.3),
                       "max_depth": randint(2, 8), "subsample": uniform(0.6, 0.4),
                       "colsample_bytree": uniform(0.6, 0.4)})
except Exception:
    pass
try:
    from lightgbm import LGBMRegressor
    ZOO["lightgbm"] = (lambda: LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1), False,
                       {"n_estimators": randint(100, 500), "learning_rate": loguniform(0.02, 0.3),
                        "num_leaves": randint(8, 64), "subsample": uniform(0.6, 0.4)})
except Exception:
    pass

def available(names):
    return [n for n in names if n in ZOO]
