"""
4_tune.py  (optional)
=====================
Randomised hyper-parameter search for the XGBoost model using K-fold CV on the
TRAIN split only (valid/calib/test are never touched). Writes the best
parameters to models/best_params.json; 2_train.py then picks them up.

    python 4_tune.py --iters 25 --folds 3
"""
import argparse
import json
import logging

import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

import config
from common import load_splits

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main(iters: int = 25, folds: int = 3):
    feats, sp = load_splits()
    X_tr, y_tr = sp["train"]
    space = {
        "max_depth": randint(3, 9),
        "learning_rate": uniform(0.02, 0.08),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 8),
        "reg_alpha": uniform(0, 5),
        "reg_lambda": uniform(0.5, 8),
        "gamma": uniform(0, 0.3),
    }
    base = XGBRegressor(n_estimators=600, random_state=config.RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(
        base, space, n_iter=iters, cv=folds, scoring="neg_root_mean_squared_error",
        random_state=config.RANDOM_STATE, verbose=1,
    )
    search.fit(X_tr, y_tr)
    best = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v)) for k, v in search.best_params_.items()}
    # n_estimators is chosen by early stopping in 2_train.py, give it headroom
    best["n_estimators"] = 3000
    logger.info(f"Best CV RMSE (log space): {-search.best_score_:.4f}")
    logger.info(f"Best params: {best}")
    with open(config.BEST_PARAMS_JSON, "w") as f:
        json.dump(best, f, indent=2)
    logger.info(f"Saved -> {config.BEST_PARAMS_JSON}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--folds", type=int, default=3)
    a = ap.parse_args()
    main(a.iters, a.folds)
