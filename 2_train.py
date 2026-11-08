"""
2_train.py
==========
Trains the production XGBoost valuation model.

Improvements over the original
-------------------------------
* Early stopping uses a dedicated VALID split (the old script early-stopped on
  the test set, which contaminates the reported score). The old config also hit
  the 600-tree cap, i.e. it was still under-fitted; this one lets early stopping
  decide.
* Split-conformal calibration on a separate CALIB split gives genuine
  prediction intervals (90% / 80%) instead of a fixed +/-10%.
* Saves a native JSON model (robust across xgboost versions) alongside the
  joblib pickle, plus metadata.json (metrics, intervals, versions).
* Picks up tuned hyper-parameters from models/best_params.json if you ran 4_tune.py.
"""
import json
import logging
import os
import platform
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import config
import conformal as _cf  # noqa: F401
from common import load_splits

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PARAMS = dict(
    n_estimators=3000, max_depth=6, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0, min_child_weight=1,
)
ALPHAS = {"90": 0.10, "80": 0.20}


def conformal_quantile(abs_resid: np.ndarray, alpha: float) -> float:
    n = len(abs_resid)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return float(np.sort(abs_resid)[k - 1])


def metrics_on_price_scale(y_log, pred_log) -> dict:
    a, p = np.expm1(y_log), np.expm1(pred_log)
    return {
        "mae": float(mean_absolute_error(a, p)),
        "rmse": float(np.sqrt(mean_squared_error(a, p))),
        "r2": float(r2_score(a, p)),
        "r2_log": float(r2_score(y_log, pred_log)),
        "mape": float(np.mean(np.abs(a - p) / a) * 100),
        "median_ape": float(np.median(np.abs(a - p) / a) * 100),
    }


def main():
    if not os.path.exists(config.CLEAN_PATH):
        logger.info("Cleaned dataset not found -- running preprocess.py first...")
        import preprocess
        preprocess.main()

    feats, sp = load_splits()
    (X_tr, y_tr), (X_va, y_va), (X_ca, y_ca), (X_te, y_te) = sp["train"], sp["valid"], sp["calib"], sp["test"]
    logger.info(f"Features: {len(feats)} | train {len(X_tr)} | valid {len(X_va)} | calib {len(X_ca)} | test {len(X_te)}")

    params = dict(DEFAULT_PARAMS)
    if os.path.exists(config.BEST_PARAMS_JSON):
        with open(config.BEST_PARAMS_JSON) as f:
            params.update(json.load(f))
        logger.info(f"Using tuned params from {config.BEST_PARAMS_JSON}")

    model = XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse", early_stopping_rounds=50,
        random_state=config.RANDOM_STATE, n_jobs=-1, **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=200)
    logger.info(f"Best iteration: {model.best_iteration}")

    # ---- split-conformal intervals in log space: global + category + property-type (Mondrian) -------
    import conformal as cf
    conformal = cf.build(model, X_ca, y_ca)

    # ---- honest test-set report ------------------------------------------------
    test_pred = model.predict(X_te)
    test_metrics = metrics_on_price_scale(y_te.values, test_pred)
    cover = {}
    for k in ALPHAS:
        q = conformal["global"][k]
        cover[k] = float(np.mean(np.abs(y_te.values - test_pred) <= q))
    logger.info(f"TEST  R2={test_metrics['r2']:.4f}  MAPE={test_metrics['mape']:.2f}%  MAE=Rs {test_metrics['mae']:,.0f}")
    logger.info(f"Interval coverage on test: 90% target -> {cover['90']:.1%} | 80% target -> {cover['80']:.1%}")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model.save_model(config.MODEL_JSON)
    joblib.dump(model, config.MODEL_PKL)
    joblib.dump(feats, config.FEATURES_PKL)

    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "xgboost_version": xgboost.__version__, "python": platform.python_version(),
        "params": params, "best_iteration": int(model.best_iteration),
        "features": feats, "n_rows": {k: len(v[0]) for k, v in sp.items()},
        "test_metrics": test_metrics, "conformal_log_quantiles": conformal,
        "empirical_coverage_test": cover,
    }
    with open(config.METADATA_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved model + metadata -> {config.MODELS_DIR}/")


if __name__ == "__main__":
    main()
