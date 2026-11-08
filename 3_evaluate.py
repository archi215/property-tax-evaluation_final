"""
3_evaluate.py
=============
Evaluates the production model on the held-out TEST split and writes:
  outputs/metrics.json            overall + per-segment metrics, interval coverage
  outputs/evaluation.png          actual-vs-predicted + residuals
  outputs/feature_importance.png  top-15 features (gain)
  outputs/error_by_segment.png    MAPE by property type and city tier
  outputs/calibration.png         empirical vs nominal interval coverage
"""
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from common import load_splits
from valuation import ValuationModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _mape(a, p):
    return float(np.mean(np.abs(a - p) / a) * 100)


def main():
    vm = ValuationModel()
    feats, sp = load_splits()
    X_te, y_te = sp["test"]
    pred_log = vm.model.predict(X_te[vm.features])
    a, p = np.expm1(y_te.values), np.expm1(pred_log)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    overall = {"mae": float(mean_absolute_error(a, p)), "rmse": float(np.sqrt(mean_squared_error(a, p))),
               "r2": float(r2_score(a, p)), "mape": _mape(a, p), "median_ape": float(np.median(np.abs(a - p) / a) * 100),
               "n_test": int(len(a))}
    logger.info(f"R2 {overall['r2']:.4f} | MAPE {overall['mape']:.2f}% | MAE Rs {overall['mae']:,.0f} | RMSE Rs {overall['rmse']:,.0f}")

    # ---- per-segment error --------------------------------------------------
    seg = X_te.copy()
    seg["ape"] = np.abs(a - p) / a * 100
    ptype_cols = [c for c in seg.columns if c.startswith("property_type_")]
    seg["property_type"] = seg[ptype_cols].idxmax(axis=1).str.replace("property_type_", "", regex=False)
    seg["tier"] = np.where(seg["is_metro"] == 1, "Metro", "Tier-2/3")
    by_type = seg.groupby("property_type")["ape"].agg(["mean", "count"]).rename(columns={"mean": "mape"})
    by_tier = seg.groupby("tier")["ape"].agg(["mean", "count"]).rename(columns={"mean": "mape"})

    # ---- interval coverage --------------------------------------------------
    cover = {}
    for level, nominal in (("80", 0.80), ("90", 0.90)):
        q = vm.conformal["global"][level]
        cover[level] = {"nominal": nominal, "empirical": float(np.mean(np.abs(y_te.values - pred_log) <= q))}
    logger.info(f"Coverage: {cover}")

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUTS_DIR, "metrics.json"), "w") as f:
        json.dump({"overall": overall, "by_property_type": by_type.round(3).to_dict("index"),
                   "by_tier": by_tier.round(3).to_dict("index"), "interval_coverage": cover}, f, indent=2)

    # ---- plots ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].scatter(a, p, alpha=0.3, color="steelblue", s=10)
    lim = [a.min(), a.max()]
    ax[0].plot(lim, lim, "r--", lw=2)
    ax[0].set_xlabel("Actual price (Rs)"); ax[0].set_ylabel("Predicted price (Rs)"); ax[0].set_title("Actual vs Predicted (test set)")
    ax[1].hist((pred_log - y_te.values) * 100, bins=50, color="salmon", edgecolor="black")
    ax[1].axvline(0, color="red", ls="--")
    ax[1].set_title("Relative error distribution (log-points x100 \u2248 % error)")
    ax[1].set_xlabel("Predicted - Actual (%)")
    plt.tight_layout(); plt.savefig(os.path.join(config.OUTPUTS_DIR, "evaluation.png"), dpi=150); plt.close(fig)

    booster = vm.model.get_booster()
    imp = pd.Series(booster.get_score(importance_type="gain")).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6)); imp.sort_values().plot(kind="barh", color="steelblue")
    plt.title("Top 15 features (average gain)"); plt.xlabel("Gain"); plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, "feature_importance.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    by_type["mape"].sort_values().plot(kind="barh", ax=ax[0], color="#1B3A6B"); ax[0].set_title("MAPE by property type (%)")
    by_tier["mape"].plot(kind="bar", ax=ax[1], color="#FF8C42", rot=0); ax[1].set_title("MAPE by city tier (%)")
    plt.tight_layout(); plt.savefig(os.path.join(config.OUTPUTS_DIR, "error_by_segment.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    xs = [cover[k]["nominal"] for k in ("80", "90")]; ys = [cover[k]["empirical"] for k in ("80", "90")]
    ax.plot([0.7, 1], [0.7, 1], "k--", lw=1); ax.plot(xs, ys, "o-", color="#2E7D5B")
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage (test)"); ax.set_title("Interval calibration")
    plt.tight_layout(); plt.savefig(os.path.join(config.OUTPUTS_DIR, "calibration.png"), dpi=150); plt.close(fig)
    logger.info("Saved evaluation outputs -> outputs/")
    return overall


if __name__ == "__main__":
    main()
