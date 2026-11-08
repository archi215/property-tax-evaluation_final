"""
model_comparison.py
====================
Trains and compares regression models on the SAME train/test split used by the
production pipeline (see preprocess.py). Models are fitted on TRAIN and scored
on the held-out TEST split. LightGBM / CatBoost are included when installed.

Outputs (outputs/): model_comparison_results.csv, r2/rmse/mae comparison PNGs,
feature_importance_all.png
"""
import os
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import config
from common import load_splits

PRODUCTION = "XGBoost"


def build_models():
    models = {
        "Baseline (mean)": (DummyRegressor(), True),
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression": (Ridge(alpha=10), True),
        "Lasso Regression": (Lasso(alpha=0.01), True),
        "ElasticNet": (ElasticNet(alpha=0.01, l1_ratio=0.5), True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42), False),
        "Random Forest": (RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42), False),
        "SVR": (SVR(kernel="rbf", C=100, epsilon=0.1), True),
        "KNN": (KNeighborsRegressor(n_neighbors=7, weights="distance"), True),
        "AdaBoost": (AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42), False),
        "Extra Trees": (ExtraTreesRegressor(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1), False),
        PRODUCTION: (XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                  reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1), False),
    }
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = (LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=48, subsample=0.8,
                                            colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1), False)
    except Exception:
        print("(LightGBM not installed -- skipping)")
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = (CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, random_seed=42, verbose=0), False)
    except Exception:
        print("(CatBoost not installed -- skipping)")
    return models


def _bar(df, col, fname, title, ylabel, base_color, fmt, hline=None):
    plt.figure(figsize=(14, 6))
    colors = ["gold" if m == PRODUCTION else base_color for m in df["Model"]]
    bars = plt.bar(df["Model"], df[col], color=colors, edgecolor="black", linewidth=0.5)
    if hline is not None:
        plt.axhline(hline, color="red", ls="--", lw=1.5, label=f"Threshold ({hline})"); plt.legend()
    plt.title(title, fontsize=14); plt.ylabel(ylabel); plt.xticks(rotation=45, ha="right")
    top = max(df[col].max(), 1e-9)
    for b, v in zip(bars, df[col]):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.01, fmt(v), ha="center", va="bottom", fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(config.OUTPUTS_DIR, fname), dpi=150); plt.close()


def main():
    feats, sp = load_splits()
    X_tr, y_tr = sp["train"]
    X_te, y_te = sp["test"]
    print(f"Train {X_tr.shape} | Test {X_te.shape}")
    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    a = np.expm1(y_te.values)

    results, trained = [], {}
    for name, (model, scaled) in build_models().items():
        t0 = time.time()
        model.fit(Xs_tr if scaled else X_tr, y_tr)
        pred = np.expm1(model.predict(Xs_te if scaled else X_te))
        mae = mean_absolute_error(a, pred); rmse = np.sqrt(mean_squared_error(a, pred)); r2 = r2_score(a, pred)
        mape = float(np.mean(np.abs(a - pred) / a) * 100)
        results.append({"Model": name, "MAE (Rs)": round(mae), "RMSE (Rs)": round(rmse), "R\u00b2 Score": round(r2, 4),
                        "MAPE (%)": round(mape, 2), "Fit time (s)": round(time.time() - t0, 1)})
        trained[name] = model
        print(f"{name:20s} R2={r2:8.4f}  MAPE={mape:7.2f}%  MAE=Rs {mae:,.0f}  ({time.time()-t0:.1f}s)")

    res = pd.DataFrame(results).sort_values("R\u00b2 Score", ascending=False).reset_index(drop=True)
    res.index += 1
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(config.OUTPUTS_DIR, "model_comparison_results.csv"))
    print("\n" + res.to_string())

    plot_df = res[res["Model"] != "Baseline (mean)"]
    _bar(plot_df, "R\u00b2 Score", "r2_comparison.png", "R\u00b2 Score \u2014 all models (test set)", "R\u00b2", "steelblue", lambda v: f"{v:.3f}", hline=0.7)
    _bar(res, "RMSE (Rs)", "rmse_comparison.png", "RMSE \u2014 lower is better", "RMSE (Rs)", "salmon", lambda v: f"{v/1e5:.1f}L")
    _bar(res, "MAE (Rs)", "mae_comparison.png", "MAE \u2014 lower is better", "MAE (Rs)", "lightgreen", lambda v: f"{v/1e5:.1f}L")

    tree_names = [n for n in (PRODUCTION, "LightGBM", "Random Forest", "Extra Trees", "Gradient Boosting") if n in trained][:4]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12)); axes = axes.flatten()
    for ax, n in zip(axes, tree_names):
        imp = pd.Series(trained[n].feature_importances_, index=X_tr.columns).sort_values(ascending=False).head(10)
        ax.barh(imp.index[::-1], imp.values[::-1], color="steelblue"); ax.set_title(f"{n} \u2014 top 10 features")
    plt.suptitle("Feature importance \u2014 tree-based models", fontsize=16, y=1.02); plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, "feature_importance_all.png"), dpi=150, bbox_inches="tight"); plt.close()

    best = res.iloc[0]
    print(f"\nBEST: {best['Model']}  R2={best['R\u00b2 Score']}  MAPE={best['MAPE (%)']}%")
    return res


if __name__ == "__main__":
    main()
