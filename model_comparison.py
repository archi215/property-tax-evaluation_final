"""
model_comparison.py
====================
Trains and compares 11 regression models on the Indian property
dataset, mirroring the structure of the original project's comparison
script. Produces a ranked comparison table and diagnostic plots.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

DATA_PATH = "data/cleaned_data.csv"
OUTPUTS_DIR = "outputs"


def main():
    # ============================================================
    # STEP 1: LOAD DATA
    # ============================================================
    df = pd.read_csv(DATA_PATH)
    print("Dataset Shape:", df.shape)

    # ============================================================
    # STEP 2: PREPROCESSING (already mostly done in preprocess.py;
    # this step just guards against any remaining missing values)
    # ============================================================
    df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=["price"])
    y = df["price"]
    print("\nFeatures shape:", X.shape)
    print("Target shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ============================================================
    # STEP 3: DEFINE ALL MODELS
    # ============================================================
    models = {
        "Linear Regression": {"model": LinearRegression(), "scaled": True},
        "Ridge Regression": {"model": Ridge(alpha=10), "scaled": True},
        "Lasso Regression": {"model": Lasso(alpha=0.01), "scaled": True},
        "ElasticNet": {"model": ElasticNet(alpha=0.01, l1_ratio=0.5), "scaled": True},
        "Decision Tree": {
            "model": DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42),
            "scaled": False,
        },
        "Random Forest": {
            "model": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            "scaled": False,
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
            "scaled": False,
        },
        "SVR": {"model": SVR(kernel="rbf", C=100, epsilon=0.1), "scaled": True},
        "KNN": {"model": KNeighborsRegressor(n_neighbors=7, weights="distance"), "scaled": True},
        "AdaBoost": {
            "model": AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
            "scaled": False,
        },
        "Extra Trees": {
            "model": ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            "scaled": False,
        },
        "XGBoost": {
            "model": XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=10, reg_lambda=10,
                random_state=42,
            ),
            "scaled": False,
        },
    }

    # ============================================================
    # STEP 4: TRAIN AND EVALUATE ALL MODELS
    # ============================================================
    results = []
    trained_models = {}

    print("\n" + "=" * 60)
    print("TRAINING ALL MODELS...")
    print("=" * 60)

    for name, config in models.items():
        print(f"\nTraining: {name}...")
        model = config["model"]
        scaled = config["scaled"]

        if scaled:
            model.fit(X_train_scaled, y_train)
            pred_log = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            pred_log = model.predict(X_test)

        pred_actual = np.expm1(pred_log)
        y_test_actual = np.expm1(y_test)

        mae = mean_absolute_error(y_test_actual, pred_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, pred_actual))
        r2 = r2_score(y_test_actual, pred_actual)

        results.append({
            "Model": name,
            "MAE (Rs)": round(mae),
            "RMSE (Rs)": round(rmse),
            "R\u00b2 Score": round(r2, 4),
        })

        trained_models[name] = model
        print(f"  MAE: Rs {mae:,.0f} | RMSE: Rs {rmse:,.0f} | R\u00b2: {r2:.4f}")

    # ============================================================
    # STEP 5: RESULTS TABLE
    # ============================================================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("R\u00b2 Score", ascending=False).reset_index(drop=True)
    results_df.index += 1

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string())

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ============================================================
    # STEP 6: PLOTS
    # ============================================================
    plt.figure(figsize=(14, 6))
    colors = ["gold" if m == "XGBoost" else "steelblue" for m in results_df["Model"]]
    bars = plt.bar(results_df["Model"], results_df["R\u00b2 Score"], color=colors, edgecolor="black", linewidth=0.5)
    plt.axhline(y=0.7, color="red", linestyle="--", linewidth=1.5, label="Threshold (0.70)")
    plt.title("R\u00b2 Score Comparison \u2014 All Models", fontsize=14)
    plt.xlabel("Model")
    plt.ylabel("R\u00b2 Score")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    for bar, val in zip(bars, results_df["R\u00b2 Score"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}",
                  ha="center", va="bottom", fontsize=8)
    plt.savefig(os.path.join(OUTPUTS_DIR, "r2_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: {OUTPUTS_DIR}/r2_comparison.png")

    plt.figure(figsize=(14, 6))
    colors2 = ["gold" if m == "XGBoost" else "salmon" for m in results_df["Model"]]
    bars2 = plt.bar(results_df["Model"], results_df["RMSE (Rs)"], color=colors2, edgecolor="black", linewidth=0.5)
    plt.title("RMSE Comparison \u2014 All Models (Lower is Better)", fontsize=14)
    plt.xlabel("Model")
    plt.ylabel("RMSE (Rs)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    for bar, val in zip(bars2, results_df["RMSE (Rs)"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(results_df["RMSE (Rs)"]) * 0.01,
                  f"Rs {val:,.0f}", ha="center", va="bottom", fontsize=7)
    plt.savefig(os.path.join(OUTPUTS_DIR, "rmse_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: {OUTPUTS_DIR}/rmse_comparison.png")

    plt.figure(figsize=(14, 6))
    colors3 = ["gold" if m == "XGBoost" else "lightgreen" for m in results_df["Model"]]
    bars3 = plt.bar(results_df["Model"], results_df["MAE (Rs)"], color=colors3, edgecolor="black", linewidth=0.5)
    plt.title("MAE Comparison \u2014 All Models (Lower is Better)", fontsize=14)
    plt.xlabel("Model")
    plt.ylabel("MAE (Rs)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    for bar, val in zip(bars3, results_df["MAE (Rs)"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(results_df["MAE (Rs)"]) * 0.01,
                  f"Rs {val:,.0f}", ha="center", va="bottom", fontsize=7)
    plt.savefig(os.path.join(OUTPUTS_DIR, "mae_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: {OUTPUTS_DIR}/mae_comparison.png")

    tree_models = {
        "XGBoost": trained_models["XGBoost"],
        "Random Forest": trained_models["Random Forest"],
        "Extra Trees": trained_models["Extra Trees"],
        "Gradient Boosting": trained_models["Gradient Boosting"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, (name, model) in enumerate(tree_models.items()):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        axes[idx].barh(imp.index[::-1], imp.values[::-1], color="steelblue")
        axes[idx].set_title(f"{name} \u2014 Top 10 Features")
        axes[idx].set_xlabel("Importance Score")
    plt.suptitle("Feature Importance \u2014 Tree Based Models", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "feature_importance_all.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUTS_DIR}/feature_importance_all.png")

    # ============================================================
    # STEP 7: BIAS-VARIANCE NOTES
    # ============================================================
    print("\n" + "=" * 60)
    print("BIAS-VARIANCE TRADEOFF ANALYSIS")
    print("=" * 60)
    bias_variance = {
        "Linear Regression": ("High Bias", "Low Variance", "Underfits -- relationships are non-linear"),
        "Ridge Regression": ("High Bias", "Low Variance", "Better than Linear -- reduces overfitting"),
        "Lasso Regression": ("High Bias", "Low Variance", "Does feature selection automatically"),
        "ElasticNet": ("Medium Bias", "Low Variance", "Combines Ridge and Lasso"),
        "Decision Tree": ("Low Bias", "High Variance", "Overfits easily without depth control"),
        "Random Forest": ("Low Bias", "Medium Variance", "Good balance -- reduces overfitting"),
        "Gradient Boosting": ("Low Bias", "Medium Variance", "Learns from errors sequentially"),
        "SVR": ("Medium Bias", "Medium Variance", "Good for small/medium datasets"),
        "KNN": ("Low Bias", "High Variance", "Sensitive to irrelevant features"),
        "AdaBoost": ("Low Bias", "Medium Variance", "Focuses on hard examples"),
        "Extra Trees": ("Low Bias", "Medium Variance", "Faster than Random Forest"),
        "XGBoost": ("Low Bias", "Low Variance", "Best balance -- regularization built in"),
    }
    bv_df = pd.DataFrame(
        [(k, v[0], v[1], v[2]) for k, v in bias_variance.items()],
        columns=["Model", "Bias", "Variance", "Notes"],
    )
    print(bv_df.to_string(index=False))

    # ============================================================
    # STEP 8: FINAL WINNER
    # ============================================================
    best_model = results_df.iloc[0]
    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)
    print(f"Model:     {best_model['Model']}")
    print(f"R\u00b2 Score:  {best_model['R\u00b2 Score']}")
    print(f"MAE:       Rs {best_model['MAE (Rs)']:,}")
    print(f"RMSE:      Rs {best_model['RMSE (Rs)']:,}")

    results_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison_results.csv"))
    print(f"\nSaved: {OUTPUTS_DIR}/model_comparison_results.csv")

    return results_df


if __name__ == "__main__":
    main()
