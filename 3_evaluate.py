"""
3_evaluate.py
=============
Evaluates the trained XGBoost model on the Indian property test set
and produces diagnostic plots, mirroring the original project's
evaluation script.
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/cleaned_data.csv"
MODEL_PATH = "models/property_model.pkl"
OUTPUTS_DIR = "outputs"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded")

    pred_log = model.predict(X_test)
    pred_actual = np.expm1(pred_log)
    y_test_actual = np.expm1(y_test)

    mae = mean_absolute_error(y_test_actual, pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, pred_actual))
    r2 = r2_score(y_test_actual, pred_actual)
    mask = y_test_actual > 0
    mape = np.mean(np.abs((y_test_actual[mask] - pred_actual[mask]) / y_test_actual[mask])) * 100

    logger.info("\nModel Performance:")
    logger.info(f"   MAE   : Rs {mae:,.0f}")
    logger.info(f"   RMSE  : Rs {rmse:,.0f}")
    logger.info(f"   R\u00b2    : {r2:.4f}")
    logger.info(f"   MAPE  : {mape:.2f}%")

    if r2 >= 0.80:
        logger.info("\nGreat model! R\u00b2 is above 0.80")
    elif r2 >= 0.60:
        logger.info("\nDecent model. R\u00b2 could be improved")
    else:
        logger.info("\nModel needs improvement")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test_actual, pred_actual, alpha=0.3, color="steelblue")
    axes[0].plot(
        [y_test_actual.min(), y_test_actual.max()],
        [y_test_actual.min(), y_test_actual.max()],
        "r--", lw=2,
    )
    axes[0].set_xlabel("Actual Price (Rs)")
    axes[0].set_ylabel("Predicted Price (Rs)")
    axes[0].set_title("Actual vs Predicted")

    residuals = y_test_actual - pred_actual
    axes[1].hist(residuals, bins=50, color="salmon", edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Residuals Distribution")
    axes[1].set_xlabel("Residual (Actual - Predicted)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "evaluation.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {OUTPUTS_DIR}/evaluation.png")

    imp = pd.Series(model.feature_importances_, index=X.columns)
    top15 = imp.sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    top15.sort_values().plot(kind="barh", color="steelblue")
    plt.title("Top 15 Important Features")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {OUTPUTS_DIR}/feature_importance.png")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


if __name__ == "__main__":
    main()
