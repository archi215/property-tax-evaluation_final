"""
2_train.py
==========
Trains the production XGBoost price-prediction model on the cleaned
Indian property dataset. Mirrors the structure of the original
project's training script.
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/cleaned_data.csv"
MODELS_DIR = "models"


def main():
    if not os.path.exists(DATA_PATH):
        logger.info("Cleaned dataset not found -- running preprocess.py first...")
        import preprocess
        preprocess.main()

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded shape: {df.shape}")

    X = df.drop(columns=["price"])
    y = df["price"]
    logger.info(f"Total Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=10,
        reg_lambda=10,
        objective="reg:squarederror",
        random_state=42,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )

    logger.info("\nTraining started...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    logger.info("Model trained successfully!")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "property_model.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, "features.pkl"))
    logger.info(f"Saved: {MODELS_DIR}/property_model.pkl")
    logger.info(f"Saved: {MODELS_DIR}/features.pkl")

    pred = np.expm1(model.predict(X_test))
    y_test_actual = np.expm1(y_test)
    r2 = r2_score(y_test_actual, pred)
    logger.info(f"R\u00b2 after reverse transform: {r2:.4f}")


if __name__ == "__main__":
    main()
