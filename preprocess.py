"""
preprocess.py
==============
Data preprocessing for the Indian property dataset.
Mirrors the original project's preprocessing structure (load -> clean ->
feature engineer -> outlier removal -> save) but adapted for Indian
property fields instead of the original US housing schema.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "india_property_data.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")


def load_or_generate_dataset() -> pd.DataFrame:
    """Load the dataset, generating it first if it doesn't exist yet."""
    if not os.path.exists(RAW_PATH):
        logger.info("Raw dataset not found -- generating it now via dataset_loader...")
        from dataset_loader import save_dataset
        save_dataset(path=RAW_PATH)
    df = pd.read_csv(RAW_PATH)
    logger.info(f"Original shape: {df.shape}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── 1. Handle missing values defensively ──────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
    logger.info("Missing values handled")

    # ── 2. Drop pure-identifier columns not useful for prediction ─────
    drop_cols = [c for c in ["district", "ward", "municipality"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    logger.info(f"Dropped identifier columns: {drop_cols}")

    # ── 3. Target encode high-cardinality categoricals (city, zone) ───
    city_mean = df.groupby("city")["market_price"].mean()
    df["city_encoded"] = df["city"].map(city_mean)

    state_mean = df.groupby("state")["market_price"].mean()
    df["state_encoded"] = df["state"].map(state_mean)
    logger.info("City and state target-encoded")

    # ── 3b. Drop circle_rate_value / govt_valuation from PREDICTIVE features ──
    # These are derived in dataset_loader.py as market_price * ratio + small noise,
    # i.e. they are near-deterministic functions of the prediction target. Leaving
    # them in would let the model "predict" price by trivially inverting that
    # formula (verified: they alone explained ~99% of feature importance),
    # which defeats the purpose of a feature-importance-driven valuation model
    # and would not reflect how real buyers reason about value. We keep these
    # two columns in the RAW dataset for the tax_engine (which legitimately
    # needs circle rate for stamp duty), but exclude them from X here.
    LEAKY_COLS = ["circle_rate_value", "govt_valuation"]
    df.drop(columns=[c for c in LEAKY_COLS if c in df.columns], inplace=True, errors="ignore")
    logger.info(f"Dropped leakage-prone columns from model features: {LEAKY_COLS}")

    # ── 4. One-hot encode low-cardinality categoricals ─────────────────
    onehot_cols = [
        "property_type", "property_category", "facing_direction",
        "construction_quality", "occupancy", "zone",
    ]
    onehot_cols = [c for c in onehot_cols if c in df.columns]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=False)
    logger.info(f"One-hot encoded: {onehot_cols}")

    # Drop raw city/state text columns now that they're encoded numerically
    df.drop(columns=["city", "state"], inplace=True, errors="ignore")

    # ── 5. Feature engineering ─────────────────────────────────────────
    df["property_age"] = (2026 - df["construction_year"]).clip(lower=0)

    # NOTE: a "price_per_sqft" feature computed as market_price / area would
    # also leak the target (it's circular: price = price_per_sqft * area).
    # We do NOT include it as a model feature for that reason.

    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["amenity_score"] = (
        df["parking"] + df["nearby_metro"] + df["nearby_school"] + df["nearby_hospital"]
        + df["nearby_market"] + df["lift_available"] + df["swimming_pool"]
        + df["solar_panels"] + df["rainwater_harvesting"] + df["green_certified"]
    )
    logger.info("New features engineered: property_age, total_rooms, amenity_score")

    # ── 6. Remove top/bottom 1% price outliers ─────────────────────────
    lower = df["market_price"].quantile(0.01)
    upper = df["market_price"].quantile(0.99)
    before = len(df)
    df = df[(df["market_price"] >= lower) & (df["market_price"] <= upper)]
    logger.info(f"Outliers removed: {before - len(df)} rows. Remaining: {len(df)}")

    # ── 7. Remove zero/negative prices ─────────────────────────────────
    df = df[df["market_price"] > 0]

    # ── 8. Log-transform target (helps tree-based models a lot) ───────
    df["price"] = np.log1p(df["market_price"])
    df.drop(columns=["market_price"], inplace=True)

    logger.info(f"\nFinal shape: {df.shape}")
    logger.info(f"\nColumns: {df.columns.tolist()}")
    return df


def main():
    df = load_or_generate_dataset()
    cleaned = preprocess(df)
    os.makedirs(DATA_DIR, exist_ok=True)
    cleaned.to_csv(CLEAN_PATH, index=False)
    logger.info(f"\nSaved: {CLEAN_PATH}")


if __name__ == "__main__":
    main()
