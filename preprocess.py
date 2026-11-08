"""
preprocess.py
==============
Load -> clean -> split -> encode (train-only statistics) -> engineer -> save.

Changes vs. the original version
--------------------------------
* The train / valid / calibration / test split is assigned HERE (stored in a
  `split` column) so every downstream script uses identical rows.
* city_encoded / state_encoded are now fitted on TRAIN rows only, with
  smoothing, and saved to models/encoders.json. Previously they were computed
  on the full dataset (test rows leaked into the feature) and the app had to
  re-read the raw CSV to reproduce them.
* A leak-free, public-anchor feature `city_anchor_psqft` (+ `is_metro`) comes
  from CITY_PROFILES, so cities never seen in training still get sensible
  predictions.
* Non-predictive columns (pin_code, population_density, rental_yield_pct) are
  kept in the cleaned file for reporting but excluded from model features.
"""
import json
import logging
import os

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LEAKY_COLS = ["circle_rate_value", "govt_valuation"]
ONEHOT_COLS = [
    "property_type", "property_category", "facing_direction",
    "construction_quality", "occupancy", "zone",
]


def load_or_generate_dataset() -> pd.DataFrame:
    if not os.path.exists(config.RAW_PATH):
        logger.info("Raw dataset not found -- generating via dataset_loader...")
        from dataset_loader import save_dataset
        save_dataset(path=config.RAW_PATH)
    df = pd.read_csv(config.RAW_PATH)
    logger.info(f"Original shape: {df.shape}")
    return df


def assign_splits(n: int, seed: int = config.RANDOM_STATE) -> np.ndarray:
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    labels = np.empty(n, dtype=object)
    start = 0
    names = list(config.SPLIT_FRACTIONS)
    for i, name in enumerate(names):
        size = n - start if i == len(names) - 1 else int(round(n * config.SPLIT_FRACTIONS[name]))
        labels[order[start:start + size]] = name
        start += size
    return labels


def fit_target_encoding(train_df: pd.DataFrame, key: str, target: str, m: float):
    """Smoothed mean encoding. Returns (mapping dict, global mean)."""
    g_mean = float(train_df[target].mean())
    stats = train_df.groupby(key)[target].agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + m * g_mean) / (stats["count"] + m)
    return smoothed.to_dict(), g_mean


def preprocess(df: pd.DataFrame):
    """Returns (cleaned_df, encoders_dict)."""
    from dataset_loader import CITY_PROFILES

    df = df.copy()

    # 1. defensive missing-value handling
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in df.select_dtypes(include=["object"]).columns:
        mode = df[col].mode()
        df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    # 2. drop identifier columns
    df.drop(columns=[c for c in ["district", "ward", "municipality"] if c in df.columns], inplace=True)

    # 3. outliers + invalid prices BEFORE any statistic is learned
    lower, upper = df["market_price"].quantile([0.01, 0.99])
    before = len(df)
    df = df[(df["market_price"] >= lower) & (df["market_price"] <= upper) & (df["market_price"] > 0)].copy()
    logger.info(f"Outliers removed: {before - len(df)} rows. Remaining: {len(df)}")

    # 4. split assignment, then train-only target encoding
    df = df.reset_index(drop=True)
    df["split"] = assign_splits(len(df))
    train = df[df["split"] == "train"]
    m = config.TARGET_ENCODING_SMOOTHING
    city_map, g_mean = fit_target_encoding(train, "city", "market_price", m)
    state_map, _ = fit_target_encoding(train, "state", "market_price", m)
    df["city_encoded"] = df["city"].map(city_map).fillna(g_mean)
    df["state_encoded"] = df["state"].map(state_map).fillna(g_mean)

    # leak-free public anchors
    df["city_anchor_psqft"] = df["city"].map(lambda c: CITY_PROFILES.get(c, {}).get("avg_price_psqft", np.nan))
    df["city_anchor_psqft"] = df["city_anchor_psqft"].fillna(df["city_anchor_psqft"].median())
    df["is_metro"] = df["city"].map(lambda c: int(CITY_PROFILES.get(c, {}).get("tier") == "metro"))

    encoders = {
        "city_map": city_map, "state_map": state_map, "global_mean": g_mean,
        "smoothing": m, "fitted_on": "train split only",
    }

    # 5. drop leakage-prone columns (kept in RAW file for the tax engine)
    df.drop(columns=[c for c in LEAKY_COLS if c in df.columns], inplace=True)

    # 6. one-hot encode
    onehot = [c for c in ONEHOT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=onehot, drop_first=False)
    df.drop(columns=["city", "state"], inplace=True)

    # 7. feature engineering
    df["property_age"] = (config.CURRENT_YEAR - df["construction_year"]).clip(lower=0)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["amenity_score"] = df[[
        "parking", "nearby_metro", "nearby_school", "nearby_hospital", "nearby_market",
        "lift_available", "swimming_pool", "solar_panels", "rainwater_harvesting", "green_certified",
    ]].sum(axis=1)
    # price = price_per_sqft x area, so in log space the natural predictor is log(area).
    # (Without these, linear models score a *negative* R2 -- see README "Findings".)
    eff_area = np.where(df["property_type_Plot"] == 1, df["plot_area_sqft"], df["built_up_area_sqft"]) \
        if "property_type_Plot" in df.columns else df["built_up_area_sqft"]
    df["log_effective_area"] = np.log(np.maximum(eff_area, 1))
    df["log_city_encoded"] = np.log(df["city_encoded"])
    df["log_city_anchor"] = np.log(df["city_anchor_psqft"])
    # price_per_sqft would leak the target (price = ppsf * area) -- deliberately NOT created.

    # 8. log-transform target
    df["price"] = np.log1p(df["market_price"])
    df.drop(columns=["market_price"], inplace=True)

    logger.info(f"Final shape: {df.shape} | splits: {df['split'].value_counts().to_dict()}")
    return df, encoders


def main():
    raw = load_or_generate_dataset()
    cleaned, encoders = preprocess(raw)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    cleaned.to_csv(config.CLEAN_PATH, index=False)
    with open(os.path.join(config.MODELS_DIR, "encoders.json"), "w") as f:
        json.dump(encoders, f, indent=2)
    logger.info(f"Saved: {config.CLEAN_PATH} and models/encoders.json")


if __name__ == "__main__":
    main()
