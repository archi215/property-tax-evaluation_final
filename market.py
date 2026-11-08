"""market.py -- market-context helpers built from the cleaned dataset (no Streamlit)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config
from dataset_loader import CITY_PROFILES


def load_market_frame() -> pd.DataFrame:
    """Reconstructs a human-readable frame (city, property_type, price, price/sqft) from
    cleaned_data.csv + encoders.json, so the app no longer needs the raw CSV."""
    if not os.path.exists(config.CLEAN_PATH):
        return pd.DataFrame()
    df = pd.read_csv(config.CLEAN_PATH)
    enc_path = os.path.join(config.MODELS_DIR, "encoders.json")
    city_inv = {}
    if os.path.exists(enc_path):
        cmap = json.load(open(enc_path))["city_map"]
        city_inv = {round(v, 4): k for k, v in cmap.items()}
    df["city"] = df["city_encoded"].round(4).map(city_inv).fillna("Unknown")
    type_cols = [c for c in df.columns if c.startswith("property_type_")]
    df["property_type"] = df[type_cols].idxmax(axis=1).str.replace("property_type_", "", regex=False)
    df["state"] = df["city"].map(lambda c: CITY_PROFILES.get(c, {}).get("state", "Unknown"))
    df["market_price"] = np.expm1(df["price"])
    area = np.where(df["property_type"] == "Plot", df["plot_area_sqft"], df["built_up_area_sqft"])
    df["price_per_sqft"] = df["market_price"] / np.maximum(area, 1)
    return df


def price_percentile(frame: pd.DataFrame, city: str, property_type: str, psqft: float):
    """Where a given price/sqft falls among comparable rows (0-100), plus the peer summary."""
    peers = frame[(frame["city"] == city) & (frame["property_type"] == property_type)]["price_per_sqft"]
    if len(peers) < 20:
        peers = frame[frame["city"] == city]["price_per_sqft"]
    if len(peers) < 20:
        return None
    return {"percentile": float((peers < psqft).mean() * 100), "n": int(len(peers)),
            "p10": float(peers.quantile(.1)), "median": float(peers.median()), "p90": float(peers.quantile(.9))}
