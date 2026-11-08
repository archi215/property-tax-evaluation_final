"""config.py -- single source of truth for paths and shared constants."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
TAX_RULES_DIR = os.path.join(ROOT, "tax_rules")

RAW_PATH = os.path.join(DATA_DIR, "india_property_data.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
HISTORY_DB = os.path.join(DATA_DIR, "history.db")

MODEL_JSON = os.path.join(MODELS_DIR, "property_model.json")   # native, version-robust
MODEL_PKL = os.path.join(MODELS_DIR, "property_model.pkl")     # joblib, back-compat
FEATURES_PKL = os.path.join(MODELS_DIR, "features.pkl")
METADATA_JSON = os.path.join(MODELS_DIR, "metadata.json")
BEST_PARAMS_JSON = os.path.join(MODELS_DIR, "best_params.json")

CURRENT_YEAR = 2026
RANDOM_STATE = 42

# Columns that are NOT price drivers in the generator (pure noise or outcomes
# of price) and are excluded from the model's feature matrix.
#   pin_code            -> random draw around a constant, carries no signal
#   population_density  -> random draw, independent of price
#   rental_yield_pct    -> an *outcome* of the property, not a price driver
NON_PREDICTIVE_COLS = ["pin_code", "population_density", "rental_yield_pct"]

# Split proportions (train / valid / calibration / test)
SPLIT_FRACTIONS = {"train": 0.70, "valid": 0.10, "calib": 0.10, "test": 0.10}

TARGET_ENCODING_SMOOTHING = 20  # pseudo-count pulling rare cities to the global mean
