"""common.py -- shared data loading for training / evaluation / comparison."""
import pandas as pd
import config


def feature_columns(df: pd.DataFrame):
    """Model features = everything except target, split label, and non-predictive cols."""
    drop = {"price", "split", *config.NON_PREDICTIVE_COLS}
    return [c for c in df.columns if c not in drop]


def load_splits(path: str = config.CLEAN_PATH):
    """Returns (features, dict split_name -> (X, y))."""
    df = pd.read_csv(path)
    feats = feature_columns(df)
    out = {}
    for name in ("train", "valid", "calib", "test"):
        part = df[df["split"] == name]
        out[name] = (part[feats], part["price"])
    return feats, out
