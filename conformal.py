"""conformal.py -- split-conformal quantiles, global and group-conditional (Mondrian) by category and property type."""
from __future__ import annotations

import numpy as np

ALPHAS = {"90": 0.10, "80": 0.20}
MIN_CATEGORY = 80   # calibration rows needed for a property-category quantile
MIN_TYPE = 30       # calibration rows needed for a property-type quantile


def quantile(abs_resid: np.ndarray, alpha: float) -> float:
    n = len(abs_resid)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(abs_resid)[k - 1])


def build(model, X_cal, y_cal) -> dict:
    """Quantiles in log space: global, by_category (>=80 rows), by_type (>=30 rows)."""
    res = np.abs(np.asarray(y_cal) - model.predict(X_cal))
    out = {"global": {k: quantile(res, a) for k, a in ALPHAS.items()}, "by_category": {}, "by_type": {}}
    for prefix, key, minimum in (("property_category_", "by_category", MIN_CATEGORY), ("property_type_", "by_type", MIN_TYPE)):
        for col in [c for c in X_cal.columns if c.startswith(prefix)]:
            mask = X_cal[col].values.astype(bool)
            if mask.sum() >= minimum:
                out[key][col[len(prefix):]] = {k: quantile(res[mask], a) for k, a in ALPHAS.items()}
                out[key][col[len(prefix):]]["n"] = int(mask.sum())
    return out


def pick(conformal: dict, category: str, ptype: str, level: str) -> float:
    """Most specific available quantile: property type, then category, then global."""
    for key, name in (("by_type", ptype), ("by_category", category)):
        if name in conformal.get(key, {}):
            return conformal[key][name][level]
    return conformal.get("global", {}).get(level, 0.10)
