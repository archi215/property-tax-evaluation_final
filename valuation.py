"""
valuation.py
============
All model-facing logic, with NO Streamlit dependency (so it is unit-testable
and reusable from scripts / notebooks / an API):

    ValuationModel.predict(inputs)      -> price + conformal intervals
    ValuationModel.explain(inputs)      -> per-group price drivers (exact SHAP via XGBoost)
    ValuationModel.what_if(...)         -> value uplift from improvements
    ValuationModel.sensitivity(...)     -> price curve as one input varies
    ValuationModel.predict_batch(df)    -> bulk valuation for a DataFrame of inputs
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

import config
from dataset_loader import CITY_PROFILES

BOOL_FIELDS = [
    "parking", "nearby_metro", "nearby_school", "nearby_hospital", "nearby_market", "nearby_airport",
    "nearby_railway_station", "corner_plot", "lift_available", "swimming_pool", "solar_panels",
    "rainwater_harvesting", "green_certified",
]

# feature -> human-friendly driver group (prefix match for one-hot columns)
_GROUP_RULES = [
    ("Location", ["city_encoded", "state_encoded", "city_anchor_psqft", "log_city_encoded", "log_city_anchor", "is_metro", "latitude", "longitude", "zone_"]),
    ("Size & layout", ["built_up_area_sqft", "carpet_area_sqft", "plot_area_sqft", "num_floors",
                       "floor_number", "bedrooms", "bathrooms", "total_rooms", "log_effective_area"]),
    ("Property type", ["property_type_", "property_category_"]),
    ("Construction quality", ["construction_quality_"]),
    ("Age", ["construction_year", "property_age"]),
    ("Amenities & green features", ["parking", "lift_available", "swimming_pool", "solar_panels",
                                    "rainwater_harvesting", "green_certified", "amenity_score"]),
    ("Surroundings & access", ["nearby_", "road_width_ft", "corner_plot"]),
    ("Facing & occupancy", ["facing_direction_", "occupancy_"]),
]


def feature_group(name: str) -> str:
    for group, keys in _GROUP_RULES:
        for k in keys:
            if name == k or (k.endswith("_") and name.startswith(k)):
                return group
    return "Other"


# Illustrative improvement catalogue. Costs are EDITABLE placeholders, not quotes.
IMPROVEMENTS = {
    "solar_panels": {"label": "Install rooftop solar", "changes": {"solar_panels": True}, "cost": 200_000},
    "rainwater_harvesting": {"label": "Add rainwater harvesting", "changes": {"rainwater_harvesting": True}, "cost": 50_000},
    "green_certified": {"label": "Get green-building certification", "changes": {"green_certified": True}, "cost": 300_000},
    "parking": {"label": "Add dedicated parking", "changes": {"parking": True}, "cost": 500_000},
    "quality_up": {"label": "Renovate: one construction-quality tier up", "changes": {"__quality_up__": True}, "cost": None, "cost_per_sqft": 1500},
}
QUALITY_ORDER = ["Basic", "Standard", "Premium", "Luxury"]


@dataclass
class Prediction:
    price: float
    lower_90: float
    upper_90: float
    lower_80: float
    upper_80: float
    pred_log: float
    warnings: list = field(default_factory=list)

    @property
    def interval_pct(self) -> float:
        return (self.upper_90 - self.lower_90) / 2 / self.price * 100 if self.price else 0.0


class ValuationModel:
    def __init__(self, models_dir: str = config.MODELS_DIR):
        self.models_dir = models_dir
        self.model = self._load_model()
        meta_path = os.path.join(models_dir, "metadata.json")
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        if self.meta.get("features"):
            self.features = self.meta["features"]
        else:
            self.features = joblib.load(os.path.join(models_dir, "features.pkl"))
        enc_path = os.path.join(models_dir, "encoders.json")
        self.encoders = json.load(open(enc_path)) if os.path.exists(enc_path) else {}
        self.conformal = self.meta.get("conformal_log_quantiles", {})

    # ---------------------------------------------------------------- loading
    def _load_model(self):
        jpath = os.path.join(self.models_dir, "property_model.json")
        if os.path.exists(jpath):
            m = xgb.XGBRegressor()
            m.load_model(jpath)
            return m
        return joblib.load(os.path.join(self.models_dir, "property_model.pkl"))

    @classmethod
    def try_load(cls, models_dir: str = config.MODELS_DIR) -> Optional["ValuationModel"]:
        try:
            return cls(models_dir)
        except Exception:
            return None

    # ------------------------------------------------------ feature building
    def _encode_city(self, city: str, state: str):
        warns = []
        g = self.encoders.get("global_mean", 8_000_000.0)
        cmap, smap = self.encoders.get("city_map", {}), self.encoders.get("state_map", {})
        c = cmap.get(city)
        if c is None:
            c = g
            warns.append(f"'{city}' is not in the training data; location effect is a national average, so treat this estimate with extra caution.")
        s = smap.get(state)
        if s is None:
            s = g
        return c, s, warns

    def build_row(self, inputs: dict) -> tuple[pd.DataFrame, list]:
        row = {f: 0 for f in self.features}
        city, state = inputs.get("city", "Delhi"), inputs.get("state", "Delhi")
        profile = CITY_PROFILES.get(city, {})
        year = int(inputs.get("construction_year", 2015))
        bed, bath = inputs.get("bedrooms", 2), inputs.get("bathrooms", 2)
        direct = {
            "latitude": inputs.get("latitude", profile.get("lat", 20.59)),
            "longitude": inputs.get("longitude", profile.get("lon", 78.96)),
            "built_up_area_sqft": inputs.get("built_up_area_sqft", 1000),
            "carpet_area_sqft": inputs.get("carpet_area_sqft", 800),
            "plot_area_sqft": inputs.get("plot_area_sqft", 1000),
            "construction_year": year,
            "property_age": max(config.CURRENT_YEAR - year, 0),
            "num_floors": inputs.get("num_floors", 1), "floor_number": inputs.get("floor_number", 0),
            "bedrooms": bed, "bathrooms": bath, "total_rooms": bed + bath,
            "road_width_ft": inputs.get("road_width_ft", 30),
            "city_anchor_psqft": profile.get("avg_price_psqft", np.nan),
            "is_metro": int(profile.get("tier") == "metro"),
        }
        defaults_true = {"parking", "nearby_school", "nearby_hospital", "nearby_market", "lift_available"}
        for b in BOOL_FIELDS:
            direct[b] = int(bool(inputs.get(b, b in defaults_true)))
        direct["amenity_score"] = sum(direct[k] for k in [
            "parking", "nearby_metro", "nearby_school", "nearby_hospital", "nearby_market",
            "lift_available", "swimming_pool", "solar_panels", "rainwater_harvesting", "green_certified"])
        city_enc, state_enc, warns = self._encode_city(city, state)
        direct["city_encoded"], direct["state_encoded"] = city_enc, state_enc
        if pd.isna(direct["city_anchor_psqft"]):
            direct["city_anchor_psqft"] = self.encoders.get("global_mean", 8e6) / 1100.0  # rough national mean psqft
        is_plot = str(inputs.get("property_type", "Apartment")) == "Plot"
        eff_area = direct["plot_area_sqft"] if is_plot else direct["built_up_area_sqft"]
        direct["log_effective_area"] = float(np.log(max(eff_area, 1)))
        direct["log_city_encoded"] = float(np.log(city_enc))
        direct["log_city_anchor"] = float(np.log(direct["city_anchor_psqft"]))
        for k, v in direct.items():
            if k in row:
                row[k] = v
        for prefix, default in [("property_type", "Apartment"), ("property_category", "Residential"),
                                ("facing_direction", "East"), ("construction_quality", "Standard"),
                                ("occupancy", "Owner Occupied"), ("zone", "Central")]:
            col = f"{prefix}_{inputs.get(prefix, default)}"
            if col in row:
                row[col] = 1
        return pd.DataFrame([row])[self.features], warns

    # ---------------------------------------------------------- prediction
    def _q(self, category: str, level: str, ptype: str = "") -> float:
        from conformal import pick
        return pick(self.conformal, category, ptype, level)

    def _interval(self, pred_log: float, category: str, ptype: str = "") -> tuple:
        q90, q80 = self._q(category, "90", ptype), self._q(category, "80", ptype)
        f = lambda v: float(np.expm1(v))
        return f(pred_log - q90), f(pred_log + q90), f(pred_log - q80), f(pred_log + q80)

    def predict(self, inputs: dict) -> Prediction:
        X, warns = self.build_row(inputs)
        pl = float(self.model.predict(X)[0])
        lo90, hi90, lo80, hi80 = self._interval(pl, inputs.get("property_category", "Residential"), inputs.get("property_type", ""))
        return Prediction(float(np.expm1(pl)), lo90, hi90, lo80, hi80, pl, warns)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Each row of df is an `inputs` dict (column names = input keys). Returns df + predictions."""
        out_rows = []
        for rec in df.to_dict("records"):
            try:
                p = self.predict(rec)
                out_rows.append({"predicted_price": round(p.price), "lower_90": round(p.lower_90),
                                 "upper_90": round(p.upper_90), "warning": "; ".join(p.warnings)})
            except Exception as e:  # keep bulk jobs alive; report per-row
                out_rows.append({"predicted_price": np.nan, "lower_90": np.nan, "upper_90": np.nan, "warning": f"error: {e}"})
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)

    # -------------------------------------------------------- explanations
    def explain(self, inputs: dict) -> pd.DataFrame:
        """Exact tree-SHAP contributions (log-price space) via XGBoost, grouped for readability.

        Columns: group, pct_effect (multiplicative effect vs. baseline), rupee_effect (share of the
        gap between baseline and prediction, allocated by log-contribution).
        """
        X, _ = self.build_row(inputs)
        contribs = self.model.get_booster().predict(xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)[0]
        bias, per_feat = contribs[-1], contribs[:-1]
        df = pd.DataFrame({"feature": X.columns, "log_contrib": per_feat})
        df["group"] = df["feature"].map(feature_group)
        g = df.groupby("group")["log_contrib"].sum().reset_index()
        total = g["log_contrib"].sum()
        base_price, price = float(np.expm1(bias)), float(np.expm1(bias + total))
        g["pct_effect"] = (np.exp(g["log_contrib"]) - 1) * 100
        g["rupee_effect"] = (price - base_price) * g["log_contrib"] / total if abs(total) > 1e-9 else 0.0
        g = g.reindex(g["rupee_effect"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        g.attrs["baseline_price"], g.attrs["price"] = base_price, price
        return g

    def top_features(self, inputs: dict, n: int = 10) -> pd.DataFrame:
        X, _ = self.build_row(inputs)
        c = self.model.get_booster().predict(xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)[0][:-1]
        df = pd.DataFrame({"feature": X.columns, "log_contrib": c, "value": X.iloc[0].values})
        return df.reindex(df["log_contrib"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)

    # ------------------------------------------------------------- what-if
    def what_if(self, inputs: dict, improvement_keys: list, costs: Optional[dict] = None) -> pd.DataFrame:
        """Value uplift of each improvement applied alone + all combined."""
        base = self.predict(inputs).price
        rows = []

        def apply(keys):
            mod = dict(inputs)
            for k in keys:
                for field_, val in IMPROVEMENTS[k]["changes"].items():
                    if field_ == "__quality_up__":
                        i = QUALITY_ORDER.index(mod.get("construction_quality", "Standard"))
                        mod["construction_quality"] = QUALITY_ORDER[min(i + 1, len(QUALITY_ORDER) - 1)]
                    else:
                        mod[field_] = val
            return self.predict(mod).price

        for k in improvement_keys:
            if k == "parking" and inputs.get("parking"):
                continue
            already = [f for f in IMPROVEMENTS[k]["changes"] if f in inputs and inputs[f] is True]
            if already and len(already) == len(IMPROVEMENTS[k]["changes"]):
                continue
            new = apply([k])
            default_cost = IMPROVEMENTS[k]["cost"]
            if default_cost is None:  # area-based placeholder
                default_cost = IMPROVEMENTS[k].get("cost_per_sqft", 0) * inputs.get("built_up_area_sqft", 0)
            cost = (costs or {}).get(k, default_cost)
            rows.append({"key": k, "improvement": IMPROVEMENTS[k]["label"], "value_uplift": new - base,
                         "uplift_pct": (new / base - 1) * 100, "cost": cost,
                         "net_gain": new - base - cost,
                         "roi_pct": ((new - base - cost) / cost * 100) if cost > 0 else np.nan})
        out = pd.DataFrame(rows)
        if len(out) > 1:
            comb = apply(list(out["key"]))
            tot_cost = float(out["cost"].sum())
            out.loc[len(out)] = {"key": "__all__", "improvement": "All selected combined", "value_uplift": comb - base,
                                 "uplift_pct": (comb / base - 1) * 100, "cost": tot_cost,
                                 "net_gain": comb - base - tot_cost,
                                 "roi_pct": ((comb - base - tot_cost) / tot_cost * 100) if tot_cost > 0 else np.nan}
        return out

    def sensitivity(self, inputs: dict, field_name: str, values) -> pd.DataFrame:
        rows = []
        for v in values:
            mod = dict(inputs)
            mod[field_name] = v
            if field_name == "built_up_area_sqft":
                mod["carpet_area_sqft"] = v * inputs.get("carpet_area_sqft", v * 0.85) / max(inputs.get("built_up_area_sqft", v), 1)
            rows.append({field_name: v, "price": self.predict(mod).price})
        return pd.DataFrame(rows)
