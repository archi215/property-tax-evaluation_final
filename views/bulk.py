"""views/bulk.py -- value a whole spreadsheet of properties."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme
from ui_common import require_model
from utils import fmt_compact

# ═════════════════════════ BULK VALUATION ═══════════════════════════════════
TEMPLATE = pd.DataFrame([
    {"state": "Delhi", "city": "Delhi", "property_type": "Apartment", "property_category": "Residential", "built_up_area_sqft": 1200,
     "carpet_area_sqft": 1000, "plot_area_sqft": 1500, "construction_year": 2015, "bedrooms": 3, "bathrooms": 2, "floor_number": 2,
     "construction_quality": "Standard", "occupancy": "Owner Occupied", "nearby_metro": 1, "parking": 1, "lift_available": 1},
    {"state": "Maharashtra", "city": "Pune", "property_type": "Villa", "property_category": "Residential", "built_up_area_sqft": 2800,
     "carpet_area_sqft": 2400, "plot_area_sqft": 3600, "construction_year": 2021, "bedrooms": 4, "bathrooms": 4, "floor_number": 0,
     "construction_quality": "Premium", "occupancy": "Owner Occupied", "nearby_metro": 0, "parking": 1, "lift_available": 0},
])
_BOOLS = ["parking", "nearby_metro", "nearby_school", "nearby_hospital", "nearby_market", "nearby_airport", "nearby_railway_station",
          "corner_plot", "lift_available", "swimming_pool", "solar_panels", "rainwater_harvesting", "green_certified"]
_REQUIRED = ["city", "state", "property_type", "construction_year"]
MAX_ROWS = 5000


def _coerce_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).strip().lower() in ("1", "true", "yes", "y", "t", "1.0"))


def render_bulk():
    theme.hero("Bulk valuation", f"Upload a spreadsheet of properties and get a value and a 90% range for each (up to {MAX_ROWS:,} rows).")
    vm = require_model()
    st.download_button("Download CSV template", TEMPLATE.to_csv(index=False).encode(), "bulk_template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        st.info("Required columns: " + ", ".join(_REQUIRED) + " + at least `built_up_area_sqft` (or `plot_area_sqft` for plots). "
                "Optional: carpet_area_sqft, bedrooms, bathrooms, floor_number, construction_quality, occupancy, and 0/1 amenity flags.")
        return
    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing or not ({"built_up_area_sqft", "plot_area_sqft"} & set(df.columns)):
        st.error(f"Missing required column(s): {missing or 'built_up_area_sqft / plot_area_sqft'}")
        return
    if len(df) > MAX_ROWS:
        st.warning(f"Only the first {MAX_ROWS:,} rows will be processed.")
        df = df.head(MAX_ROWS)
    df = df.copy()
    for b in _BOOLS:
        if b in df.columns:
            df[b] = _coerce_bool(df[b])
    if "property_category" not in df.columns:
        from ui_common import PROPERTY_CATEGORY_MAP
        df["property_category"] = df["property_type"].map(PROPERTY_CATEGORY_MAP).fillna("Residential")
    if "built_up_area_sqft" not in df.columns:
        df["built_up_area_sqft"] = df["plot_area_sqft"]
    if "plot_area_sqft" not in df.columns:
        df["plot_area_sqft"] = df["built_up_area_sqft"]
    if "carpet_area_sqft" not in df.columns:
        df["carpet_area_sqft"] = df["built_up_area_sqft"] * 0.85
    with st.spinner(f"Valuing {len(df):,} properties..."):
        out = vm.predict_batch(df)
    ok = out["predicted_price"].notna()
    c = st.columns(3)
    with c[0]: theme.kpi("Valued", f"{ok.sum():,} of {len(out):,}")
    with c[1]: theme.kpi("Portfolio value", fmt_compact(out["predicted_price"].sum()), "", "stamp")
    with c[2]: theme.kpi("Rows with warnings", f"{(out['warning'] != '').sum():,}")
    fig = go.Figure(go.Histogram(x=out["predicted_price"].dropna(), nbinsx=30, marker_color=theme.ACCENT))
    fig.update_layout(title="Spread of estimated values", xaxis_title="Estimated value", yaxis_title="Properties")
    fig.update_xaxes(tickformat="~s")
    theme.show(theme.style(fig, 280, legend=False), key="b_hist")
    st.dataframe(out, width="stretch", hide_index=True)
    st.download_button("Download results CSV", out.to_csv(index=False).encode(), "bulk_valuations.csv", "text/csv")
