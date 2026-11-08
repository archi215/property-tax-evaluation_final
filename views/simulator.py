"""views/simulator.py -- what an upgrade is worth, and how value moves with each input."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import config
import theme
from ui_common import CONSTRUCTION_QUALITY, require_model
from utils import fmt_compact, fmt_inr
from valuation import IMPROVEMENTS

DEFAULT = {"city": "Delhi", "state": "Delhi", "property_type": "Apartment", "property_category": "Residential", "built_up_area_sqft": 1200,
           "carpet_area_sqft": 1020, "plot_area_sqft": 1500, "construction_year": 2012, "bedrooms": 3, "bathrooms": 2, "floor_number": 2,
           "construction_quality": "Standard", "occupancy": "Owner Occupied"}


def render():
    theme.hero("Value-add simulator", "Pick upgrades, enter what they would cost, and see what the model thinks they add to the value.",
               ["Uses the property from the Valuation Studio"])
    vm = require_model()
    inp = st.session_state.get("studio_inputs")
    if not inp:
        st.info("No property chosen yet, so this uses a standard Delhi apartment. Set your own in the Valuation Studio and come back.")
        inp = DEFAULT
    base = vm.predict(inp).price
    st.markdown(f"Starting from a **{inp['built_up_area_sqft']:,} sq ft {inp['property_type'].lower()} in {inp['city']}** worth **{fmt_inr(base)}**.")

    st.subheader("Choose upgrades")
    keys = []
    costs = {}
    cols = st.columns(len(IMPROVEMENTS))
    for col, (k, d) in zip(cols, IMPROVEMENTS.items()):
        with col:
            on = st.toggle(d["label"], value=k in ("solar_panels", "quality_up"), key=f"sim_on_{k}")
            default = d["cost"] if d["cost"] is not None else d.get("cost_per_sqft", 0) * inp["built_up_area_sqft"]
            costs[k] = st.number_input("Cost (₹)", 0, value=int(default), step=10000, key=f"sim_c_{k}", disabled=not on, help="A placeholder. Replace it with a real quote.")
            if on:
                keys.append(k)
    if keys:
        out = vm.what_if(inp, keys, {k: costs[k] for k in keys})
        if out.empty:
            st.info("The property already has what you selected.")
        else:
            single = out[out["key"] != "__all__"]
            fig = go.Figure()
            fig.add_bar(y=single["improvement"], x=single["value_uplift"], orientation="h", name="Value added", marker_color=theme.SURVEY,
                        customdata=single["roi_pct"], hovertemplate="%{y}<br>Adds %{x:,.0f}<br>Return %{customdata:,.0f}% on cost<extra></extra>")
            fig.add_bar(y=single["improvement"], x=single["cost"], orientation="h", name="Cost", marker_color=theme.STAMP)
            fig.update_layout(barmode="group", title="What each upgrade adds against what it costs")
            fig.update_xaxes(tickformat="~s")
            theme.show(theme.style(fig, 140 + 80 * len(single)), key="sim_bar")
            tot = out[out["key"] == "__all__"]
            row = tot.iloc[0] if len(tot) else single.iloc[0]
            a, b, c = st.columns(3)
            with a: theme.kpi("Added value", fmt_compact(row["value_uplift"]), f"{row['uplift_pct']:+.1f}% on the price", "survey")
            with b: theme.kpi("Total cost", fmt_compact(row["cost"]))
            with c: theme.kpi("Net gain", fmt_compact(row["net_gain"]), "after the cost", "stamp" if row["net_gain"] < 0 else "survey")
            theme.note("The model learned from a synthetic dataset in which every amenity adds a flat premium of about 2%. "
                       "Treat the amounts as a demonstration of the tool, not evidence of real resale gains.")
    else:
        st.info("Switch on at least one upgrade.")

    st.divider()
    st.subheader("How value moves")
    a, b = st.columns(2, gap="large")
    with a:
        areas = np.linspace(max(inp["built_up_area_sqft"] * 0.4, 200), inp["built_up_area_sqft"] * 2, 14)
        fig = go.Figure()
        for q in CONSTRUCTION_QUALITY:
            s = vm.sensitivity({**inp, "construction_quality": q}, "plot_area_sqft" if inp["property_type"] == "Plot" else "built_up_area_sqft", areas)
            fig.add_scatter(x=s.iloc[:, 0], y=s["price"], name=q, mode="lines+markers", line=dict(width=3 if q == inp.get("construction_quality", "Standard") else 1.5))
        fig.update_layout(title="Value against size, by construction quality", xaxis_title="Area (sq ft)")
        fig.update_yaxes(tickformat="~s")
        theme.show(theme.style(fig, 360), key="sim_area")
    with b:
        yrs = np.arange(1980, config.CURRENT_YEAR + 1, 3)
        s = vm.sensitivity(inp, "construction_year", yrs)
        fig = go.Figure(go.Scatter(x=s["construction_year"], y=s["price"], mode="lines+markers", line=dict(color=theme.ACCENT, width=3)))
        fig.add_vline(x=inp["construction_year"], line_dash="dot", annotation_text="Yours")
        fig.update_layout(title="Value against year of construction", xaxis_title="Built in")
        fig.update_yaxes(tickformat="~s")
        theme.show(theme.style(fig, 360, legend=False), key="sim_year")
