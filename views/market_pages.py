"""views/market_pages.py -- interactive market map and city face-off."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme
from dataset_loader import CITY_PROFILES
from tax_engine import PropertyInput, calculate_taxes
from ui_common import PROPERTY_CATEGORY_MAP, PROPERTY_TYPES, get_market, require_model, tax_rules_banner
from utils import fmt_compact, fmt_inr


def render_map():
    theme.hero("Market map", "Every bubble is a city. Click one to look inside it.", ["90 cities", "Median prices from the training dataset"])
    mk = get_market()
    if mk.empty:
        st.warning("No dataset found. Run `python run_pipeline.py` first.")
        return
    c1, c2, c3 = st.columns([3, 2, 3])
    types = c1.pills("Property types", sorted(mk["property_type"].unique()), selection_mode="multi", default=["Apartment"], key="m_types") or ["Apartment"]
    metric = c2.segmented_control("Colour by", ["₹ per sq ft", "Median price", "Listings"], default="₹ per sq ft", key="m_metric") or "₹ per sq ft"
    states = c3.multiselect("Limit to states", sorted(mk["state"].unique()), key="m_states", placeholder="All states")
    f = mk[mk["property_type"].isin(types)]
    if states:
        f = f[f["state"].isin(states)]
    agg = f.groupby("city").agg(psqft=("price_per_sqft", "median"), price=("market_price", "median"), n=("market_price", "size")).query("n >= 5").reset_index()
    if agg.empty:
        st.info("Nothing matches these filters.")
        return
    agg["lat"] = agg["city"].map(lambda x: CITY_PROFILES.get(x, {}).get("lat"))
    agg["lon"] = agg["city"].map(lambda x: CITY_PROFILES.get(x, {}).get("lon"))
    agg = agg.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    col = {"₹ per sq ft": "psqft", "Median price": "price", "Listings": "n"}[metric]
    fig = go.Figure(go.Scattermap(lat=agg["lat"], lon=agg["lon"], mode="markers", customdata=agg[["city", "psqft", "price", "n"]].values,
                                  marker=dict(size=agg["n"], sizemode="area", sizeref=2 * agg["n"].max() / 34 ** 2, sizemin=7, color=agg[col],
                                              colorscale="Viridis", showscale=True, colorbar=dict(title=metric, tickformat="~s"), opacity=0.85),
                                  hovertemplate="<b>%{customdata[0]}</b><br>₹%{customdata[1]:,.0f} per sq ft<br>Median price ₹%{customdata[2]:,.0f}<br>%{customdata[3]} listings<extra></extra>"))
    fig.update_layout(map=dict(style="carto-positron", center=dict(lat=22.6, lon=79.0), zoom=3.5), margin=dict(l=0, r=0, t=0, b=0), height=520,
                      paper_bgcolor="rgba(0,0,0,0)")
    ev = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key="m_map", width="stretch",
                         config={"displaylogo": False, "scrollZoom": True})
    picked = None
    try:
        pts = ev.selection.points if ev and ev.selection else []
        if pts:
            picked = pts[0]["customdata"][0]
    except Exception:
        picked = None
    cities = sorted(agg["city"])
    if picked in cities:
        st.session_state["m_city"] = picked
    st.divider()
    left, right = st.columns([2, 5], gap="large")
    with left:
        city = st.selectbox("Look inside a city", cities, index=cities.index(st.session_state.get("m_city", cities[0])) if st.session_state.get("m_city", cities[0]) in cities else 0, key="m_city")
        row = agg[agg["city"] == city].iloc[0]
        theme.kpi("Median per sq ft", fmt_inr(row["psqft"]), CITY_PROFILES.get(city, {}).get("state", ""))
        theme.kpi("Median price", fmt_compact(row["price"]), f"{int(row['n'])} listings", "stamp")
        anchor = CITY_PROFILES.get(city, {}).get("avg_price_psqft")
        if anchor:
            theme.kpi("Public market anchor", fmt_inr(anchor), "per sq ft, from published 2025-26 reports", "survey")
    with right:
        cf = mk[mk["city"] == city]
        fig = go.Figure()
        for i, t in enumerate(sorted(cf["property_type"].unique())):
            fig.add_box(y=cf[cf["property_type"] == t]["price_per_sqft"], name=t, boxpoints=False, marker_color=theme.SEQ[i % len(theme.SEQ)])
        fig.update_layout(title=f"Price per sq ft in {city}, by property type", yaxis_title="₹ per sq ft")
        theme.show(theme.style(fig, 340, legend=False), key="m_box")
    top = agg.sort_values(col, ascending=False).head(15).iloc[::-1]
    fig = go.Figure(go.Bar(y=top["city"], x=top[col], orientation="h", marker_color=theme.ACCENT, hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
    fig.update_layout(title=f"Top 15 cities by {metric.lower()}")
    fig.update_xaxes(tickformat="~s")
    theme.show(theme.style(fig, 420, legend=False), key="m_rank")


def render_faceoff():
    theme.hero("City face-off", "Put the same property in up to four cities and compare the value, the charges and the yearly tax.")
    tax_rules_banner()
    vm = require_model()
    cities = st.multiselect("Cities", list(CITY_PROFILES), default=["Delhi", "Mumbai", "Bengaluru"], max_selections=4, key="f_cities")
    c = st.columns(4)
    ptype = c[0].pills("Type", PROPERTY_TYPES, default="Apartment", key="f_type") or "Apartment"
    area = c[1].slider("Area (sq ft)", 300, 5000, 1200, 50, key="f_area")
    year = c[2].slider("Built in", 1960, 2025, 2015, key="f_year")
    quality = c[3].select_slider("Quality", ["Basic", "Standard", "Premium", "Luxury"], value="Standard", key="f_q")
    if len(cities) < 2:
        st.info("Choose at least two cities.")
        return
    rows = []
    cat = PROPERTY_CATEGORY_MAP[ptype]
    for city in cities:
        prof = CITY_PROFILES[city]
        inp = {"state": prof["state"], "city": city, "property_type": ptype, "property_category": cat, "built_up_area_sqft": area, "carpet_area_sqft": area * 0.85,
               "plot_area_sqft": area, "construction_year": year, "construction_quality": quality, "bedrooms": 3 if cat == "Residential" else 0, "bathrooms": 2, "floor_number": 2}
        p = vm.predict(inp)
        t = calculate_taxes(PropertyInput(state=prof["state"], city=city, property_type=ptype, property_category=cat, built_up_area_sqft=area,
                                          construction_year=year, market_value=p.price, construction_quality=quality))
        rows.append({"City": city, "Value": p.price, "Low 90%": p.lower_90, "High 90%": p.upper_90, "Per sq ft": p.price / area, "Annual property tax": t.annual_property_tax,
                     "Stamp duty": t.stamp_duty, "Registration": t.registration_charges, "Total to acquire": t.total_purchase_cost,
                     "Stamp and registration, % of value": (t.stamp_duty + t.registration_charges) / p.price * 100})
    df = pd.DataFrame(rows)
    a, b = st.columns(2, gap="large")
    with a:
        fig = go.Figure()
        fig.add_bar(x=df["City"], y=df["Value"], name="Value", marker_color=theme.ACCENT,
                    error_y=dict(type="data", symmetric=False, array=df["High 90%"] - df["Value"], arrayminus=df["Value"] - df["Low 90%"]))
        fig.add_bar(x=df["City"], y=df["Total to acquire"] - df["Value"], name="Charges on top", marker_color=theme.STAMP)
        fig.update_layout(barmode="stack", title="Value with its 90% range, plus charges")
        fig.update_yaxes(tickformat="~s")
        theme.show(theme.style(fig, 360), key="f_val")
    with b:
        fig = go.Figure()
        fig.add_bar(x=df["City"], y=df["Stamp and registration, % of value"], marker_color=theme.GOLD, name="Stamp duty and registration")
        fig.add_bar(x=df["City"], y=df["Annual property tax"] / df["Value"] * 100, marker_color=theme.SURVEY, name="Yearly property tax")
        fig.update_layout(barmode="group", title="Charges as a percentage of value", yaxis_title="% of value")
        theme.show(theme.style(fig, 360), key="f_pct")
    st.dataframe(df.style.format({c_: fmt_inr for c_ in df.columns if c_ not in ("City", "Stamp and registration, % of value")} | {"Stamp and registration, % of value": "{:.2f}%"}),
                 width="stretch", hide_index=True)
