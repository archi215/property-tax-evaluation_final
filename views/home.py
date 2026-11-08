"""views/home.py -- landing hub organised by topic, with a 10-second valuation."""
from __future__ import annotations

import streamlit as st

import storage
import theme
from dataset_loader import CITY_PROFILES
from ui_common import PROPERTY_TYPES, get_market, require_model
from utils import fmt_compact, fmt_inr


def render():
    vm = require_model()
    tm = vm.meta.get("test_metrics", {})
    cov = vm.meta.get("empirical_coverage_test", {})
    theme.hero("What is it worth, and what will it cost you?",
               "Value an Indian property with an honest range, see what drives the number, then follow it through to stamp duty, "
               "property tax, a home loan and the long run.",
               [f"Test R² {tm.get('r2', 0):.3f}", f"Typical error {tm.get('median_ape', 0):.1f}%",
                f"90% range holds {cov.get('90', 0):.0%} of the time", "Trained on a synthetic, real-anchored dataset"])

    st.subheader("Try a valuation")
    c1, c2, c3 = st.columns([2, 3, 3], gap="large")
    state_city = {c: p["state"] for c, p in CITY_PROFILES.items()}
    city = c1.selectbox("City", sorted(state_city), index=sorted(state_city).index("Pune"), key="h_city")
    ptype = c2.pills("Property type", PROPERTY_TYPES, default="Apartment", key="h_ptype") or "Apartment"
    area = c3.slider("Built-up area (sq ft)", 300, 5000, 1100, 50, key="h_area")
    inp = {"city": city, "state": state_city[city], "property_type": ptype, "built_up_area_sqft": area, "carpet_area_sqft": area * 0.85,
           "plot_area_sqft": area, "construction_year": 2015, "bedrooms": 2 if ptype in ("Apartment", "Villa", "Independent House") else 0, "bathrooms": 2,
           "floor_number": 2, "property_category": {"Plot": "Land", "Office": "Commercial", "Shop": "Commercial", "Warehouse": "Industrial"}.get(ptype, "Residential")}
    p = vm.predict(inp)
    l, r = st.columns([3, 2], gap="large")
    with l:
        theme.headline_price(p.price, f"about {fmt_inr(p.price / area)} per sq ft")
        theme.ruler(p.price, p.lower_90, p.upper_90, p.lower_80, p.upper_80)
    with r:
        st.caption("This quick view assumes a standard 2015 build. The Studio lets you set every detail, the taxes and the loan.")
        if st.button("Open this in the Valuation Studio", type="primary", width="stretch", icon=":material/arrow_forward:"):
            st.session_state.update({"s_state": state_city[city], "s_city": city, "s_ptype": ptype, "s_area": area})
            st.switch_page(theme.PAGES["studio"])

    st.divider()
    st.subheader("Explore by topic")
    topics = [
        ("Value", "Price a property, test what an upgrade is worth, or value a whole portfolio from a spreadsheet.",
         [("studio", "Valuation Studio"), ("simulator", "Value-add simulator"), ("bulk", "Bulk valuation")]),
        ("Loans and affordability", "See what a loan really costs, what you can afford, and whether buying beats renting.",
         [("emi", "EMI lab"), ("afford", "Affordability"), ("rentbuy", "Rent or buy"), ("investor", "Investor view")]),
        ("Markets", "Compare price levels across 90 cities on a map, or put the same home in several cities side by side.",
         [("map", "Market map"), ("faceoff", "City face-off")]),
        ("Trust and method", "Check how accurate the model is, play with the confidence range, and read how it all works.",
         [("health", "Model health"), ("conformal", "Confidence lab"), ("how", "How it works")]),
        ("Your records", "Reopen saved valuations and see which tax rules the app uses.",
         [("history", "History"), ("rules", "Tax rules and about")]),
    ]
    cols = st.columns(3, gap="large")
    for i, (title, body, links) in enumerate(topics):
        with cols[i % 3]:
            theme.topic(title, body)
            for key, label in links:
                st.page_link(theme.PAGES[key], label=label)
            st.write("")

    mk = get_market()
    if not mk.empty:
        st.divider()
        st.subheader("Price levels in the dataset (apartments, median per sq ft)")
        med = mk[mk["property_type"] == "Apartment"].groupby("city")["price_per_sqft"].agg(["median", "size"]).query("size >= 20")["median"].sort_values()
        a, b, c = st.columns(3)
        with a: theme.kpi("Most affordable", f"{fmt_inr(med.iloc[0])}", med.index[0])
        with b: theme.kpi("Typical across cities", f"{fmt_inr(med.median())}", "median of city medians")
        with c: theme.kpi("Most expensive", f"{fmt_inr(med.iloc[-1])}", med.index[-1], "stamp")

    hist = storage.load_history()
    if not hist.empty:
        st.divider()
        st.subheader("Your latest valuations")
        show = hist.head(5)[["timestamp", "city", "property_type", "area_sqft", "predicted_price", "total_purchase_cost"]].rename(columns={
            "timestamp": "When", "city": "City", "property_type": "Type", "area_sqft": "Sq ft", "predicted_price": "Value", "total_purchase_cost": "Total to acquire"})
        st.dataframe(show.style.format({"Value": fmt_inr, "Total to acquire": fmt_inr, "Sq ft": "{:,.0f}"}), width="stretch", hide_index=True)
