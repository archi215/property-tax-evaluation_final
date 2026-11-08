"""views/data_pages.py -- saved valuations and tax-rule coverage."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import storage
import theme
from ui_common import get_model, get_tax_rule_files
from utils import fmt_inr


def render_history():
    theme.hero("History", "Valuations you saved from the Studio. They stay here between sessions.")
    hist = storage.load_history()
    if hist.empty:
        st.info("Nothing saved yet. Open the Valuation Studio and choose Save to history.")
        st.page_link(theme.PAGES["studio"], label="Go to the Valuation Studio", icon=":material/arrow_forward:")
        return
    show = hist.drop(columns=["inputs_json"])
    c = st.columns(3)
    with c[0]: theme.kpi("Saved valuations", f"{len(hist)}")
    with c[1]: theme.kpi("Cities", f"{hist['city'].nunique()}")
    with c[2]: theme.kpi("Latest", str(hist.iloc[0]["timestamp"])[:16].replace("T", " "))
    chart = hist.sort_values("id")
    labels = [f"#{i} {c_}" for i, c_ in zip(chart["id"], chart["city"])]
    fig = go.Figure()
    fig.add_bar(x=labels, y=chart["predicted_price"], marker_color=theme.ACCENT, name="Value",
                error_y=dict(type="data", symmetric=False, array=chart["upper_90"] - chart["predicted_price"], arrayminus=chart["predicted_price"] - chart["lower_90"]))
    fig.update_layout(title="Saved valuations with their 90% range")
    fig.update_yaxes(tickformat="~s")
    theme.show(theme.style(fig, 340, legend=False), key="hist_bar")
    st.dataframe(show.style.format({c_: fmt_inr for c_ in ["predicted_price", "lower_90", "upper_90", "annual_property_tax", "stamp_duty", "total_purchase_cost", "emi"]}),
                 width="stretch", hide_index=True)
    a, b, d = st.columns(3)
    a.download_button("Download CSV", show.to_csv(index=False).encode(), "valuation_history.csv", "text/csv", width="stretch", icon=":material/download:")
    rid = b.selectbox("Entry to delete", hist["id"].tolist(), label_visibility="collapsed")
    if b.button("Delete entry", width="stretch", icon=":material/delete:"):
        storage.delete_valuation(int(rid)); st.rerun()
    if d.button("Clear everything", width="stretch", icon=":material/delete_sweep:"):
        storage.clear_history(); st.rerun()


def render_rules():
    theme.hero("Tax rules and about", "Which rule sets drive the tax figures, and what this project is.")
    rows = [{"File": f, "State": d.get("state", "?"), "System": d.get("system", "?"), "Financial year": d.get("financial_year", "?")}
            for f, d in get_tax_rule_files().items() if f != "_default.json"]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.warning("No tax rule files are loaded. Copy them into the tax_rules folder.")
    st.markdown("States without a dedicated rule file use a generic estimate from `_default.json`, and the app says so wherever it happens.")
    files = get_tax_rule_files()
    if files:
        with st.expander("Read a rule file"):
            st.json(files[st.selectbox("File", list(files))])
    vm = get_model()
    if vm:
        st.write(f"Production model: XGBoost with {len(vm.features)} features.")
    theme.note("Academic and portfolio project. Estimates only. Verify with your municipal corporation and a qualified professional before any financial decision.")
