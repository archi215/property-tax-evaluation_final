"""views/trust.py -- model health, the confidence lab, and how it works."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import theme
from ui_common import get_market, get_model, require_model
from utils import fmt_compact


@st.cache_data(show_spinner=False)
def scored() -> pd.DataFrame:
    vm, mk = get_model(), get_market()
    out = mk[["city", "property_type", "split", "is_metro"]].copy()
    out["log_actual"], out["log_pred"] = mk["price"].values, vm.model.predict(mk[vm.features])
    out["actual"], out["pred"] = np.expm1(out["log_actual"]), np.expm1(out["log_pred"])
    return out


def _ext() -> dict:
    p = os.path.join(config.OUTPUTS_DIR, "extended_results.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _q(res: np.ndarray, conf: float) -> float:
    n = len(res)
    k = min(int(math.ceil((n + 1) * conf)), n)
    return float(np.sort(res)[k - 1])


# ───────────────────────────── model health ─────────────────────────────
def render_health():
    vm = require_model()
    meta, tm = vm.meta, vm.meta.get("test_metrics", {})
    theme.hero("Model health", "How accurate the valuation model is, where it struggles, and what the audit found.",
               [f"Trained {meta.get('trained_at', '')[:10]}", f"XGBoost {meta.get('xgboost_version', '')}", f"{len(meta.get('features', []))} features"])
    k = st.columns(4)
    with k[0]: theme.kpi("R² on unseen data", f"{tm.get('r2', 0):.3f}")
    with k[1]: theme.kpi("Typical error", f"{tm.get('median_ape', 0):.1f}%", f"average {tm.get('mape', 0):.1f}%")
    with k[2]: theme.kpi("Average miss", fmt_compact(tm.get("mae", 0)))
    cov = meta.get("empirical_coverage_test", {})
    with k[3]: theme.kpi("Range reliability", f"{cov.get('90', 0):.1%}", "of outcomes inside the 90% range", "survey")
    theme.note("The dataset is synthetic and built from a multiplicative formula, so these figures show the pipeline works. They do not promise this accuracy on real transactions.")
    t_acc, t_seg, t_mod, t_aud = st.tabs(["Accuracy", "Where it struggles", "Model comparison", "Audit"])
    sc = scored()
    te = sc[sc["split"] == "test"]
    with t_acc:
        s = te.sample(min(len(te), 700), random_state=1)
        fig = go.Figure()
        fig.add_scattergl(x=s["actual"], y=s["pred"], mode="markers", marker=dict(color="rgba(43,58,140,.45)", size=6),
                          customdata=s[["city", "property_type"]].values, hovertemplate="%{customdata[0]}, %{customdata[1]}<br>Actual %{x:,.0f}<br>Predicted %{y:,.0f}<extra></extra>", name="Properties")
        lo, hi = float(s["actual"].min()), float(s["actual"].max())
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=theme.STAMP, dash="dash"), name="Perfect")
        logscale = st.toggle("Log scale", True, key="h_log")
        fig.update_layout(title="Actual against predicted price (unseen test properties)", xaxis_title="Actual", yaxis_title="Predicted")
        if logscale:
            fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
        theme.show(theme.style(fig, 420), key="h_sc")
        err = (te["pred"] / te["actual"] - 1) * 100
        fig = go.Figure(go.Histogram(x=err.clip(-60, 80), nbinsx=60, marker_color=theme.ACCENT))
        fig.update_layout(title="How far off the estimates are", xaxis_title="Predicted minus actual, %", yaxis_title="Properties")
        theme.show(theme.style(fig, 300, legend=False), key="h_err")
    with t_seg:
        te = te.assign(ape=(te["pred"] / te["actual"] - 1).abs() * 100)
        g = te.groupby("property_type")["ape"].agg(["mean", "size"]).sort_values("mean")
        fig = go.Figure(go.Bar(y=g.index, x=g["mean"], orientation="h", marker_color=[theme.STAMP if v > 15 else theme.ACCENT for v in g["mean"]],
                               customdata=g["size"], hovertemplate="%{y}: %{x:.1f}% average error<br>%{customdata} test rows<extra></extra>"))
        fig.update_layout(title="Average error by property type", xaxis_title="%")
        theme.show(theme.style(fig, 320, legend=False), key="h_seg")
        cc = _ext().get("cov_cond", {}).get("ptype")
        if cc:
            d = pd.DataFrame(cc).T.reset_index().rename(columns={"index": "type", "mean": "coverage"}).sort_values("coverage")
            fig = go.Figure(go.Bar(x=d["type"], y=d["coverage"] * 100, marker_color=[theme.STAMP if v < 85 else theme.SURVEY for v in d["coverage"] * 100],
                                   customdata=d["count"], hovertemplate="%{x}: %{y:.1f}%<br>%{customdata} test rows<extra></extra>"))
            fig.add_hline(y=90, line_dash="dash", annotation_text="Target 90%")
            fig.update_layout(title="How often the 90% range holds, by property type", yaxis_title="%")
            theme.show(theme.style(fig, 320, legend=False), key="h_cov")
            st.caption("Warehouses fall well short because there are few of them to calibrate on. A reminder that a range can be reliable overall and wrong for a small group.")
    with t_mod:
        p = os.path.join(config.OUTPUTS_DIR, "model_comparison_results.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0)
            metric = st.segmented_control("Compare on", ["R² Score", "MAPE (%)", "MAE (Rs)"], default="MAPE (%)", key="h_metric") or "MAPE (%)"
            d = df[df["Model"] != "Baseline (mean)"].sort_values(metric, ascending=metric != "R² Score")
            fig = go.Figure(go.Bar(x=d["Model"], y=d[metric], marker_color=[theme.STAMP if m == "XGBoost" else theme.ACCENT for m in d["Model"]],
                                   hovertemplate="%{x}: %{y:,.3f}<extra></extra>"))
            fig.update_layout(title=f"{metric} on the test set (production model in red)", xaxis_tickangle=-40)
            theme.show(theme.style(fig, 400, legend=False), key="h_models")
            st.dataframe(df, width="stretch")
            theme.note("A plain ridge regression with log-area features edges out XGBoost here because the synthetic prices are log-linear by construction. "
                       "XGBoost stays in production for its exact explanations and its handling of non-linear real-world effects.")
        else:
            st.info("Run `python model_comparison.py` to create the comparison.")
    with t_aud:
        ext = _ext()
        if not ext:
            st.info("Run `python extended_experiments.py` to create the audit results.")
        else:
            sd = ext["seeds"]
            a, b, c = st.columns(3)
            with a: theme.kpi("R² over 5 splits", f"{sd['mean']['xgb_r2']:.3f} ± {sd['std']['xgb_r2']:.3f}", "XGBoost")
            with b: theme.kpi("Error over 5 splits", f"{sd['mean']['xgb_mape']:.1f}% ± {sd['std']['xgb_mape']:.1f}", "XGBoost")
            with c: theme.kpi("Ridge, same splits", f"{sd['mean']['ridge_mape']:.1f}% ± {sd['std']['ridge_mape']:.1f}", "average error")
            lk = pd.DataFrame([{"Experiment": "Production model", "R²": ext["xgb"]["r2"], "Error %": ext["xgb"]["mape"]},
                               {"Experiment": "Circle rate added as a feature (leak)", "R²": ext["leak_circle"]["r2"], "Error %": ext["leak_circle"]["mape"]},
                               {"Experiment": "City means from all rows", "R²": ext["leak_te"]["r2"], "Error %": ext["leak_te"]["mape"]},
                               {"Experiment": "Stopped early on the test set", "R²": ext["es_on_test"]["r2"], "Error %": ext["es_on_test"]["mape"]}])
            fig = go.Figure(go.Bar(y=lk["Experiment"], x=lk["Error %"], orientation="h", marker_color=[theme.ACCENT, theme.STAMP, theme.ACCENT, theme.ACCENT],
                                   hovertemplate="%{y}<br>%{x:.2f}% average error<extra></extra>"))
            fig.add_vline(x=7.18, line_dash="dash", annotation_text="Noise floor 7.2%")
            fig.update_layout(title="What leakage does to the headline error", xaxis_title="Average error, %")
            theme.show(theme.style(fig, 300, legend=False), key="h_leak")
            lc = pd.DataFrame(ext["learning_curve"]).T
            fig = go.Figure()
            fig.add_scatter(x=lc["n"], y=lc["xgb_mape"], name="XGBoost", mode="lines+markers", line=dict(color=theme.ACCENT, width=3))
            fig.add_scatter(x=lc["n"], y=lc["ridge_mape"], name="Ridge", mode="lines+markers", line=dict(color=theme.SURVEY, width=3))
            fig.update_layout(title="How much data each model needs", xaxis_title="Training rows", yaxis_title="Average error, %")
            theme.show(theme.style(fig, 320), key="h_lc")


# ───────────────────────────── confidence lab ─────────────────────────────
def render_conformal():
    theme.hero("Confidence lab", "Choose how sure you want to be. The range widens, and we check it against properties the model has never seen.")
    require_model()
    sc = scored()
    cal, te = sc[sc["split"] == "calib"], sc[sc["split"] == "test"]
    res_cal = (cal["log_actual"] - cal["log_pred"]).abs().values
    res_te = (te["log_actual"] - te["log_pred"]).abs().values
    c1, c2 = st.columns([3, 2])
    conf = c1.slider("How sure do you want to be?", 50, 99, 90, format="%d%%", key="c_conf") / 100
    fixed = c2.toggle("Compare with a fixed ±10% band", False, key="c_fixed")
    q = _q(res_cal, conf)
    cover = float(np.mean(res_te <= q))
    up, down = math.exp(q) - 1, 1 - math.exp(-q)
    k = st.columns(4)
    with k[0]: theme.kpi("Range around the estimate", f"−{down:.0%} to +{up:.0%}")
    with k[1]: theme.kpi("You asked for", f"{conf:.0%}")
    with k[2]: theme.kpi("Held on unseen properties", f"{cover:.1%}", "", "survey" if abs(cover - conf) < 0.04 else "stamp")
    if fixed:
        fc = float(np.mean(np.abs(te["actual"] / te["pred"] - 1) <= 0.10))
        with k[3]: theme.kpi("A fixed ±10% band holds", f"{fc:.1%}", "far less than it claims", "stamp")
    s = te.sort_values("pred")
    ratio = (s["actual"] / s["pred"]).values
    inside = (ratio >= math.exp(-q)) & (ratio <= math.exp(q))
    fig = go.Figure()
    fig.add_scattergl(x=np.arange(len(s))[inside], y=ratio[inside], mode="markers", marker=dict(color="rgba(47,125,107,.55)", size=5), name="Inside the range", hoverinfo="skip")
    fig.add_scattergl(x=np.arange(len(s))[~inside], y=ratio[~inside], mode="markers", marker=dict(color=theme.STAMP, size=6), name="Outside", hoverinfo="skip")
    fig.add_hrect(y0=math.exp(-q), y1=math.exp(q), fillcolor="rgba(43,58,140,.10)", line_width=0)
    if fixed:
        fig.add_hline(y=0.9, line_dash="dash", line_color=theme.GOLD); fig.add_hline(y=1.1, line_dash="dash", line_color=theme.GOLD)
    fig.add_hline(y=1, line_color="rgba(128,128,128,.6)")
    fig.update_layout(title="Actual price divided by the estimate, for every unseen property (cheapest to dearest)", xaxis_title="Properties, sorted by estimated price",
                      yaxis_title="Actual ÷ estimate")
    theme.show(theme.style(fig, 400), key="c_scatter")
    confs = np.arange(50, 100)
    covs = [float(np.mean(res_te <= _q(res_cal, c / 100))) * 100 for c in confs]
    fig = go.Figure()
    fig.add_scatter(x=confs, y=confs, name="Perfect", line=dict(color="rgba(128,128,128,.7)", dash="dash"))
    fig.add_scatter(x=confs, y=covs, name="Measured", line=dict(color=theme.ACCENT, width=3))
    fig.add_scatter(x=[conf * 100], y=[cover * 100], name="Your choice", mode="markers", marker=dict(size=13, color=theme.STAMP))
    fig.update_layout(title="Does the range hold as often as it claims?", xaxis_title="Confidence asked for, %", yaxis_title="Share of properties inside the range, %")
    theme.show(theme.style(fig, 320), key="c_curve")
    theme.learn("How this works", "The range is built with **split-conformal prediction**. Take properties the model never trained on, measure how far its estimates "
                "miss in log-price terms, and pick the miss size that covers the share you asked for. That miss size becomes the half-width of the range for every new property. "
                "Because the errors here are proportional (about the same percentage whatever the price), a range that is a fixed percentage of the estimate fits well.")


# ───────────────────────────── how it works ─────────────────────────────
SECTIONS = {
    "The data": """The dataset is **synthetic but anchored to real market levels**. City price levels come from public 2025-26 reports (Knight Frank, Magicbricks, 99acres, Square Yards, NoBroker and Statista).
Individual properties are generated from those anchors with realistic effects: better quality and metro access raise the price, age lowers it, and every record carries random noise.
No row is a real sale. That is why accuracy numbers in this app describe the pipeline, not the market.""",
    "The model": """An **XGBoost** model predicts the logarithm of the price. Working in logs suits prices because most effects are proportional: a better build adds a percentage, not a fixed amount.
The model uses 65 features: size, location encodings, age, quality, type, amenities and surroundings. Columns that would give the answer away (circle rate and government valuation) are deliberately left out, and city averages are learned from training rows only.""",
    "The range": """Each estimate comes with a **90% range and an 80% range**, calibrated on properties the model never trained on. On unseen test properties the 90% range held 91.0% of the time and the 80% range 80.0%.
Try it yourself in the Confidence lab. Warning: for small groups such as warehouses the range does not hold as often, and the app says so.""",
    "The explanation": """Every estimate can be taken apart with **SHAP**, an exact method for tree models. The Valuation Studio groups the contributions into location, size, quality, age, type, amenities, surroundings and a few others, and shows them as a waterfall from an average property to yours.
These are properties of the model, not causes in the real market.""",
    "Taxes and loans": """The tax engine reads rule files for **Delhi, Maharashtra (Mumbai and Pune) and Karnataka (Bengaluru)**. Every other state uses a generic estimate and the app tells you when it does.
Capital gains use 12.5% without indexation, or 20% with indexation for property bought before 23 July 2024. Loan tax breaks (Section 24(b) and 80C) apply to the old regime only. All of it is an estimate, not advice for filing.""",
    "Limits": """- The data is synthetic, with no dates, no neighbourhood detail and a simple multiplicative price formula.
- On this data a plain linear model with log features is as accurate as XGBoost. Real markets will differ.
- The 90% range is reliable overall, not for every small segment.
- Tax rules change and were reconstructed from public guides. Check your municipal corporation before relying on any figure.""",
}


def render_how():
    theme.hero("How it works", "Six short topics, in plain language.")
    pick = st.segmented_control("Topic", list(SECTIONS), default="The data", key="how_topic") or "The data"
    st.markdown(SECTIONS[pick])
    if pick == "The model":
        shares = {"Size and layout": 36.0, "Location": 29.0, "Construction quality": 16.9, "Age": 6.8, "Property type": 6.3, "Surroundings": 3.1, "Amenities": 1.4, "Facing and occupancy": 0.4}
        fig = go.Figure(go.Bar(y=list(shares)[::-1], x=list(shares.values())[::-1], orientation="h", marker_color=theme.ACCENT, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.update_layout(title="What the model pays most attention to (share of attribution)", xaxis_title="%")
        theme.show(theme.style(fig, 340, legend=False), key="how_shap")
        st.caption("Facing direction and occupancy do not enter the price formula, so their 0.4% shows how little the model reads into noise.")
