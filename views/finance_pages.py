"""views/finance_pages.py -- EMI lab, affordability, rent-or-buy, investor view."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import finance
import theme
from ui_common import PROPERTY_TYPES, get_market
from utils import fmt_compact, fmt_inr


def _prefill():
    res = st.session_state.get("studio_result")
    if res:
        return float(res["pred"].price), float(res["tax"].one_time_total), float(res["tax"].annual_property_tax), True
    return 8_000_000.0, 500_000.0, 10_000.0, False


# ───────────────────────────── EMI lab ─────────────────────────────
def render_emi():
    price0, _, _, linked = _prefill()
    theme.hero("EMI lab", "Drag the sliders. The instalment, the interest and the payoff date change as you go.",
               ["Linked to your Studio valuation"] if linked else None)
    c = st.columns(4)
    price = c[0].number_input("Property price (₹)", 100_000, 2_000_000_000, int(price0), 100_000, key="e_price")
    dp = c[1].slider("Down payment", 0, 90, 20, format="%d%%", key="e_dp")
    rate = c[2].slider("Interest rate", 6.0, 14.0, 8.5, 0.1, format="%.1f%%", key="e_rate")
    years = c[3].slider("Tenure (years)", 5, 30, 20, key="e_years")
    loan = price * (1 - dp / 100)
    emi, total, interest = finance.calculate_mortgage(loan, rate, years)
    k = st.columns(4)
    with k[0]: theme.kpi("Loan", fmt_compact(loan))
    with k[1]: theme.kpi("EMI a month", fmt_inr(emi), "", "stamp")
    with k[2]: theme.kpi("Interest over the loan", fmt_compact(interest), f"{interest / max(loan, 1) * 100:.0f}% of what you borrow")
    with k[3]: theme.kpi("Total repaid", fmt_compact(total))

    a, b = st.columns(2, gap="large")
    with a:
        rates = np.arange(6.0, 14.01, 0.5)
        tenures = [5, 10, 15, 20, 25, 30]
        z = [[finance.emi(loan, r_, t_) for r_ in rates] for t_ in tenures]
        fig = go.Figure(go.Heatmap(z=z, x=[f"{r_:.1f}%" for r_ in rates], y=[f"{t_} yrs" for t_ in tenures], colorscale="Blues", reversescale=False,
                                   hovertemplate="Rate %{x}, %{y}<br>EMI %{z:,.0f}<extra></extra>", colorbar=dict(tickformat="~s")))
        fig.add_scatter(x=[f"{min(rates, key=lambda v: abs(v - rate)):.1f}%"], y=[f"{min(tenures, key=lambda v: abs(v - years))} yrs"],
                        mode="markers", marker=dict(size=14, color=theme.STAMP, symbol="x"), name="Your choice", hoverinfo="skip")
        fig.update_layout(title="EMI for every rate and tenure")
        theme.show(theme.style(fig, 380, legend=False), key="e_heat")
    with b:
        m1, m2 = st.columns(2)
        mp = m1.slider("Extra each month (₹)", 0, 50000, 5000, 1000, key="e_mp")
        ls = m2.slider("One-off in month 12 (₹)", 0, 2_000_000, 0, 50000, key="e_ls")
        imp = finance.prepayment_impact(loan, rate, years, mp, ls, 12)
        sched = finance.amortization_schedule(loan, rate, years, mp, ls, 12)
        base = finance.amortization_schedule(loan, rate, years)
        fig = go.Figure()
        fig.add_scatter(x=base["month"] / 12, y=base["balance"], name="No prepayment", line=dict(color=theme.MUTED, dash="dot"))
        fig.add_scatter(x=sched["month"] / 12, y=sched["balance"], name="With prepayment", line=dict(color=theme.SURVEY, width=3), fill="tozeroy", fillcolor="rgba(47,125,107,.10)")
        fig.update_layout(title="How fast the loan shrinks", xaxis_title="Years", yaxis_title="Still owed")
        fig.update_yaxes(tickformat="~s")
        theme.show(theme.style(fig, 300), key="e_bal")
        st.success(f"Interest saved: {fmt_inr(imp['interest_saved'])}. Done {imp['months_saved']} months sooner.") if (mp or ls) else st.caption("Add a prepayment to see the saving.")


# ───────────────────────────── affordability ─────────────────────────────
def render_afford():
    theme.hero("Affordability", "Start from your income, not from a property. See the most you can borrow, how a rate rise bites, and where that buys you a home.")
    c = st.columns(4)
    inc = c[0].number_input("Monthly take-home (₹)", 0, 10_000_000, 150_000, 10_000, key="a_inc")
    ex = c[1].number_input("Existing EMIs (₹ a month)", 0, 5_000_000, 0, 1_000, key="a_ex")
    foir = c[2].slider("Share of income for EMIs", 30, 65, 45, format="%d%%", key="a_foir", help="Lenders call this FOIR.")
    dp = c[3].slider("Down payment", 0, 90, 20, format="%d%%", key="a_dp")
    c = st.columns(2)
    rate = c[0].slider("Interest rate", 6.0, 14.0, 8.5, 0.1, format="%.1f%%", key="a_rate")
    years = c[1].slider("Tenure (years)", 5, 30, 20, key="a_years")
    a = finance.affordability(inc, ex, rate, years, dp, foir, 2.0)
    k = st.columns(4)
    with k[0]: theme.kpi("Highest comfortable EMI", fmt_inr(a["max_emi"]))
    with k[1]: theme.kpi("Largest loan", fmt_compact(a["max_loan"]))
    with k[2]: theme.kpi("Most you can pay for a home", fmt_compact(a["max_property_price"]), f"with {dp}% down", "stamp")
    with k[3]: theme.kpi("If rates rise 2 points", fmt_compact(a["stressed_max_price"]), "the budget shrinks", "survey")

    l, r = st.columns(2, gap="large")
    with l:
        bumps = [0, 0.5, 1, 1.5, 2, 3, 4]
        vals = [finance.affordability(inc, ex, rate, years, dp, foir, b)["stressed_max_price"] if b else a["max_property_price"] for b in bumps]
        fig = go.Figure(go.Bar(x=[f"+{b:g}" for b in bumps], y=vals, marker_color=[theme.ACCENT] + [theme.STAMP] * (len(bumps) - 1),
                               hovertemplate="Rate up %{x} points<br>Budget %{y:,.0f}<extra></extra>"))
        fig.update_layout(title="Budget as the interest rate rises (points)", xaxis_title="Rate increase")
        fig.update_yaxes(tickformat="~s")
        theme.show(theme.style(fig, 340, legend=False), key="a_stress")
    with r:
        mk = get_market()
        ptype = st.pills("What kind of property?", PROPERTY_TYPES, default="Apartment", key="a_type") or "Apartment"
        area = st.slider("Size you want (sq ft)", 300, 4000, 1000, 50, key="a_area")
    if not mk.empty:
        med = mk[mk["property_type"] == ptype].groupby("city")["price_per_sqft"].agg(["median", "size"]).query("size >= 8")["median"] * area
        med = med.sort_values()
        ok = med <= a["max_property_price"]
        fig = go.Figure(go.Bar(x=med.index, y=med.values, marker_color=[theme.SURVEY if v else "rgba(128,128,128,.45)" for v in ok],
                               hovertemplate="%{x}<br>Typical price %{y:,.0f}<extra></extra>"))
        fig.add_hline(y=a["max_property_price"], line_color=theme.STAMP, line_width=3, annotation_text="Your budget")
        fig.update_layout(title=f"Typical price of a {area:,} sq ft {ptype.lower()} by city", xaxis_tickangle=-60)
        fig.update_yaxes(tickformat="~s")
        theme.show(theme.style(fig, 420, legend=False), key="a_cities")
        st.write(f"In **{int(ok.sum())} of {len(med)}** cities the typical price is within your budget (green).")


# ───────────────────────────── rent or buy ─────────────────────────────
def render_rentbuy():
    price0, close0, tax0, linked = _prefill()
    theme.hero("Rent or buy", "Two futures from the same money: own the home, or rent and invest the difference. Move the assumptions and watch the crossover.",
               ["Linked to your Studio valuation"] if linked else None)
    c = st.columns(4)
    price = c[0].number_input("Property price (₹)", 100_000, 2_000_000_000, int(price0), 100_000, key="r_price")
    dp = c[1].slider("Down payment", 0, 90, 20, format="%d%%", key="r_dp")
    rate = c[2].slider("Loan rate", 6.0, 14.0, 8.5, 0.1, format="%.1f%%", key="r_rate")
    years = c[3].slider("Loan tenure", 5, 30, 20, key="r_years")
    c = st.columns(4)
    rent = c[0].number_input("Rent for a similar home (₹ a month)", 0, 5_000_000, max(int(price * 0.03 / 12 // 100 * 100), 5000), 1000, key="r_rent")
    rg = c[1].slider("Rent grows", 0.0, 12.0, 5.0, 0.5, format="%.1f%% a year", key="r_rg")
    ap = c[2].slider("Property grows", 0.0, 12.0, 6.0, 0.5, format="%.1f%% a year", key="r_ap")
    inv = c[3].slider("Investments earn", 0.0, 15.0, 9.0, 0.5, format="%.1f%% a year", key="r_inv")
    c = st.columns(3)
    horizon = c[0].slider("Years", 3, 30, 15, key="r_h")
    maint = c[1].number_input("Maintenance (₹ a year)", 0, 5_000_000, int(price * 0.005), 5000, key="r_m")
    sell = c[2].slider("Cost of selling", 0.0, 5.0, 2.0, 0.5, format="%.1f%%", key="r_s")
    df = finance.rent_vs_buy(price, close0, dp, rate, years, rent, rg, ap, inv, horizon, maint, tax0, sell)
    be = finance.breakeven_year(df)
    last = df.iloc[-1]
    k = st.columns(3)
    with k[0]: theme.kpi(f"Owning after {horizon} years", fmt_compact(last["buy_net_worth"]), "equity after selling costs")
    with k[1]: theme.kpi(f"Renting after {horizon} years", fmt_compact(last["rent_net_worth"]), "investments")
    with k[2]: theme.kpi("Owning pulls ahead", f"year {be}" if be else "not within the horizon", "", "stamp" if not be else "survey")
    fig = go.Figure()
    fig.add_scatter(x=df["year"], y=df["buy_net_worth"], name="Own", line=dict(color=theme.ACCENT, width=3))
    fig.add_scatter(x=df["year"], y=df["rent_net_worth"], name="Rent and invest", line=dict(color=theme.STAMP, width=3))
    if be:
        fig.add_vline(x=be, line_dash="dot", annotation_text=f"Crossover, year {be}")
    fig.update_layout(title="Net worth under each choice", xaxis_title="Years")
    fig.update_yaxes(tickformat="~s")
    theme.show(theme.style(fig, 360), key="r_lines")

    aps, invs = np.arange(2, 12.1, 2), np.arange(4, 14.1, 2)
    z = [[finance.rent_vs_buy(price, close0, dp, rate, years, rent, rg, a_, i_, horizon, maint, tax0, sell).iloc[-1]["advantage_buy"] for a_ in aps] for i_ in invs]
    fig = go.Figure(go.Heatmap(z=z, x=[f"{a_:g}%" for a_ in aps], y=[f"{i_:g}%" for i_ in invs], colorscale="RdBu", zmid=0,
                               hovertemplate="Property grows %{x}, investments earn %{y}<br>Owning minus renting: %{z:,.0f}<extra></extra>", colorbar=dict(tickformat="~s")))
    fig.update_layout(title=f"Owning minus renting after {horizon} years (blue favours owning)", xaxis_title="Property growth a year", yaxis_title="Investment return a year")
    theme.show(theme.style(fig, 340, legend=False), key="r_heat")
    st.caption("Ignores tax benefits and the non-financial value of owning. It is arithmetic on your assumptions.")


# ───────────────────────────── investor view ─────────────────────────────
def render_investor():
    price0, close0, tax0, linked = _prefill()
    theme.hero("Investor view", "What a rental property earns after vacancy, tax and upkeep, and what the home-loan tax breaks are worth.")
    c = st.columns(3)
    price = c[0].number_input("Property price (₹)", 100_000, 2_000_000_000, int(price0), 100_000, key="i_price")
    rent = c[1].number_input("Rent (₹ a year)", 0, 100_000_000, int(price * 0.03), 10_000, key="i_rent")
    vac = c[2].slider("Empty months", 0.0, 25.0, 5.0, 0.5, format="%.1f%% of the year", key="i_vac")
    maint = st.slider("Upkeep (₹ a year)", 0, int(max(price * 0.03, 50_000)), int(price * 0.004), 5_000, key="i_maint")
    r = finance.rental_roi(price, close0, rent, tax0, maint, vac)
    k = st.columns(4)
    with k[0]: theme.kpi("Gross yield", f"{r['gross_yield_pct']:.2f}%")
    with k[1]: theme.kpi("Net yield on total cost", f"{r['net_yield_pct']:.2f}%", "", "survey")
    with k[2]: theme.kpi("Net income a year", fmt_compact(r["net_annual_income"]))
    with k[3]: theme.kpi("Years to earn it back", "never" if not np.isfinite(r["payback_years"]) else f"{r['payback_years']:.0f}", "", "stamp")
    eff = rent * (1 - vac / 100)
    fig = go.Figure(go.Waterfall(x=["Rent", "Empty months", "Property tax", "Upkeep", "Net income"], measure=["absolute", "relative", "relative", "relative", "total"],
                                 y=[rent, -(rent - eff), -tax0, -maint, 0], text=[fmt_compact(v) for v in (rent, -(rent - eff), -tax0, -maint, r["net_annual_income"])],
                                 textposition="outside", decreasing=dict(marker=dict(color=theme.STAMP)), totals=dict(marker=dict(color=theme.SURVEY)),
                                 increasing=dict(marker=dict(color=theme.ACCENT))))
    fig.update_layout(title="From rent to what you keep")
    fig.update_yaxes(tickformat="~s")
    theme.show(theme.style(fig, 340, legend=False), key="i_wf")

    st.subheader("Home-loan tax break")
    c = st.columns(4)
    dp = c[0].slider("Down payment", 0, 90, 20, format="%d%%", key="i_dp")
    rate = c[1].slider("Interest rate", 6.0, 14.0, 8.5, 0.1, format="%.1f%%", key="i_rate")
    years = c[2].slider("Tenure", 5, 30, 20, key="i_years")
    slab = c[3].select_slider("Tax slab", [5, 10, 15, 20, 30], value=30, format_func=lambda x: f"{x}%", key="i_slab")
    c = st.columns(3)
    regime = c[0].segmented_control("Tax regime", ["old", "new"], default="old", format_func=str.title, key="i_reg") or "old"
    self_occ = c[1].toggle("I live here", True, key="i_self")
    o80 = c[2].number_input("Other 80C used (₹)", 0, 150000, 0, 10000, key="i_80c")
    y1 = finance.amortization_schedule(price * (1 - dp / 100), rate, years).iloc[:12]
    tb = finance.home_loan_tax_benefit(float(y1["interest"].sum()), float(y1["principal"].sum()), close0, self_occ, regime, slab, o80, first_year=True)
    k = st.columns(3)
    with k[0]: theme.kpi("Interest deduction", fmt_compact(tb.interest_deduction), f"year-1 interest {fmt_compact(y1['interest'].sum())}")
    with k[1]: theme.kpi("Section 80C", fmt_compact(tb.sec80c_deduction), "capped at ₹1.5 lakh")
    with k[2]: theme.kpi("Estimated tax saved", fmt_compact(tb.estimated_tax_saved), "in year 1", "survey")
    theme.note("Section 24(b) interest (up to ₹2 lakh for a home you live in) and Section 80C apply under the <b>old</b> regime only. "
               "The Income-tax Act, 2025 renumbers sections from FY 2026-27, so confirm current provisions with a CA.")
