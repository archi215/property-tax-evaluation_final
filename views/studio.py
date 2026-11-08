"""views/studio.py -- Valuation Studio: every input is live, results update as you change them."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import finance
import insights
import market
import storage
import theme
from dataset_loader import CITY_PROFILES
from report import build_pdf_report
from tax_engine import PropertyInput, calculate_taxes
from ui_common import (ALL_STATES, CONSTRUCTION_QUALITY, FACING_DIRECTIONS, PROPERTY_CATEGORY_MAP, PROPERTY_TYPES,
                       STATE_CITY_MAP, get_market, require_model, tax_rules_banner)
from utils import fmt_compact, fmt_inr

AMENITIES = {"Parking": "parking", "Lift": "lift_available", "Swimming pool": "swimming_pool", "Solar panels": "solar_panels",
             "Rainwater harvesting": "rainwater_harvesting", "Green certified": "green_certified", "Corner plot": "corner_plot"}
NEARBY = {"Metro": "nearby_metro", "School": "nearby_school", "Hospital": "nearby_hospital", "Market": "nearby_market",
          "Airport": "nearby_airport", "Railway station": "nearby_railway_station"}
OWNER_OPTS = ["Senior citizen", "Woman owner", "Disabled or ex-servicemen", "Pays tax in one go"]
CY = config.CURRENT_YEAR


# ═════════════════════════════ inputs ═════════════════════════════
def _panel() -> dict:
    ss = st.session_state
    t_place, t_prop, t_feat, t_owner, t_loan = st.tabs(["Place", "Property", "Features", "Owner", "Loan"])

    with t_place:
        state = st.selectbox("State", ALL_STATES, index=ALL_STATES.index("Delhi"), key="s_state")
        cities = STATE_CITY_MAP.get(state, [])
        if cities:
            if ss.get("s_city") not in cities:
                ss["s_city"] = cities[0]
            city = st.selectbox("City", cities, key="s_city")
        else:
            city = st.text_input("City (not in the dataset, so a national average is used)", value=state, key="s_city_txt")
        zone = st.segmented_control("Zone of the city", ["Central", "North", "South", "East", "West"], default="Central", key="s_zone") or "Central"

    with t_prop:
        ptype = st.pills("Property type", PROPERTY_TYPES, default="Apartment", key="s_ptype") or "Apartment"
        cat_default = PROPERTY_CATEGORY_MAP.get(ptype, "Residential")
        c1, c2 = st.columns(2)
        category = c1.selectbox("Use", ["Residential", "Commercial", "Industrial", "Land"], index=["Residential", "Commercial", "Industrial", "Land"].index(cat_default), key=f"s_cat_{ptype}")
        occupancy = c2.selectbox("Occupancy", ["Owner Occupied", "Rented", "Vacant"], key="s_occ")
        area = st.slider("Built-up area (sq ft)", 200, 10000, 1200, 50, key="s_area")
        carpet_pct = st.slider("Carpet area as a share of built-up", 60, 95, 85, 1, format="%d%%", key="s_carpet")
        plot = area
        if ptype in ("Plot", "Villa", "Independent House"):
            plot = st.slider("Plot area (sq ft)", 200, 20000, max(area, 1500), 50, key="s_plot")
        c1, c2 = st.columns(2)
        year = c1.slider("Construction year", 1950, CY, 2015, key="s_year")
        quality = c2.select_slider("Construction quality", CONSTRUCTION_QUALITY, value="Standard", key="s_quality")
        c1, c2, c3 = st.columns(3)
        beds = c1.number_input("Bedrooms", 0, 12, 3, key="s_beds")
        baths = c2.number_input("Bathrooms", 0, 12, 2, key="s_baths")
        floor_no = c3.number_input("Floor", 0, 80, 2, key="s_floor")
        c1, c2, c3 = st.columns(3)
        floors = c1.number_input("Floors in building", 1, 50, 1, key="s_floors")
        structure = c2.selectbox("Structure", ["RCC", "Load Bearing", "Non-RCC", "Chawl"], key="s_struct")
        facing = c3.selectbox("Facing", FACING_DIRECTIONS, key="s_facing")

    with t_feat:
        am = st.pills("Amenities and green features", list(AMENITIES), selection_mode="multi", default=["Parking", "Lift"], key="s_am") or []
        nb = st.pills("Close by", list(NEARBY), selection_mode="multi", default=["School", "Hospital", "Market"], key="s_nb") or []
        road = st.slider("Road width (ft)", 4, 150, 30, key="s_road")
        c1, c2 = st.columns(2)
        circle = c1.number_input("Circle rate value (₹, optional)", 0, value=0, step=50000, key="s_circle",
                                 help="The government's guideline value. It only affects stamp duty, not the market valuation.")
        govt = c2.number_input("Government valuation (₹, optional)", 0, value=0, step=50000, key="s_govt")

    with t_owner:
        gender = st.segmented_control("Buyer category (stamp duty)", ["Male", "Female", "Joint"], default="Male", key="s_gender") or "Male"
        own = st.pills("Rebates that apply to you", OWNER_OPTS, selection_mode="multi", default=["Pays tax in one go"], key="s_own") or []
        c1, c2 = st.columns(2)
        hold = c1.number_input("Months you will hold it (capital gains)", 0, 600, 36, key="s_hold")
        rent_income = c2.number_input("Annual rent you receive (₹)", 0, value=0, step=10000, key="s_rent")
        c1, c2 = st.columns(2)
        purchase = c1.number_input("What you paid (₹, 0 = estimate)", 0, value=0, step=50000, key="s_paid")
        infl = c2.slider("Inflation for projections", 2.0, 12.0, 6.0, 0.5, format="%.1f%%", key="s_infl")
        override = st.text_input("Tax zone or category override (optional)", "", key="s_override", help="A–H for Delhi, A–F for Bengaluru")

    with t_loan:
        dp = st.slider("Down payment", 0, 90, 20, format="%d%%", key="s_dp")
        rate = st.slider("Interest rate", 6.0, 14.0, 8.5, 0.1, format="%.1f%%", key="s_rate")
        years = st.slider("Tenure (years)", 5, 30, 20, key="s_years")

    prof = CITY_PROFILES.get(city, {})
    inp = {"state": state, "city": city, "zone": zone, "property_type": ptype, "property_category": category, "occupancy": occupancy,
           "built_up_area_sqft": area, "carpet_area_sqft": round(area * carpet_pct / 100), "plot_area_sqft": plot, "construction_year": year,
           "num_floors": floors, "floor_number": floor_no, "bedrooms": beds, "bathrooms": baths, "construction_quality": quality,
           "structure_type": structure, "facing_direction": facing, "road_width_ft": road,
           "circle_rate_value": circle, "govt_valuation": govt, "latitude": prof.get("lat", 20.5937), "longitude": prof.get("lon", 78.9629),
           "buyer_gender": gender, "is_senior_citizen": "Senior citizen" in own, "is_woman_owner": "Woman owner" in own,
           "is_disabled_or_exserviceman": "Disabled or ex-servicemen" in own, "pays_in_lump_sum": "Pays tax in one go" in own,
           "zone_or_category": override.strip() or None, "holding_period_months": hold, "annual_rental_income": rent_income,
           "purchase_price": purchase or None, "inflation_rate_pct": infl, "down_payment_pct": dp, "loan_rate": rate, "loan_years": years}
    for label, key in {**AMENITIES, **NEARBY}.items():
        inp[key] = label in am or label in nb
    inp["has_solar_panels"], inp["has_rainwater_harvesting"] = inp["solar_panels"], inp["rainwater_harvesting"]
    return inp


# ═════════════════════════════ compute ═════════════════════════════
def compute(inp: dict) -> dict:
    vm = require_model()
    pred = vm.predict(inp)
    tax = calculate_taxes(PropertyInput(
        state=inp["state"], city=inp["city"], property_type=inp["property_type"], property_category=inp["property_category"],
        built_up_area_sqft=inp["built_up_area_sqft"], carpet_area_sqft=inp["carpet_area_sqft"], plot_area_sqft=inp["plot_area_sqft"],
        construction_year=inp["construction_year"], market_value=pred.price, circle_rate_value=inp["circle_rate_value"] or None,
        govt_guideline_value=inp["govt_valuation"] or None, occupancy=inp["occupancy"], buyer_gender=inp["buyer_gender"],
        construction_quality=inp["construction_quality"], structure_type=inp["structure_type"], is_senior_citizen=inp["is_senior_citizen"],
        is_woman_owner=inp["is_woman_owner"], is_disabled_or_exserviceman=inp["is_disabled_or_exserviceman"],
        has_solar_panels=inp["solar_panels"], has_rainwater_harvesting=inp["rainwater_harvesting"], is_green_certified=inp["green_certified"],
        zone_or_category=inp["zone_or_category"], pays_in_lump_sum=inp["pays_in_lump_sum"], annual_rental_income=inp["annual_rental_income"],
        holding_period_months=inp["holding_period_months"], purchase_price=inp["purchase_price"], inflation_rate_pct=inp["inflation_rate_pct"]))
    loan_amt = pred.price * (1 - inp["down_payment_pct"] / 100)
    emi, total_pay, total_int = finance.calculate_mortgage(loan_amt, inp["loan_rate"], inp["loan_years"])
    return {"inputs": inp, "pred": pred, "tax": tax, "drivers": vm.explain(inp),
            "loan": {"loan_amount": loan_amt, "rate": inp["loan_rate"], "years": inp["loan_years"], "emi": emi,
                     "total_payment": total_pay, "total_interest": total_int, "tax_benefit_year1": 0.0},
            "recs": insights.generate_recommendations(inp, tax, pred.price),
            "scores": {"investment": insights.investment_score(inp, tax, pred.price), "risk": insights.risk_score(inp, tax, pred.interval_pct)}}


# ═════════════════════════════ result tabs ═════════════════════════════
def _drivers(res):
    d = res["drivers"]
    base, price = d.attrs["baseline_price"], d.attrs["price"]
    x = ["Average property"] + list(d["group"]) + ["This property"]
    y = [base] + list(d["rupee_effect"]) + [price]
    text = [fmt_compact(base)] + [fmt_compact(r) for r in d["rupee_effect"]] + [fmt_compact(price)]
    hover = ["Typical property in the training data"] + [f"{g}: {p:+.1f}% on the price" for g, p in zip(d["group"], d["pct_effect"])] + ["Model estimate"]
    fig = go.Figure(go.Waterfall(x=x, y=y, measure=["absolute"] + ["relative"] * len(d) + ["total"], text=text, textposition="outside",
                                 hovertext=hover, hoverinfo="text+x", connector=dict(line=dict(color="rgba(128,128,128,.4)")),
                                 increasing=dict(marker=dict(color=theme.SURVEY)), decreasing=dict(marker=dict(color=theme.STAMP)),
                                 totals=dict(marker=dict(color=theme.ACCENT))))
    fig.update_layout(title="From an average property to this one")
    fig.update_yaxes(tickformat="~s")
    theme.show(theme.style(fig, 420, legend=False), key="st_wf")
    st.caption("Each bar is the exact SHAP contribution of a group of features, in rupees. These describe what the model learned from a synthetic dataset, not causes in the real market.")


def _market(res):
    inp, p = res["inputs"], res["pred"]
    area = inp["plot_area_sqft"] if inp["property_type"] == "Plot" else inp["built_up_area_sqft"]
    psqft = p.price / max(area, 1)
    pos = market.price_percentile(get_market(), inp["city"], inp["property_type"], psqft)
    if not pos:
        st.info("Not enough comparable listings for this city in the dataset.")
        return
    mk = get_market()
    peers = mk[(mk["city"] == inp["city"]) & (mk["property_type"] == inp["property_type"])]["price_per_sqft"]
    if len(peers) < 20:
        peers = mk[mk["city"] == inp["city"]]["price_per_sqft"]
    fig = go.Figure(go.Histogram(x=peers, nbinsx=34, marker_color="rgba(43,58,140,.35)", name="Comparable listings"))
    fig.add_vline(x=psqft, line_color=theme.STAMP, line_width=3, annotation_text="This property", annotation_position="top")
    fig.add_vline(x=pos["median"], line_dash="dash", line_color=theme.ACCENT, annotation_text="Median", annotation_position="bottom right")
    fig.update_layout(title=f"Price per sq ft among {inp['city']} listings", xaxis_title="₹ per sq ft", yaxis_title="Listings")
    theme.show(theme.style(fig, 340, legend=False), key="st_mk")
    st.write(f"Priced above **{pos['percentile']:.0f}%** of {pos['n']} comparable listings. The middle 80% of them sit between "
             f"{fmt_inr(pos['p10'])} and {fmt_inr(pos['p90'])} per sq ft.")


def _costs(res):
    t, p, inp = res["tax"], res["pred"], res["inputs"]
    extra = t.one_time_total
    c1, c2, c3 = st.columns(3)
    with c1: theme.kpi("One-time charges on top of the price", fmt_compact(extra), f"{extra / p.price * 100:.1f}% of the value", "stamp")
    with c2: theme.kpi("Total to acquire", fmt_compact(t.total_purchase_cost))
    with c3: theme.kpi("Recurring per year", fmt_compact(t.recurring_annual_total), f"{fmt_inr(t.recurring_annual_total / 12)} a month")
    l, r = st.columns(2)
    with l:
        vals = {"Stamp duty": t.stamp_duty, "Registration": t.registration_charges, "Mutation": t.mutation_charges}
        fig = go.Figure(go.Pie(labels=list(vals), values=[max(v, 0) for v in vals.values()], hole=0.55, sort=False,
                               marker=dict(colors=[theme.STAMP, theme.ACCENT, theme.GOLD]), textinfo="label+percent"))
        fig.update_layout(title="One-time charges")
        theme.show(theme.style(fig, 320, legend=False), key="st_ot")
    with r:
        rec = {"Property tax": t.annual_property_tax, "Water": t.water_tax, "Sewerage": t.sewerage_tax, "Solid waste": t.solid_waste_management_charge,
               "Cesses": t.education_cess + t.fire_cess + t.health_cess + t.urban_development_cess}
        if sum(max(v, 0) for v in rec.values()) > 0:
            fig = go.Figure(go.Pie(labels=list(rec), values=[max(v, 0) for v in rec.values()], hole=0.55, sort=False,
                                   marker=dict(colors=theme.SEQ), textinfo="label+percent"))
            fig.update_layout(title="Yearly recurring charges")
            theme.show(theme.style(fig, 320, legend=False), key="st_rec")
        else:
            st.info("No recurring tax to chart for this setup.")
    g1, g2 = st.columns(2)
    with g1: theme.kpi("Capital gains tax if sold (estimate)", fmt_compact(t.capital_gains_tax_estimate), t.capital_gains_method[:90], "survey")
    with g2: theme.kpi("Tax on rental income (estimate)", fmt_compact(t.rental_income_tax_estimate), "After the 30% standard deduction")
    with st.expander("Every line of the tax calculation"):
        rows = [("Annual property tax", t.annual_property_tax), ("Water tax", t.water_tax), ("Sewerage tax", t.sewerage_tax),
                ("Solid waste charge", t.solid_waste_management_charge), ("Education cess", t.education_cess), ("Fire cess", t.fire_cess),
                ("Health cess", t.health_cess), ("Urban development cess", t.urban_development_cess),
                ("Luxury property tax", t.luxury_property_tax), ("Green rebate (deducted)", -t.green_tax_rebate)]
        st.dataframe(pd.DataFrame(rows, columns=["Charge", "₹ per year"]), width="stretch", hide_index=True)
        if t.property_tax_breakdown: st.json(t.property_tax_breakdown, expanded=False)
    for w in t.warnings:
        st.warning(w)
    theme.note(f"Rules used: <b>{t.rule_source}</b>{' (generic fallback, so treat as a rough estimate)' if t.is_fallback_used else ''}. {t.disclaimer}")


def _loan(res):
    L, t, inp = res["loan"], res["tax"], res["inputs"]
    c = st.columns(4)
    with c[0]: theme.kpi("Loan", fmt_compact(L["loan_amount"]))
    with c[1]: theme.kpi("Monthly EMI", fmt_inr(L["emi"]), "", "stamp")
    with c[2]: theme.kpi("Total interest", fmt_compact(L["total_interest"]), f"{L['total_interest'] / max(L['loan_amount'], 1) * 100:.0f}% of the loan")
    with c[3]: theme.kpi("Total repaid", fmt_compact(L["total_payment"]))
    c1, c2 = st.columns(2)
    mp = c1.slider("Extra every month (₹)", 0, 50000, 0, 1000, key="s_mp")
    ls = c2.slider("One-off prepayment in month 12 (₹)", 0, 2_000_000, 0, 50000, key="s_ls")
    imp = finance.prepayment_impact(L["loan_amount"], L["rate"], L["years"], mp, ls, 12)
    sched = finance.amortization_schedule(L["loan_amount"], L["rate"], L["years"], mp, ls, 12)
    ys = finance.yearly_summary(sched)
    if mp or ls:
        st.success(f"You save {fmt_inr(imp['interest_saved'])} in interest and finish {imp['months_saved']} months earlier.")
    fig = go.Figure()
    fig.add_bar(x=ys["year"], y=ys["principal"] + ys["prepayment"], name="Principal", marker_color=theme.SURVEY)
    fig.add_bar(x=ys["year"], y=ys["interest"], name="Interest", marker_color=theme.STAMP)
    fig.add_scatter(x=ys["year"], y=ys["closing_balance"], name="Still owed", mode="lines+markers", line=dict(color=theme.ACCENT, width=3), yaxis="y2")
    fig.update_layout(barmode="stack", title="Where each year's payments go", xaxis_title="Loan year", yaxis_title="Paid in the year",
                      yaxis2=dict(title="Balance", overlaying="y", side="right", showgrid=False))
    fig.update_yaxes(tickformat="~s")
    theme.show(theme.style(fig, 360), key="st_am")
    st.markdown("**Tax benefit on the loan (old tax regime)**")
    c = st.columns(3)
    self_occ = c[0].toggle("I will live here", inp["occupancy"] != "Rented", key="s_selfocc")
    slab = c[1].select_slider("Your tax slab", [5, 10, 15, 20, 30], value=30, format_func=lambda x: f"{x}%", key="s_slab")
    o80 = c[2].number_input("Other 80C already used (₹)", 0, 150000, 0, step=10000, key="s_80c")
    y1 = sched.iloc[:12]
    tb = finance.home_loan_tax_benefit(float(y1["interest"].sum()), float(y1["principal"].sum() + y1["prepayment"].sum()),
                                       t.stamp_duty + t.registration_charges, self_occ, "old", slab, o80, first_year=True)
    res["loan"]["tax_benefit_year1"] = tb.estimated_tax_saved
    c = st.columns(3)
    with c[0]: theme.kpi("Interest deduction (24b)", fmt_compact(tb.interest_deduction))
    with c[1]: theme.kpi("Section 80C", fmt_compact(tb.sec80c_deduction), "Principal plus stamp duty and registration")
    with c[2]: theme.kpi("Tax saved in year 1", fmt_compact(tb.estimated_tax_saved), "", "survey")
    st.caption(tb.note + " The new regime gives no such deduction for a home you live in.")


def _future(res):
    t, p, inp = res["tax"], res["pred"], res["inputs"]
    c = st.columns(3)
    bear = c[0].slider("Slow growth", 0.0, 8.0, 3.0, 0.5, format="%.1f%% a year", key="s_bear")
    base = c[1].slider("Steady growth", 0.0, 12.0, 6.0, 0.5, format="%.1f%% a year", key="s_base")
    bull = c[2].slider("Fast growth", 2.0, 15.0, 9.0, 0.5, format="%.1f%% a year", key="s_bull")
    sc = finance.appreciation_scenarios(p.price, 15, bear, base, bull)
    fig = go.Figure()
    fig.add_scatter(x=sc["year"], y=sc["bull"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=sc["year"], y=sc["bear"], line=dict(width=0), fill="tonexty", fillcolor="rgba(43,58,140,.12)", showlegend=False, hoverinfo="skip")
    for k, name, col in (("bear", "Slow", theme.STAMP), ("base", "Steady", theme.ACCENT), ("bull", "Fast", theme.SURVEY)):
        fig.add_scatter(x=sc["year"], y=sc[k], name=name, line=dict(color=col, width=3), hovertemplate="Year %{x}: %{y:,.0f}<extra>" + name + "</extra>")
    fig.update_layout(title="What the property could be worth", xaxis_title="Years from now", yaxis_title="Value")
    fig.update_yaxes(tickformat="~s")
    theme.show(theme.style(fig, 360), key="st_fut")
    st.caption("Growth rates are your assumptions. The model only values the property today.")
    yrs = [0, 5, 10, 15]
    vals = [t.recurring_annual_total] + [t.projections.get(f"{y}_year", {}).get("projected_annual_tax", 0) for y in (5, 10, 15)]
    fig = go.Figure(go.Scatter(x=yrs, y=vals, mode="lines+markers", line=dict(color=theme.ACCENT, width=3), fill="tozeroy", fillcolor="rgba(43,58,140,.08)"))
    fig.update_layout(title=f"Yearly recurring charges at {inp['inflation_rate_pct']:.1f}% inflation", xaxis_title="Years from now")
    fig.update_yaxes(tickformat="~s")
    theme.show(theme.style(fig, 300, legend=False), key="st_taxp")


def _advice(res):
    for r in res["recs"]:
        st.markdown(f"- {r}")
    c1, c2 = st.columns(2)
    for col, key, title, good_high in ((c1, "investment", "Investment score", True), (c2, "risk", "Risk score (lower is safer)", False)):
        fig = go.Figure(go.Indicator(mode="gauge+number", value=res["scores"][key], title={"text": title},
                                     gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=theme.ACCENT),
                                                steps=[dict(range=[0, 40], color="rgba(47,125,107,.25)" if not good_high else "rgba(184,58,47,.20)"),
                                                       dict(range=[40, 70], color="rgba(184,137,43,.22)"),
                                                       dict(range=[70, 100], color="rgba(184,58,47,.20)" if not good_high else "rgba(47,125,107,.25)")])))
        with col: theme.show(theme.style(fig, 230, legend=False), key=f"st_g_{key}")
    st.caption("Scores are rule-of-thumb indicators for exploring the tool, not financial advice.")


# ═════════════════════════════ page ═════════════════════════════
def render():
    theme.hero("Valuation Studio", "Change any detail on the left and the value, the range, the taxes and the loan update straight away.",
               ["Calibrated 90% and 80% ranges", "Exact price drivers", "Tax rules for 3 states plus a generic fallback"])
    tax_rules_banner()
    vm = require_model()
    left, right = st.columns([5, 7], gap="large")
    with left:
        inp = _panel()
    res = compute(inp)
    st.session_state["studio_inputs"], st.session_state["studio_result"] = inp, res
    p, t = res["pred"], res["tax"]
    area = inp["plot_area_sqft"] if inp["property_type"] == "Plot" else inp["built_up_area_sqft"]
    anchor = CITY_PROFILES.get(inp["city"], {}).get("avg_price_psqft")
    with right:
        theme.headline_price(p.price, f"for a {inp['built_up_area_sqft']:,} sq ft {inp['property_type'].lower()} in {inp['city']}")
        theme.ruler(p.price, p.lower_90, p.upper_90, p.lower_80, p.upper_80)
        for w in p.warnings:
            st.warning(w)
        c = st.columns(4)
        with c[0]: theme.kpi("Per sq ft", fmt_inr(p.price / max(area, 1)), f"{(p.price / max(area, 1) / anchor - 1) * 100:+.0f}% vs the {inp['city']} average" if anchor else "")
        with c[1]: theme.kpi("Total to acquire", fmt_compact(t.total_purchase_cost), "price, stamp duty and fees", "stamp")
        with c[2]: theme.kpi("EMI", fmt_inr(res["loan"]["emi"]), f"{inp['loan_years']} years at {inp['loan_rate']:.1f}%")
        with c[3]: theme.kpi("Property tax", fmt_compact(t.annual_property_tax), "per year", "survey")
        b1, b2, b3 = st.columns(3)
        if b1.button("Save to history", width="stretch", icon=":material/bookmark:"):
            storage.save_valuation({"state": inp["state"], "city": inp["city"], "property_type": inp["property_type"], "area_sqft": area,
                                    "predicted_price": p.price, "lower_90": p.lower_90, "upper_90": p.upper_90,
                                    "annual_property_tax": t.annual_property_tax, "stamp_duty": t.stamp_duty,
                                    "total_purchase_cost": t.total_purchase_cost, "emi": res["loan"]["emi"]}, inputs=inp)
            st.toast("Saved to history.", icon=":material/check:")
        if b2.button("Pin to compare", width="stretch", icon=":material/push_pin:"):
            pins = st.session_state.setdefault("pins", [])
            pins.append({"Label": f"{inp['city']} {inp['property_type']} {inp['built_up_area_sqft']:,} sq ft", "Value": p.price, "Low 90%": p.lower_90,
                         "High 90%": p.upper_90, "Per sq ft": p.price / max(area, 1), "Total to acquire": t.total_purchase_cost,
                         "EMI": res["loan"]["emi"], "Property tax": t.annual_property_tax})
            st.toast("Pinned. Change the inputs and pin another to compare.", icon=":material/push_pin:")
        try:
            pdf = build_pdf_report(inputs=inp, prediction=p, tax=t, loan=res["loan"], drivers=res["drivers"], recommendations=res["recs"],
                                   scores=res["scores"], model_meta=vm.meta)
            b3.download_button("Download PDF", pdf, file_name=f"valuation_{inp['city']}.pdf", mime="application/pdf", width="stretch", icon=":material/download:")
        except Exception as e:
            b3.error(f"PDF failed: {e}")
        tabs = st.tabs(["Why this price", "Local market", "Costs and taxes", "Loan", "Future", "Advice"])
        for tab, fn in zip(tabs, (_drivers, _market, _costs, _loan, _future, _advice)):
            with tab:
                fn(res)

    pins = st.session_state.get("pins", [])
    if pins:
        st.divider()
        st.subheader("Pinned for comparison")
        df = pd.DataFrame(pins)
        fig = go.Figure()
        fig.add_bar(x=df["Label"], y=df["Value"], name="Value", marker_color=theme.ACCENT,
                    error_y=dict(type="data", symmetric=False, array=df["High 90%"] - df["Value"], arrayminus=df["Value"] - df["Low 90%"]))
        fig.add_bar(x=df["Label"], y=df["Total to acquire"] - df["Value"], name="Charges on top", marker_color=theme.STAMP)
        fig.update_layout(barmode="stack", title="Value with its 90% range, and the charges on top")
        fig.update_yaxes(tickformat="~s")
        theme.show(theme.style(fig, 340), key="st_pins")
        st.dataframe(df.style.format({c: fmt_inr for c in df.columns if c != "Label"}), width="stretch", hide_index=True)
        if st.button("Clear pins", icon=":material/delete:"):
            st.session_state["pins"] = []
            st.rerun()
