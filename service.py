"""service.py -- Streamlit-free business logic used by the web API (server.py)."""
from __future__ import annotations

import dataclasses

import numpy as np

import finance
import insights
import market
from dataset_loader import CITY_PROFILES
from tax_engine import PropertyInput, calculate_taxes
from valuation import ValuationModel

AMEN = {"Parking": "parking", "Lift": "lift_available", "Swimming pool": "swimming_pool", "Solar panels": "solar_panels",
        "Rainwater harvesting": "rainwater_harvesting", "Green certified": "green_certified", "Corner plot": "corner_plot"}
NEAR = {"Metro": "nearby_metro", "School": "nearby_school", "Hospital": "nearby_hospital", "Market": "nearby_market",
        "Airport": "nearby_airport", "Railway station": "nearby_railway_station"}
CATEGORY = {"Plot": "Land", "Office": "Commercial", "Shop": "Commercial", "Warehouse": "Industrial"}

_vm = None
_scored = None


def vm() -> ValuationModel:
    global _vm
    if _vm is None:
        _vm = ValuationModel()
    return _vm


def build_inputs(b: dict) -> dict:
    ptype = b.get("property_type", "Apartment")
    area = float(b.get("built_up_area_sqft", 1200))
    city = b.get("city", "Delhi")
    prof = CITY_PROFILES.get(city, {})
    inp = {"state": b.get("state") or prof.get("state", "Delhi"), "city": city, "property_type": ptype,
           "property_category": b.get("property_category") or CATEGORY.get(ptype, "Residential"),
           "built_up_area_sqft": area, "carpet_area_sqft": round(area * float(b.get("carpet_pct", 85)) / 100),
           "plot_area_sqft": float(b.get("plot_area_sqft") or max(area, 1500)), "construction_year": int(b.get("construction_year", 2015)),
           "bedrooms": int(b.get("bedrooms", 3)), "bathrooms": int(b.get("bathrooms", 2)), "floor_number": int(b.get("floor_number", 2)),
           "num_floors": int(b.get("num_floors", 1)), "construction_quality": b.get("construction_quality", "Standard"),
           "occupancy": b.get("occupancy", "Owner Occupied"), "structure_type": "RCC", "facing_direction": "East", "zone": "Central",
           "road_width_ft": 30, "latitude": prof.get("lat", 20.59), "longitude": prof.get("lon", 78.96),
           "circle_rate_value": 0, "govt_valuation": 0, "buyer_gender": b.get("buyer_gender", "Male"), "is_senior_citizen": False,
           "is_woman_owner": False, "is_disabled_or_exserviceman": False, "pays_in_lump_sum": True, "zone_or_category": None,
           "holding_period_months": 36, "annual_rental_income": 0, "purchase_price": None, "inflation_rate_pct": 6.0,
           "down_payment_pct": float(b.get("down_payment_pct", 20)), "loan_rate": float(b.get("loan_rate", 8.5)),
           "loan_years": int(b.get("loan_years", 20))}
    am, nb = set(b.get("amen", [])), set(b.get("near", []))
    for label, key in AMEN.items():
        inp[key] = label in am
    for label, key in NEAR.items():
        inp[key] = label in nb
    inp["has_solar_panels"], inp["has_rainwater_harvesting"] = inp["solar_panels"], inp["rainwater_harvesting"]
    return inp


def valuate_objs(body: dict):
    inp = build_inputs(body)
    m = vm()
    pred = m.predict(inp)
    tax = calculate_taxes(PropertyInput(
        state=inp["state"], city=inp["city"], property_type=inp["property_type"], property_category=inp["property_category"],
        built_up_area_sqft=inp["built_up_area_sqft"], carpet_area_sqft=inp["carpet_area_sqft"], plot_area_sqft=inp["plot_area_sqft"],
        construction_year=inp["construction_year"], market_value=pred.price, occupancy=inp["occupancy"], buyer_gender=inp["buyer_gender"],
        construction_quality=inp["construction_quality"], structure_type=inp["structure_type"], has_solar_panels=inp["solar_panels"],
        has_rainwater_harvesting=inp["rainwater_harvesting"], is_green_certified=inp["green_certified"], pays_in_lump_sum=True,
        holding_period_months=inp["holding_period_months"], inflation_rate_pct=inp["inflation_rate_pct"]))
    loan_amt = pred.price * (1 - inp["down_payment_pct"] / 100)
    emi, total, interest = finance.calculate_mortgage(loan_amt, inp["loan_rate"], inp["loan_years"])
    loan = {"loan_amount": loan_amt, "rate": inp["loan_rate"], "years": inp["loan_years"], "emi": emi, "total_payment": total,
            "total_interest": interest, "tax_benefit_year1": 0.0}
    drivers = m.explain(inp)
    recs = insights.generate_recommendations({**inp, "has_solar_panels": inp["solar_panels"]}, tax, pred.price)
    scores = {"investment": insights.investment_score(inp, tax, pred.price), "risk": insights.risk_score(inp, tax, pred.interval_pct)}
    return inp, pred, tax, loan, drivers, recs, scores


def valuate(body: dict) -> dict:
    inp, pred, tax, loan, d, recs, scores = valuate_objs(body)
    ys = finance.yearly_summary(finance.amortization_schedule(loan["loan_amount"], inp["loan_rate"], inp["loan_years"]))
    area = inp["plot_area_sqft"] if inp["property_type"] == "Plot" else inp["built_up_area_sqft"]
    anchor = CITY_PROFILES.get(inp["city"], {}).get("avg_price_psqft")
    return {"pred": dataclasses.asdict(pred), "tax": dataclasses.asdict(tax), "per_sqft": pred.price / max(area, 1), "anchor_psqft": anchor,
            "loan": {"amount": loan["loan_amount"], "emi": loan["emi"], "total": loan["total_payment"], "interest": loan["total_interest"], "yearly": ys.to_dict("list")},
            "drivers": {"baseline": d.attrs["baseline_price"], "price": d.attrs["price"], "groups": d[["group", "pct_effect", "rupee_effect"]].to_dict("records")},
            "recs": recs, "scores": scores}


def report_pdf(body: dict) -> bytes:
    from report import build_pdf_report
    inp, pred, tax, loan, d, recs, scores = valuate_objs(body)
    return build_pdf_report(inputs=inp, prediction=pred, tax=tax, loan=loan, drivers=d, recommendations=recs, scores=scores, model_meta=vm().meta)


def negotiate(body: dict) -> dict:
    """Where an asking price sits in the model's predictive distribution (empirical residuals from the calibration split)."""
    inp = build_inputs(body)
    m = vm()
    X, _ = m.build_row(inp)
    pl = float(m.model.predict(X)[0])
    residuals()
    split, y, p = _scored
    signed = (y - p)[split == "calib"]
    ask = float(body.get("asking", 0))
    e = np.log1p(ask) - pl
    pct = float((signed <= e).mean() * 100)
    qs = {str(q): float(np.expm1(pl + np.quantile(signed, q / 100))) for q in (10, 25, 50, 75, 90)}
    return {"asking": ask, "percentile": pct, "quantiles": qs, "median_gap_pct": (ask / qs["50"] - 1) * 100 if qs["50"] else 0}


def loan_lab(b: dict) -> dict:
    price, dp, rate, years = float(b["price"]), float(b["down"]), float(b["rate"]), int(b["years"])
    loan = price * (1 - dp / 100)
    emi, total, interest = finance.calculate_mortgage(loan, rate, years)
    mp, ls = float(b.get("prepay", 0)), float(b.get("lump", 0))
    imp = finance.prepayment_impact(loan, rate, years, mp, ls, 12)
    base = finance.amortization_schedule(loan, rate, years)
    new = finance.amortization_schedule(loan, rate, years, mp, ls, 12)
    rates = [round(6 + 0.5 * i, 1) for i in range(17)]
    tenures = [5, 10, 15, 20, 25, 30]
    return {"loan": loan, "emi": emi, "total": total, "interest": interest, "impact": imp,
            "base": {"year": (base["month"] / 12).round(3).tolist(), "balance": base["balance"].round(0).tolist()},
            "new": {"year": (new["month"] / 12).round(3).tolist(), "balance": new["balance"].round(0).tolist()},
            "heat": {"rates": rates, "tenures": tenures, "z": [[finance.emi(loan, r, t) for r in rates] for t in tenures]}}


def afford(b: dict) -> dict:
    inc, ex, foir, dp, rate, years = float(b["income"]), float(b.get("existing", 0)), float(b["foir"]), float(b["down"]), float(b["rate"]), int(b["years"])
    a = finance.affordability(inc, ex, rate, years, dp, foir, 2.0)
    bumps = [0, 0.5, 1, 1.5, 2, 3, 4]
    stress = [finance.affordability(inc, ex, rate, years, dp, foir, x)["stressed_max_price"] if x else a["max_property_price"] for x in bumps]
    mk = market.load_market_frame()
    ptype, area = b.get("ptype", "Apartment"), float(b.get("area", 1000))
    med = (mk[mk["property_type"] == ptype].groupby("city")["price_per_sqft"].agg(["median", "size"]).query("size >= 8")["median"] * area).sort_values()
    return {"result": a, "bumps": bumps, "stress": stress, "cities": [{"city": c, "price": float(v)} for c, v in med.items()]}


def rent_buy(b: dict) -> dict:
    args = dict(price=float(b["price"]), closing_costs=float(b.get("closing", 0)), down_payment_pct=float(b["down"]), loan_rate_pct=float(b["rate"]),
                tenure_years=int(b["years"]), monthly_rent=float(b["rent"]), rent_growth_pct=float(b["rg"]), appreciation_pct=float(b["ap"]),
                investment_return_pct=float(b["inv"]), years=int(b["horizon"]), annual_maintenance=float(b.get("maint", 0)),
                annual_property_tax=float(b.get("ptax", 0)), selling_cost_pct=float(b.get("sell", 2)))
    df = finance.rent_vs_buy(**args)
    aps, invs = [2, 4, 6, 8, 10, 12], [4, 6, 8, 10, 12, 14]
    z = [[finance.rent_vs_buy(**{**args, "appreciation_pct": a, "investment_return_pct": i}).iloc[-1]["advantage_buy"] for a in aps] for i in invs]
    return {"series": df[["year", "buy_net_worth", "rent_net_worth"]].to_dict("list"), "breakeven": finance.breakeven_year(df), "heat": {"aps": aps, "invs": invs, "z": z}}


_BOOLS = list(NEAR.values()) + list(AMEN.values())


def bulk(csv_text: str, limit: int = 5000) -> dict:
    import io
    import pandas as pd
    df = pd.read_csv(io.StringIO(csv_text)).head(limit)
    missing = [c for c in ("city", "state", "property_type", "construction_year") if c not in df.columns]
    if missing or not ({"built_up_area_sqft", "plot_area_sqft"} & set(df.columns)):
        return {"error": f"Missing required column(s): {missing or 'built_up_area_sqft or plot_area_sqft'}"}
    for c in _BOOLS:
        if c in df.columns:
            df[c] = df[c].map(lambda v: str(v).strip().lower() in ("1", "true", "yes", "y", "t", "1.0"))
    if "property_category" not in df.columns:
        df["property_category"] = df["property_type"].map(CATEGORY).fillna("Residential")
    if "built_up_area_sqft" not in df.columns:
        df["built_up_area_sqft"] = df["plot_area_sqft"]
    if "plot_area_sqft" not in df.columns:
        df["plot_area_sqft"] = df["built_up_area_sqft"]
    if "carpet_area_sqft" not in df.columns:
        df["carpet_area_sqft"] = df["built_up_area_sqft"] * 0.85
    out = vm().predict_batch(df)
    return {"columns": list(out.columns), "rows": out.where(out.notna(), None).values.tolist(), "n": len(out)}


def save(body: dict) -> int:
    import storage
    inp, pred, tax, loan, *_ = valuate_objs(body)
    area = inp["plot_area_sqft"] if inp["property_type"] == "Plot" else inp["built_up_area_sqft"]
    return storage.save_valuation({"state": inp["state"], "city": inp["city"], "property_type": inp["property_type"], "area_sqft": area,
                                   "predicted_price": pred.price, "lower_90": pred.lower_90, "upper_90": pred.upper_90,
                                   "annual_property_tax": tax.annual_property_tax, "stamp_duty": tax.stamp_duty,
                                   "total_purchase_cost": tax.total_purchase_cost, "emi": loan["emi"]}, inputs=body)


def history() -> list:
    import storage
    return storage.load_history().drop(columns=["inputs_json"]).to_dict("records")


def delete_history(rid: int):
    import storage
    storage.delete_valuation(int(rid))
def meta() -> dict:
    m = vm()
    cities = [{"city": c, "state": p["state"]} for c, p in sorted(CITY_PROFILES.items())]
    return {"cities": cities, "amenities": list(AMEN), "nearby": list(NEAR), "model": {k: m.meta.get(k) for k in ("trained_at", "test_metrics", "empirical_coverage_test", "conformal_log_quantiles")},
            "n_features": len(m.features)}


def market_rows() -> list:
    mk = market.load_market_frame()
    g = mk.groupby(["city", "property_type"]).agg(psqft=("price_per_sqft", "median"), price=("market_price", "median"), n=("market_price", "size")).reset_index()
    g = g[g["n"] >= 5]
    g["lat"] = g["city"].map(lambda c: CITY_PROFILES.get(c, {}).get("lat"))
    g["lon"] = g["city"].map(lambda c: CITY_PROFILES.get(c, {}).get("lon"))
    g["state"] = g["city"].map(lambda c: CITY_PROFILES.get(c, {}).get("state"))
    return g.dropna(subset=["lat", "lon"]).to_dict("records")


def residuals() -> dict:
    global _scored
    if _scored is None:
        m, mk = vm(), market.load_market_frame()
        pred = m.model.predict(mk[m.features])
        _scored = (mk["split"].values, mk["price"].values, pred)
    split, y, p = _scored
    out = {}
    for name in ("calib", "test"):
        s = split == name
        out[name] = {"abs_log": np.abs(y[s] - p[s]).round(5).tolist(), "ratio": (np.expm1(y[s]) / np.expm1(p[s])).round(4).tolist()}
    return out
