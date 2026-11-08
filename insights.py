"""insights.py -- rule-based recommendations and scores (ported from app.py, Streamlit-free & testable)."""
from __future__ import annotations

import numpy as np

import config
from dataset_loader import CITY_PROFILES


def effective_yield_pct(prop_inputs: dict, predicted_price: float) -> float:
    """Gross rental yield: from the user's declared rent if given, else a 3% planning assumption."""
    rent = prop_inputs.get("annual_rental_income") or 0
    if rent and predicted_price:
        return rent / predicted_price * 100
    return 3.0


def generate_recommendations(prop_inputs: dict, tax_result, predicted_price: float) -> list:
    recs = []
    for k in tax_result.rebates_applied:
        recs.append(f"You're already benefiting from a **{k.replace('_', ' ').title()}** -- keep the supporting documents handy when filing.")

    if not prop_inputs.get("has_solar_panels") and not prop_inputs.get("has_rainwater_harvesting"):
        recs.append("Installing **solar panels** or **rainwater harvesting** can qualify for municipal green rebates in some cities (e.g. Pune/PMC) and also adds modelled resale value.")

    age = config.CURRENT_YEAR - prop_inputs.get("construction_year", 2015)
    if age > 25:
        recs.append("Properties over 25 years old usually get an **age-based depreciation** in unit-area / capital-value systems -- confirm it is applied on your municipal bill.")

    if prop_inputs.get("property_category") == "Residential" and prop_inputs.get("occupancy") == "Rented":
        recs.append("If you actually live here, switching the record to **owner-occupied** can lower property tax in cities like Mumbai, Pune and Bengaluru.")
    if prop_inputs.get("property_category") == "Commercial":
        recs.append("Commercial property attracts a notably higher property-tax rate than residential -- include it in ROI calculations.")

    circle = prop_inputs.get("circle_rate_value", 0)
    if circle and predicted_price and circle > predicted_price * 0.85:
        recs.append("The **circle rate is close to the predicted market value**, so stamp duty will be near the maximum for this property.")

    y = effective_yield_pct(prop_inputs, predicted_price)
    if prop_inputs.get("annual_rental_income") and y >= 4.0:
        recs.append(f"Your declared rent implies a gross yield of **{y:.1f}%**, above the typical 2-3.5% for Indian residential property.")
    elif prop_inputs.get("annual_rental_income") and y < 2.0:
        recs.append(f"Your declared rent implies a gross yield of only **{y:.1f}%** -- check whether the rent is below market.")

    if tax_result.is_fallback_used:
        recs.append("This state uses the **generic estimation ruleset**; cross-check against your municipal corporation's published rates.")

    market_avg = CITY_PROFILES.get(prop_inputs.get("city", ""), {}).get("avg_price_psqft")
    area = prop_inputs.get("built_up_area_sqft") or prop_inputs.get("plot_area_sqft")
    if market_avg and area and predicted_price:
        implied = predicted_price / max(area, 1)
        if implied > market_avg * 1.15:
            recs.append(f"Predicted price/sqft (\u20b9{implied:,.0f}) is **above the {prop_inputs.get('city')} average** (\u20b9{market_avg:,.0f}) -- premium location, quality or amenities.")
        elif implied < market_avg * 0.85:
            recs.append(f"Predicted price/sqft (\u20b9{implied:,.0f}) is **below the {prop_inputs.get('city')} average** (\u20b9{market_avg:,.0f}) -- possible value, or an older / less-amenitised property.")
    return recs or ["No specific tax-saving opportunities were identified -- the setup looks standard for its category."]


def investment_score(prop_inputs: dict, tax_result, predicted_price: float) -> int:
    score = 50.0
    score += float(np.clip((effective_yield_pct(prop_inputs, predicted_price) - 2.5) * 8, -20, 20))
    tier = CITY_PROFILES.get(prop_inputs.get("city", ""), {}).get("tier", "tier2")
    score += 12 if tier == "metro" else 4
    score -= min(15, (config.CURRENT_YEAR - prop_inputs.get("construction_year", 2015)) * 0.4)
    flags = [prop_inputs.get("nearby_metro"), prop_inputs.get("has_solar_panels"),
             prop_inputs.get("swimming_pool"), prop_inputs.get("green_certified")]
    score += sum(bool(x) for x in flags) * 2.5
    if tax_result.is_fallback_used:
        score -= 3
    return int(np.clip(score, 0, 100))


def risk_score(prop_inputs: dict, tax_result, interval_pct: float = 10.0) -> int:
    """Higher = riskier. Now also reflects valuation uncertainty (width of the 90% interval)."""
    risk = 25.0
    risk += min(25, (config.CURRENT_YEAR - prop_inputs.get("construction_year", 2015)) * 0.6)
    if prop_inputs.get("occupancy") == "Vacant":
        risk += 10
    if prop_inputs.get("property_category") == "Land":
        risk += 8
    if tax_result.is_fallback_used:
        risk += 7
    risk += float(np.clip((interval_pct - 15) * 0.8, 0, 10))
    return int(np.clip(risk, 0, 100))
