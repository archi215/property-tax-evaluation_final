"""
app.py
======
Smart Indian Property Valuation & Property Tax Assessment System
------------------------------------------------------------------
A Streamlit application that predicts an Indian property's market value
using a trained XGBoost model, then runs the prediction through a
state/city-aware tax_engine to compute every applicable Indian property
tax, with mortgage calculation, prediction history, confidence interval,
and feature importance views preserved from the original project.

Run with: streamlit run app.py
"""

import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tax_engine import PropertyInput, calculate_taxes
from dataset_loader import CITY_PROFILES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "property_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.pkl")
DATA_PATH = "data/cleaned_data.csv"
TAX_RULES_DIR = "tax_rules"

CURRENT_YEAR = 2026

# ──────────────────────────────────────────────────────────────────────────
# Page config + styling
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Indian Property Valuation & Tax System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --saffron: #FF8C42;
    --indigo: #1B3A6B;
    --leaf: #2E7D5B;
    --sand: #FBF7EF;
    --ink: #1F2330;
}
.main { background-color: var(--sand); }
h1, h2, h3 { color: var(--indigo); font-family: 'Georgia', serif; }
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 10px rgba(27,58,107,0.08);
    border-left: 5px solid var(--saffron);
    color: #1F2330 !important;
}
.metric-card h3, .metric-card h4 {
    color: #1B3A6B !important;
    margin: 0.2rem 0 0 0;
}
.tax-card {
    background: white;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-left: 4px solid var(--leaf);
    margin-bottom: 0.6rem;
    color: #1F2330 !important;
}
.tax-card h4 {
    color: #1B3A6B !important;
    margin: 0.2rem 0 0 0;
}
.disclaimer-box {
    background: #FFF4E5;
    border: 1px solid #FFD8A8;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #6B4400;
}
.stButton>button {
    background-color: var(--indigo);
    color: white;
    border-radius: 8px;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: var(--saffron);
    color: var(--ink);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Cached resource loading
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_features():
    try:
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, features
    except FileNotFoundError:
        return None, None


@st.cache_data(show_spinner=False)
def load_clean_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_city_state_price_means():
    """Computes real mean market_price per city/state from the RAW dataset
    (before city/state columns were dropped during preprocessing), matching
    exactly how city_encoded/state_encoded were built for model training."""
    raw_path = "data/india_property_data.csv"
    if not os.path.exists(raw_path):
        return {}, {}, 8000000.0
    raw = pd.read_csv(raw_path)
    city_mean = raw.groupby("city")["market_price"].mean().to_dict()
    state_mean = raw.groupby("state")["market_price"].mean().to_dict()
    global_mean = float(raw["market_price"].mean())
    return city_mean, state_mean, global_mean


@st.cache_data(show_spinner=False)
def load_tax_rule_files():
    files = {}
    if os.path.isdir(TAX_RULES_DIR):
        for fname in sorted(os.listdir(TAX_RULES_DIR)):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(TAX_RULES_DIR, fname), "r", encoding="utf-8") as f:
                        files[fname] = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not parse {fname}: {e}")
    return files


model, feature_names = load_model_and_features()
clean_df = load_clean_data()
tax_rule_files = load_tax_rule_files()
CITY_MEAN_PRICE, STATE_MEAN_PRICE, GLOBAL_MEAN_PRICE = load_city_state_price_means()

STATE_CITY_MAP = {}
for city, profile in CITY_PROFILES.items():
    STATE_CITY_MAP.setdefault(profile["state"], []).append(city)
ALL_STATES = sorted(STATE_CITY_MAP.keys())

# Any Indian state/UT not in our dataset's city list still needs to be
# selectable (the tax engine has a generic fallback for them).
OTHER_INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Himachal Pradesh", "Jharkhand", "Kerala", "Madhya Pradesh", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal", "Jammu and Kashmir", "Ladakh", "Puducherry", "Andaman and Nicobar Islands",
    "Dadra and Nagar Haveli and Daman and Diu", "Lakshadweep",
]
for s in OTHER_INDIAN_STATES:
    ALL_STATES_SET = set(ALL_STATES)
    if s not in ALL_STATES_SET:
        ALL_STATES.append(s)
ALL_STATES = sorted(set(ALL_STATES))

PROPERTY_TYPES = ["Apartment", "Villa", "Independent House", "Plot", "Office", "Shop", "Warehouse"]
PROPERTY_CATEGORY_MAP = {
    "Apartment": "Residential", "Villa": "Residential", "Independent House": "Residential",
    "Plot": "Land", "Office": "Commercial", "Shop": "Commercial", "Warehouse": "Industrial",
}
FACING_DIRECTIONS = ["North", "South", "East", "West", "North-East", "North-West", "South-East", "South-West"]
CONSTRUCTION_QUALITY = ["Basic", "Standard", "Premium", "Luxury"]

# ──────────────────────────────────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


# ──────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────
def _safe_int(value, default: int = 0) -> int:
    """Safely coerce a value (possibly a string from a text_input) to int."""
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def fmt_inr(value: float) -> str:
    """Format a number in Indian comma style (e.g. 1,23,45,678)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    is_negative = value < 0
    value = abs(value)
    s = f"{value:,.0f}"
    # Convert standard comma grouping to Indian grouping
    if "," in s:
        s = s.replace(",", "")
        if len(s) > 3:
            last3 = s[-3:]
            rest = s[:-3]
            groups = []
            while len(rest) > 2:
                groups.insert(0, rest[-2:])
                rest = rest[:-2]
            if rest:
                groups.insert(0, rest)
            s = ",".join(groups) + "," + last3
    sign = "-" if is_negative else ""
    return f"{sign}\u20b9{s}"


def build_feature_row(inputs: dict, feature_names: list) -> pd.DataFrame:
    """Build a single-row dataframe matching the model's training features,
    filling any feature not directly provided with a sensible default (0)."""
    row = {f: 0 for f in feature_names}

    direct_map = {
        "pin_code": _safe_int(inputs.get("pin_code", 110001), default=110001),
        "latitude": inputs.get("latitude", 28.61),
        "longitude": inputs.get("longitude", 77.20),
        "built_up_area_sqft": inputs.get("built_up_area_sqft", 1000),
        "carpet_area_sqft": inputs.get("carpet_area_sqft", 800),
        "plot_area_sqft": inputs.get("plot_area_sqft", 1000),
        "construction_year": inputs.get("construction_year", 2015),
        "property_age": CURRENT_YEAR - inputs.get("construction_year", 2015),
        "num_floors": inputs.get("num_floors", 1),
        "floor_number": inputs.get("floor_number", 0),
        "bedrooms": inputs.get("bedrooms", 2),
        "bathrooms": inputs.get("bathrooms", 2),
        "parking": int(inputs.get("parking", True)),
        "road_width_ft": inputs.get("road_width_ft", 30),
        "nearby_metro": int(inputs.get("nearby_metro", False)),
        "nearby_school": int(inputs.get("nearby_school", True)),
        "nearby_hospital": int(inputs.get("nearby_hospital", True)),
        "nearby_market": int(inputs.get("nearby_market", True)),
        "nearby_airport": int(inputs.get("nearby_airport", False)),
        "nearby_railway_station": int(inputs.get("nearby_railway_station", False)),
        "corner_plot": int(inputs.get("corner_plot", False)),
        "lift_available": int(inputs.get("lift_available", True)),
        "swimming_pool": int(inputs.get("swimming_pool", False)),
        "solar_panels": int(inputs.get("solar_panels", False)),
        "rainwater_harvesting": int(inputs.get("rainwater_harvesting", False)),
        "green_certified": int(inputs.get("green_certified", False)),
        "population_density": inputs.get("population_density", 10000),
        "rental_yield_pct": inputs.get("rental_yield_pct", 3.0),
        "circle_rate_value": inputs.get("circle_rate_value", 0),
        "govt_valuation": inputs.get("govt_valuation", 0),
    }
    for k, v in direct_map.items():
        if k in row:
            row[k] = v

    city = inputs.get("city", "Delhi")
    state = inputs.get("state", "Delhi")
    city_mean_lookup = CITY_MEAN_PRICE.get(city)
    state_mean_lookup = STATE_MEAN_PRICE.get(state)
    if "city_encoded" in row:
        row["city_encoded"] = city_mean_lookup if city_mean_lookup is not None else GLOBAL_MEAN_PRICE
    if "state_encoded" in row:
        row["state_encoded"] = state_mean_lookup if state_mean_lookup is not None else GLOBAL_MEAN_PRICE

    onehot_targets = {
        f"property_type_{inputs.get('property_type', 'Apartment')}": 1,
        f"property_category_{inputs.get('property_category', 'Residential')}": 1,
        f"facing_direction_{inputs.get('facing_direction', 'East')}": 1,
        f"construction_quality_{inputs.get('construction_quality', 'Standard')}": 1,
        f"occupancy_{inputs.get('occupancy', 'Owner Occupied')}": 1,
        f"zone_{inputs.get('zone', 'Central')}": 1,
    }
    for col, val in onehot_targets.items():
        if col in row:
            row[col] = val

    if "total_rooms" in row:
        row["total_rooms"] = direct_map["bedrooms"] + direct_map["bathrooms"]
    if "amenity_score" in row:
        row["amenity_score"] = sum([
            direct_map["parking"], direct_map["nearby_metro"], direct_map["nearby_school"],
            direct_map["nearby_hospital"], direct_map["nearby_market"], direct_map["lift_available"],
            direct_map["swimming_pool"], direct_map["solar_panels"], direct_map["rainwater_harvesting"],
            direct_map["green_certified"],
        ])

    return pd.DataFrame([row])[feature_names]


def predict_price(inputs: dict):
    """Returns (predicted_price, lower_bound, upper_bound) or (None, None, None)."""
    if model is None or feature_names is None:
        return None, None, None
    X_row = build_feature_row(inputs, feature_names)
    pred_log = model.predict(X_row)[0]
    pred_price = float(np.expm1(pred_log))

    # Practical confidence interval: the trained model's overall test-set MAPE
    # (~9-10%, see outputs/evaluation.png) is used as a representative margin.
    margin = 0.10
    lower = pred_price * (1 - margin)
    upper = pred_price * (1 + margin)
    return pred_price, lower, upper


def calculate_mortgage(principal: float, annual_rate_pct: float, years: int):
    """Standard EMI calculation."""
    if principal <= 0 or years <= 0:
        return 0.0, 0.0, 0.0
    monthly_rate = annual_rate_pct / 12 / 100
    n_months = years * 12
    if monthly_rate == 0:
        emi = principal / n_months
    else:
        emi = principal * monthly_rate * (1 + monthly_rate) ** n_months / ((1 + monthly_rate) ** n_months - 1)
    total_payment = emi * n_months
    total_interest = total_payment - principal
    return emi, total_payment, total_interest


def generate_recommendations(prop_inputs: dict, tax_result, predicted_price: float) -> list:
    """Rule-based AI recommendations based on computed tax + property attributes."""
    recs = []

    if tax_result.rebates_applied:
        for k in tax_result.rebates_applied:
            label = k.replace("_", " ").title()
            recs.append(f"You're already benefiting from a **{label}** -- make sure to keep the supporting documents handy when filing.")

    if not prop_inputs.get("has_solar_panels") and not prop_inputs.get("has_rainwater_harvesting"):
        recs.append("Installing **solar panels** or a **rainwater harvesting system** could make you eligible for a municipal green rebate in cities like Pune (PMC), reducing your annual property tax by up to 10%.")

    if prop_inputs.get("construction_year", 2015) and (CURRENT_YEAR - prop_inputs.get("construction_year", 2015)) > 25:
        recs.append("Your property qualifies for an **age-based depreciation rebate** in most Unit-Area and Capital-Value tax systems -- confirm this has been correctly applied on your municipal bill.")

    if prop_inputs.get("property_category") == "Residential" and prop_inputs.get("occupancy") == "Rented":
        recs.append("Since this property is rented, you may convert it to **owner-occupied** status (if applicable) to access lower self-occupancy tax rates in cities like Mumbai, Pune, and Bengaluru.")

    if prop_inputs.get("property_category") == "Commercial":
        recs.append("Commercial properties attract a significantly higher property tax rate (often 1.5-2x residential) -- factor this into your ROI calculations before purchase.")

    circle_rate = prop_inputs.get("circle_rate_value", 0)
    if circle_rate and predicted_price and circle_rate > predicted_price * 0.85:
        recs.append("The **circle rate is close to your predicted market value** -- this is common in high-demand areas and means your stamp duty liability is near the maximum for this property size.")

    rental_yield = prop_inputs.get("rental_yield_pct", 3.0)
    if rental_yield and rental_yield >= 4.0:
        recs.append(f"This property type/location shows a healthy estimated rental yield of **{rental_yield:.1f}%**, above the typical Indian residential average of 2-3.5%.")

    if tax_result.is_fallback_used:
        recs.append("This state doesn't yet have a dedicated researched tax ruleset in this build -- the figures shown use a **generic estimation model** and should be cross-checked against your local municipal corporation's published rates.")

    market_avg = CITY_PROFILES.get(prop_inputs.get("city", ""), {}).get("avg_price_psqft")
    if market_avg and prop_inputs.get("built_up_area_sqft"):
        implied_psqft = predicted_price / max(prop_inputs.get("built_up_area_sqft", 1), 1)
        if implied_psqft > market_avg * 1.15:
            recs.append(f"This property's predicted price/sqft (\u20b9{implied_psqft:,.0f}) is **above the {prop_inputs.get('city')} city average** (\u20b9{market_avg:,.0f}) -- likely reflecting premium location, quality, or amenities.")
        elif implied_psqft < market_avg * 0.85:
            recs.append(f"This property's predicted price/sqft (\u20b9{implied_psqft:,.0f}) is **below the {prop_inputs.get('city')} city average** (\u20b9{market_avg:,.0f}) -- could represent good value or signal an aging/under-amenitized property.")

    if not recs:
        recs.append("No specific tax-saving opportunities were identified for this configuration -- your setup looks standard for its category.")

    return recs


def investment_score(prop_inputs: dict, tax_result, predicted_price: float) -> int:
    """A simple 0-100 composite score for demo purposes."""
    score = 50
    rental_yield = prop_inputs.get("rental_yield_pct", 3.0) or 3.0
    score += min(20, (rental_yield - 2.5) * 8)

    city = prop_inputs.get("city", "")
    tier = CITY_PROFILES.get(city, {}).get("tier", "tier2")
    score += 12 if tier == "metro" else 4

    age = CURRENT_YEAR - prop_inputs.get("construction_year", 2015)
    score -= min(15, age * 0.4)

    amenity_flags = [
        prop_inputs.get("nearby_metro"), prop_inputs.get("has_solar_panels"),
        prop_inputs.get("swimming_pool"), prop_inputs.get("green_certified"),
    ]
    score += sum(bool(x) for x in amenity_flags) * 2.5

    if tax_result.is_fallback_used:
        score -= 3  # less certainty in tax modeling for this location

    return int(np.clip(score, 0, 100))


def risk_score(prop_inputs: dict, tax_result) -> int:
    """A simple 0-100 risk score (higher = riskier), inverse-ish of investment score."""
    risk = 30
    age = CURRENT_YEAR - prop_inputs.get("construction_year", 2015)
    risk += min(25, age * 0.6)
    if prop_inputs.get("occupancy") == "Vacant":
        risk += 10
    if prop_inputs.get("property_category") == "Land":
        risk += 8
    if tax_result.is_fallback_used:
        risk += 7
    return int(np.clip(risk, 0, 100))


# ──────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.title("🏠 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏡 Valuation & Tax Calculator",
        "📊 Model Comparison",
        "📈 Prediction History",
        "🏙️ City & State Comparison",
        "ℹ️ About the Dataset & Tax Rules",
    ],
)

st.sidebar.markdown("---")
st.sidebar.session_dark = st.sidebar.toggle("🌙 Dark mode (preview)", value=st.session_state.dark_mode)
st.session_state.dark_mode = st.sidebar.session_dark
if st.session_state.dark_mode:
    st.markdown(
        "<style>.main{background-color:#14171F !important;} h1,h2,h3{color:#FF8C42 !important;} "
        ".metric-card,.tax-card{background:#1F2330 !important;color:#EAEAEA;}</style>",
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built as an end-to-end Property Valuation + Indian Property Tax Calculator. "
    "Tax figures for Delhi, Maharashtra (Mumbai/Pune) and Karnataka (Bengaluru) are "
    "reconstructed from public 2025-26 municipal guides; other states use a labelled "
    "generic estimate. Always verify with your local municipal corporation."
)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: Valuation & Tax Calculator
# ══════════════════════════════════════════════════════════════════════════
if page == "🏡 Valuation & Tax Calculator":
    st.title("🏠 Smart Indian Property Valuation & Tax Assessment")
    st.caption("Predict your property's market value, then see every applicable Indian property tax, instantly.")

    if model is None:
        st.error(
            "No trained model found. Run `python 2_train.py` first (after `python preprocess.py`) "
            "to generate `models/property_model.pkl`."
        )
        st.stop()

    with st.form("property_form"):
        st.subheader("📍 Location")
        c1, c2, c3 = st.columns(3)
        with c1:
            state = st.selectbox("State", ALL_STATES, index=ALL_STATES.index("Delhi") if "Delhi" in ALL_STATES else 0)
        with c2:
            cities_for_state = STATE_CITY_MAP.get(state, [])
            if cities_for_state:
                city = st.selectbox("City", cities_for_state)
            else:
                city = st.text_input("City (not in curated dataset -- generic estimate will be used)", value=state)
        with c3:
            pin_code = st.text_input("PIN Code", value="110001")

        c4, c5 = st.columns(2)
        with c4:
            municipality = st.text_input("Municipality / Corporation", value=f"{city} Municipal Corporation")
        with c5:
            ward = st.text_input("Ward (optional)", value="")
        zone = st.selectbox("Zone", ["Central", "North", "South", "East", "West"])

        st.markdown("---")
        st.subheader("🏢 Property Details")
        c6, c7, c8 = st.columns(3)
        with c6:
            property_type = st.selectbox("Property Type", PROPERTY_TYPES)
        with c7:
            property_category = st.selectbox(
                "Property Category (use)",
                ["Residential", "Commercial", "Industrial", "Land"],
                index=["Residential", "Commercial", "Industrial", "Land"].index(
                    PROPERTY_CATEGORY_MAP.get(property_type, "Residential")
                ),
            )
        with c8:
            occupancy = st.selectbox("Occupancy", ["Owner Occupied", "Rented", "Vacant"])

        c9, c10, c11 = st.columns(3)
        with c9:
            built_up_area = st.number_input("Built-up Area (sqft)", min_value=0, value=1200, step=50)
        with c10:
            carpet_area = st.number_input("Carpet Area (sqft)", min_value=0, value=1000, step=50)
        with c11:
            plot_area = st.number_input("Plot Area (sqft)", min_value=0, value=1500, step=50)

        c12, c13, c14 = st.columns(3)
        with c12:
            construction_year = st.number_input("Construction Year", min_value=1950, max_value=CURRENT_YEAR, value=2015)
        with c13:
            num_floors = st.number_input("Number of Floors", min_value=1, max_value=50, value=1)
        with c14:
            floor_number = st.number_input("Floor Number (if apartment)", min_value=0, max_value=80, value=2)

        c15, c16, c17 = st.columns(3)
        with c15:
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=12, value=3)
        with c16:
            bathrooms = st.number_input("Bathrooms", min_value=0, max_value=12, value=2)
        with c17:
            construction_quality = st.selectbox("Construction Quality", CONSTRUCTION_QUALITY, index=1)

        c18, c19 = st.columns(2)
        with c18:
            structure_type = st.selectbox("Structure Type", ["RCC", "Load Bearing", "Non-RCC", "Chawl"])
        with c19:
            facing_direction = st.selectbox("Facing Direction", FACING_DIRECTIONS)

        st.markdown("---")
        st.subheader("🛣️ Surroundings & Amenities")
        c20, c21, c22, c23 = st.columns(4)
        with c20:
            parking = st.checkbox("Parking", value=True)
            corner_plot = st.checkbox("Corner Plot", value=False)
        with c21:
            lift_available = st.checkbox("Lift Available", value=True)
            swimming_pool = st.checkbox("Swimming Pool", value=False)
        with c22:
            solar_panels = st.checkbox("Solar Panels", value=False)
            rainwater_harvesting = st.checkbox("Rainwater Harvesting", value=False)
        with c23:
            green_certified = st.checkbox("Green Building Certified", value=False)
            nearby_metro = st.checkbox("Near Metro", value=False)

        c24, c25, c26 = st.columns(3)
        with c24:
            nearby_school = st.checkbox("Near School", value=True)
            nearby_hospital = st.checkbox("Near Hospital", value=True)
        with c25:
            nearby_market = st.checkbox("Near Market", value=True)
            nearby_airport = st.checkbox("Near Airport", value=False)
        with c26:
            nearby_railway = st.checkbox("Near Railway Station", value=False)
            road_width_ft = st.number_input("Road Width (ft)", min_value=4, max_value=150, value=30)

        st.markdown("---")
        st.subheader("💰 Valuation Inputs")
        c27, c28 = st.columns(2)
        with c27:
            circle_rate_value = st.number_input(
                "Circle Rate / Guideline Value (\u20b9, optional -- improves stamp duty accuracy)",
                min_value=0, value=0, step=50000,
            )
        with c28:
            govt_guideline_value = st.number_input("Government Valuation (\u20b9, optional)", min_value=0, value=0, step=50000)

        st.markdown("---")
        st.subheader("👤 Buyer & Tax Profile")
        c29, c30, c31 = st.columns(3)
        with c29:
            buyer_gender = st.selectbox("Buyer Category (for stamp duty)", ["Male", "Female", "Joint"])
        with c30:
            is_senior_citizen = st.checkbox("Senior Citizen Owner", value=False)
            is_woman_owner = st.checkbox("Woman Owner", value=False)
        with c31:
            is_disabled_or_exserviceman = st.checkbox("Disabled / Ex-Servicemen", value=False)
            pays_in_lump_sum = st.checkbox("Pays Tax in Lump Sum (early-bird rebate)", value=True)

        c32, c33, c34 = st.columns(3)
        with c32:
            zone_or_category_override = st.text_input(
                "Tax Zone/Category Override (optional, e.g. 'A'-'H' for Delhi, 'A'-'F' for Bengaluru)",
                value="",
            )
        with c33:
            holding_period_months = st.number_input("Holding Period (months, for capital gains)", min_value=0, value=36)
        with c34:
            annual_rental_income = st.number_input("Annual Rental Income (\u20b9, if rented)", min_value=0, value=0, step=10000)

        c35, c36 = st.columns(2)
        with c35:
            purchase_price = st.number_input("Original Purchase Price (\u20b9, for capital gains; 0 = auto-estimate)", min_value=0, value=0, step=50000)
        with c36:
            inflation_rate_pct = st.slider("Inflation Rate for Future Projections (%)", 2.0, 12.0, 6.0, 0.5)

        st.markdown("---")
        st.subheader("🏦 Mortgage Calculator")
        c37, c38, c39 = st.columns(3)
        with c37:
            down_payment_pct = st.slider("Down Payment (%)", 0, 90, 20)
        with c38:
            loan_interest_rate = st.slider("Loan Interest Rate (% p.a.)", 6.0, 14.0, 8.5, 0.1)
        with c39:
            loan_tenure_years = st.slider("Loan Tenure (years)", 5, 30, 20)

        submitted = st.form_submit_button("🔮 Predict Value & Calculate Taxes", use_container_width=True)

    if submitted:
        profile = CITY_PROFILES.get(city, {})
        prop_inputs = {
            "state": state, "city": city, "pin_code": pin_code, "municipality": municipality,
            "ward": ward, "zone": zone,
            "property_type": property_type, "property_category": property_category, "occupancy": occupancy,
            "built_up_area_sqft": built_up_area, "carpet_area_sqft": carpet_area, "plot_area_sqft": plot_area,
            "construction_year": construction_year, "num_floors": num_floors, "floor_number": floor_number,
            "bedrooms": bedrooms, "bathrooms": bathrooms, "construction_quality": construction_quality,
            "structure_type": structure_type, "facing_direction": facing_direction,
            "parking": parking, "corner_plot": corner_plot, "lift_available": lift_available,
            "swimming_pool": swimming_pool, "solar_panels": solar_panels, "has_solar_panels": solar_panels,
            "rainwater_harvesting": rainwater_harvesting, "has_rainwater_harvesting": rainwater_harvesting,
            "green_certified": green_certified, "nearby_metro": nearby_metro, "nearby_school": nearby_school,
            "nearby_hospital": nearby_hospital, "nearby_market": nearby_market, "nearby_airport": nearby_airport,
            "nearby_railway_station": nearby_railway, "road_width_ft": road_width_ft,
            "circle_rate_value": circle_rate_value, "govt_valuation": govt_guideline_value,
            "latitude": profile.get("lat", 20.5937), "longitude": profile.get("lon", 78.9629),
            "population_density": 10000, "rental_yield_pct": 3.0,
        }

        predicted_price, lower_bound, upper_bound = predict_price(prop_inputs)

        if predicted_price is None:
            st.error("Prediction failed -- the model could not be loaded.")
            st.stop()

        tax_input = PropertyInput(
            state=state, city=city, property_type=property_type, property_category=property_category,
            built_up_area_sqft=built_up_area, carpet_area_sqft=carpet_area, plot_area_sqft=plot_area,
            construction_year=construction_year, market_value=predicted_price,
            circle_rate_value=circle_rate_value or None, govt_guideline_value=govt_guideline_value or None,
            occupancy=occupancy, buyer_gender=buyer_gender, construction_quality=construction_quality,
            structure_type=structure_type, is_senior_citizen=is_senior_citizen, is_woman_owner=is_woman_owner,
            is_disabled_or_exserviceman=is_disabled_or_exserviceman, has_solar_panels=solar_panels,
            has_rainwater_harvesting=rainwater_harvesting, is_green_certified=green_certified,
            zone_or_category=zone_or_category_override.strip() or None, pays_in_lump_sum=pays_in_lump_sum,
            annual_rental_income=annual_rental_income, holding_period_months=holding_period_months,
            purchase_price=purchase_price or None, inflation_rate_pct=inflation_rate_pct,
        )
        tax_result = calculate_taxes(tax_input)

        loan_amount = predicted_price * (1 - down_payment_pct / 100)
        emi, total_payment, total_interest = calculate_mortgage(loan_amount, loan_interest_rate, loan_tenure_years)

        # Persist to history
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "city": city, "state": state, "property_type": property_type,
            "area_sqft": built_up_area, "predicted_price": predicted_price,
            "annual_property_tax": tax_result.annual_property_tax,
            "stamp_duty": tax_result.stamp_duty, "total_purchase_cost": tax_result.total_purchase_cost,
            "emi": emi,
        })

        st.success("Prediction complete!")

        # ── Valuation summary ──────────────────────────────────────────
        st.subheader("💎 Predicted Market Value")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='metric-card'><h3>{fmt_inr(predicted_price)}</h3>Predicted Market Value</div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-card'><h3>{fmt_inr(lower_bound)} \u2013 {fmt_inr(upper_bound)}</h3>90% Confidence Interval</div>", unsafe_allow_html=True)
        with m3:
            psqft = predicted_price / max(built_up_area or plot_area, 1)
            st.markdown(f"<div class='metric-card'><h3>{fmt_inr(psqft)}/sqft</h3>Implied Price per Sqft</div>", unsafe_allow_html=True)

        if tax_result.is_fallback_used:
            st.markdown(
                f"<div class='disclaimer-box'>⚠️ <b>{state}</b> does not yet have a dedicated researched tax "
                f"ruleset in this build. The figures below use a generic estimation model. "
                f"{tax_result.disclaimer}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='disclaimer-box'>ℹ️ Tax figures use the <b>{tax_result.rule_source}</b> ruleset. "
                f"{tax_result.disclaimer}</div>",
                unsafe_allow_html=True,
            )

        # ── Tax breakdown ───────────────────────────────────────────────
        st.subheader("🧾 Property Tax & Government Charges")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.markdown(f"<div class='tax-card'><b>Annual Property Tax</b><br><h4>{fmt_inr(tax_result.annual_property_tax)}</h4></div>", unsafe_allow_html=True)
        with t2:
            st.markdown(f"<div class='tax-card'><b>Monthly Property Tax</b><br><h4>{fmt_inr(tax_result.monthly_property_tax)}</h4></div>", unsafe_allow_html=True)
        with t3:
            st.markdown(f"<div class='tax-card'><b>Recurring Annual Total</b><br><h4>{fmt_inr(tax_result.recurring_annual_total)}</h4></div>", unsafe_allow_html=True)
        with t4:
            st.markdown(f"<div class='tax-card'><b>Total Rebates Applied</b><br><h4>{fmt_inr(tax_result.total_rebate_amount)}</h4></div>", unsafe_allow_html=True)

        with st.expander("📋 Full Recurring Charges Breakdown"):
            recurring_table = pd.DataFrame([
                {"Charge": "Annual Property Tax", "Amount (\u20b9)": tax_result.annual_property_tax},
                {"Charge": "Water Tax (est.)", "Amount (\u20b9)": tax_result.water_tax},
                {"Charge": "Sewerage Tax (est.)", "Amount (\u20b9)": tax_result.sewerage_tax},
                {"Charge": "Solid Waste Management Charge", "Amount (\u20b9)": tax_result.solid_waste_management_charge},
                {"Charge": "Education Cess", "Amount (\u20b9)": tax_result.education_cess},
                {"Charge": "Fire Cess", "Amount (\u20b9)": tax_result.fire_cess},
                {"Charge": "Health Cess", "Amount (\u20b9)": tax_result.health_cess},
                {"Charge": "Urban Development Cess", "Amount (\u20b9)": tax_result.urban_development_cess},
                {"Charge": "Luxury Property Tax (if applicable)", "Amount (\u20b9)": tax_result.luxury_property_tax},
                {"Charge": "Green Rebate (deducted)", "Amount (\u20b9)": -tax_result.green_tax_rebate},
            ])
            st.dataframe(recurring_table, use_container_width=True, hide_index=True)
            if tax_result.property_tax_breakdown:
                st.caption("Property tax calculation detail:")
                st.json(tax_result.property_tax_breakdown)
            if tax_result.rebates_applied:
                st.caption("Rebates applied to property tax:")
                st.json(tax_result.rebates_applied)

        st.subheader("🏛️ One-Time Government Charges (At Purchase)")
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            st.markdown(f"<div class='tax-card'><b>Stamp Duty</b><br><h4>{fmt_inr(tax_result.stamp_duty)}</h4></div>", unsafe_allow_html=True)
        with o2:
            st.markdown(f"<div class='tax-card'><b>Registration Charges</b><br><h4>{fmt_inr(tax_result.registration_charges)}</h4></div>", unsafe_allow_html=True)
        with o3:
            st.markdown(f"<div class='tax-card'><b>Mutation Charges</b><br><h4>{fmt_inr(tax_result.mutation_charges)}</h4></div>", unsafe_allow_html=True)
        with o4:
            st.markdown(f"<div class='tax-card'><b>One-Time Total</b><br><h4>{fmt_inr(tax_result.one_time_total)}</h4></div>", unsafe_allow_html=True)

        st.markdown(f"### 💰 Total Purchase Cost: {fmt_inr(tax_result.total_purchase_cost)}")
        st.caption("Market value + stamp duty + registration + mutation charges.")

        # ── Capital gains & rental income ───────────────────────────────
        st.subheader("📈 Capital Gains & Rental Income Tax (Estimates)")
        cg1, cg2 = st.columns(2)
        with cg1:
            st.markdown(f"<div class='tax-card'><b>Estimated Capital Gains Tax</b><br><h4>{fmt_inr(tax_result.capital_gains_tax_estimate)}</h4>"
                        f"<small>{tax_result.capital_gains_method}</small></div>", unsafe_allow_html=True)
        with cg2:
            st.markdown(f"<div class='tax-card'><b>Estimated Rental Income Tax</b><br><h4>{fmt_inr(tax_result.rental_income_tax_estimate)}</h4>"
                        f"<small>Based on declared annual rental income, after 30% standard deduction.</small></div>", unsafe_allow_html=True)

        # ── Future projections ──────────────────────────────────────────
        st.subheader("🔮 Future Tax Projection")
        proj_cols = st.columns(3)
        for idx, (key, label) in enumerate([("5_year", "5-Year"), ("10_year", "10-Year"), ("15_year", "15-Year")]):
            data = tax_result.projections.get(key, {})
            with proj_cols[idx]:
                st.markdown(
                    f"<div class='tax-card'><b>{label} Projection</b><br>"
                    f"Annual Tax: <b>{fmt_inr(data.get('projected_annual_tax', 0))}</b><br>"
                    f"Cumulative Paid: <b>{fmt_inr(data.get('cumulative_tax_paid', 0))}</b></div>",
                    unsafe_allow_html=True,
                )

        years = [0, 5, 10, 15]
        proj_values = [tax_result.recurring_annual_total] + [
            tax_result.projections.get(f"{y}_year", {}).get("projected_annual_tax", 0) for y in (5, 10, 15)
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(years, proj_values, marker="o", color="#1B3A6B", linewidth=2.5)
        ax.fill_between(years, proj_values, alpha=0.1, color="#1B3A6B")
        ax.set_xlabel("Years from now")
        ax.set_ylabel("Projected Annual Recurring Tax (\u20b9)")
        ax.set_title(f"Tax Projection at {inflation_rate_pct}% Inflation")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        # ── Pie + bar chart of tax breakdown ────────────────────────────
        st.subheader("📊 Tax Breakdown Visualizations")
        v1, v2 = st.columns(2)
        with v1:
            labels = ["Property Tax", "Water Tax", "Sewerage Tax", "Solid Waste", "Other Cess"]
            values = [
                tax_result.annual_property_tax, tax_result.water_tax, tax_result.sewerage_tax,
                tax_result.solid_waste_management_charge,
                tax_result.education_cess + tax_result.fire_cess + tax_result.health_cess + tax_result.urban_development_cess,
            ]
            values = [max(0, v) for v in values]
            if sum(values) > 0:
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                colors = ["#FF8C42", "#1B3A6B", "#2E7D5B", "#C9A227", "#8E5572"]
                ax2.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
                ax2.set_title("Recurring Tax Breakdown")
                st.pyplot(fig2)
                plt.close(fig2)
            else:
                st.info("No recurring tax breakdown to display (annual tax is zero, possibly due to an exemption).")
        with v2:
            cost_labels = ["Stamp Duty", "Registration", "Mutation"]
            cost_values = [tax_result.stamp_duty, tax_result.registration_charges, tax_result.mutation_charges]
            fig3, ax3 = plt.subplots(figsize=(5, 5))
            ax3.bar(cost_labels, cost_values, color=["#FF8C42", "#1B3A6B", "#2E7D5B"])
            ax3.set_ylabel("Amount (\u20b9)")
            ax3.set_title("One-Time Purchase Charges")
            for i, v in enumerate(cost_values):
                ax3.text(i, v, fmt_inr(v), ha="center", va="bottom", fontsize=8)
            st.pyplot(fig3)
            plt.close(fig3)

        # ── Mortgage ──────────────────────────────────────────────────
        st.subheader("🏦 Mortgage / Home Loan Summary")
        mg1, mg2, mg3, mg4 = st.columns(4)
        with mg1:
            st.markdown(f"<div class='metric-card'><h4>{fmt_inr(loan_amount)}</h4>Loan Amount</div>", unsafe_allow_html=True)
        with mg2:
            st.markdown(f"<div class='metric-card'><h4>{fmt_inr(emi)}</h4>Monthly EMI</div>", unsafe_allow_html=True)
        with mg3:
            st.markdown(f"<div class='metric-card'><h4>{fmt_inr(total_interest)}</h4>Total Interest Payable</div>", unsafe_allow_html=True)
        with mg4:
            st.markdown(f"<div class='metric-card'><h4>{fmt_inr(total_payment)}</h4>Total Repayment</div>", unsafe_allow_html=True)

        # ── AI Recommendations ─────────────────────────────────────────
        st.subheader("🤖 AI Recommendations")
        recs = generate_recommendations(prop_inputs, tax_result, predicted_price)
        for r in recs:
            st.markdown(f"- {r}")

        # ── Investment & Risk Score ────────────────────────────────────
        st.subheader("🎯 Investment & Risk Scores")
        inv_score = investment_score(prop_inputs, tax_result, predicted_price)
        rsk_score = risk_score(prop_inputs, tax_result)
        s1, s2 = st.columns(2)
        with s1:
            st.metric("Investment Score (0-100)", inv_score)
            st.progress(inv_score / 100)
        with s2:
            st.metric("Risk Score (0-100, lower is safer)", rsk_score)
            st.progress(rsk_score / 100)

        # ── Feature importance for this prediction's model ─────────────
        with st.expander("🔍 Model Feature Importance (Top 15)"):
            try:
                importances = pd.Series(model.feature_importances_, index=feature_names)
                top15 = importances.sort_values(ascending=False).head(15)
                fig4, ax4 = plt.subplots(figsize=(8, 6))
                top15.sort_values().plot(kind="barh", ax=ax4, color="#1B3A6B")
                ax4.set_xlabel("Importance Score")
                ax4.set_title("Top 15 Features Driving This Model")
                st.pyplot(fig4)
                plt.close(fig4)
            except Exception as e:
                st.warning(f"Feature importance unavailable: {e}")

        if tax_result.warnings:
            with st.expander("⚠️ Calculation Warnings"):
                for w in tax_result.warnings:
                    st.warning(w)

        # ── Export ───────────────────────────────────────────────────────
        st.subheader("📥 Export Report")
        export_df = pd.DataFrame([{
            "Timestamp": datetime.now().isoformat(),
            "State": state, "City": city, "Property Type": property_type,
            "Built-up Area (sqft)": built_up_area, "Predicted Market Value": predicted_price,
            "Annual Property Tax": tax_result.annual_property_tax,
            "Stamp Duty": tax_result.stamp_duty, "Registration Charges": tax_result.registration_charges,
            "Total Purchase Cost": tax_result.total_purchase_cost,
            "Capital Gains Tax Estimate": tax_result.capital_gains_tax_estimate,
            "Monthly EMI": emi,
        }])
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Summary as CSV", data=csv_bytes, file_name="property_valuation_report.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════
# PAGE: Model Comparison
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.caption("Performance of 12 regression models trained on the Indian property dataset.")

    results_path = "outputs/model_comparison_results.csv"
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path, index_col=0)
        st.dataframe(results_df, use_container_width=True)

        best = results_df.iloc[0]
        st.success(f"🏆 Best model: **{best['Model']}** with R\u00b2 = {best['R\u00b2 Score']:.4f}")

        img_cols = st.columns(2)
        with img_cols[0]:
            if os.path.exists("outputs/r2_comparison.png"):
                st.image("outputs/r2_comparison.png", caption="R\u00b2 Score Comparison")
        with img_cols[1]:
            if os.path.exists("outputs/rmse_comparison.png"):
                st.image("outputs/rmse_comparison.png", caption="RMSE Comparison")

        if os.path.exists("outputs/feature_importance_all.png"):
            st.image("outputs/feature_importance_all.png", caption="Feature Importance Across Tree-Based Models")
    else:
        st.warning("Run `python model_comparison.py` first to generate comparison results.")

    if os.path.exists("outputs/evaluation.png"):
        st.subheader("📈 XGBoost (Production Model) Evaluation")
        st.image("outputs/evaluation.png")


# ══════════════════════════════════════════════════════════════════════════
# PAGE: Prediction History
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Prediction History":
    st.title("📈 Prediction History")

    if not st.session_state.history:
        st.info("No predictions yet this session. Go to the Valuation & Tax Calculator page to make one.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(hist_df)), hist_df["predicted_price"], marker="o", color="#1B3A6B")
        ax.set_xlabel("Prediction #")
        ax.set_ylabel("Predicted Price (\u20b9)")
        ax.set_title("Predicted Price Across Session History")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full History as CSV", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv")

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PAGE: City & State Comparison
# ══════════════════════════════════════════════════════════════════════════
elif page == "🏙️ City & State Comparison":
    st.title("🏙️ City & State Comparison")
    st.caption("Compare predicted cost-of-ownership for the same property profile across two cities.")

    if model is None:
        st.error("No trained model found. Run `python 2_train.py` first.")
        st.stop()

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Property A")
        city_a = st.selectbox("City A", list(CITY_PROFILES.keys()), index=0, key="city_a")
    with colB:
        st.subheader("Property B")
        city_b = st.selectbox("City B", list(CITY_PROFILES.keys()), index=1, key="city_b")

    shared_area = st.number_input("Built-up Area (sqft) for both", min_value=200, value=1200, step=50)
    shared_type = st.selectbox("Property Type for both", PROPERTY_TYPES, index=0)
    shared_year = st.number_input("Construction Year for both", min_value=1950, max_value=CURRENT_YEAR, value=2015)

    if st.button("⚖️ Compare", use_container_width=True):
        comparison_rows = []
        for city in (city_a, city_b):
            profile = CITY_PROFILES.get(city, {})
            state = profile.get("state", "Delhi")
            inputs = {
                "state": state, "city": city, "property_type": shared_type,
                "property_category": PROPERTY_CATEGORY_MAP.get(shared_type, "Residential"),
                "occupancy": "Owner Occupied", "built_up_area_sqft": shared_area,
                "carpet_area_sqft": shared_area * 0.85, "plot_area_sqft": shared_area,
                "construction_year": shared_year, "num_floors": 1, "floor_number": 2,
                "bedrooms": 3, "bathrooms": 2, "construction_quality": "Standard",
                "structure_type": "RCC", "facing_direction": "East", "parking": True,
                "corner_plot": False, "lift_available": True, "swimming_pool": False,
                "solar_panels": False, "rainwater_harvesting": False, "green_certified": False,
                "nearby_metro": False, "nearby_school": True, "nearby_hospital": True,
                "nearby_market": True, "nearby_airport": False, "nearby_railway_station": False,
                "road_width_ft": 30, "circle_rate_value": 0, "govt_valuation": 0,
                "latitude": profile.get("lat", 20.59), "longitude": profile.get("lon", 78.96),
                "population_density": 10000, "rental_yield_pct": 3.0,
                "zone": "Central",
            }
            pred_price, lower, upper = predict_price(inputs)
            tax_input = PropertyInput(
                state=state, city=city, property_type=shared_type,
                property_category=PROPERTY_CATEGORY_MAP.get(shared_type, "Residential"),
                built_up_area_sqft=shared_area, construction_year=shared_year, market_value=pred_price,
                occupancy="Owner Occupied",
            )
            tax_result = calculate_taxes(tax_input)
            comparison_rows.append({
                "City": city, "State": state, "Predicted Value": pred_price,
                "Annual Property Tax": tax_result.annual_property_tax,
                "Stamp Duty": tax_result.stamp_duty, "Registration": tax_result.registration_charges,
                "Total Purchase Cost": tax_result.total_purchase_cost,
                "5yr Tax Projection": tax_result.projections.get("5_year", {}).get("cumulative_tax_paid", 0),
            })

        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(comp_df["City"], comp_df["Predicted Value"], color=["#FF8C42", "#1B3A6B"])
        axes[0].set_title("Predicted Market Value")
        axes[0].set_ylabel("\u20b9")
        axes[1].bar(comp_df["City"], comp_df["Total Purchase Cost"], color=["#FF8C42", "#1B3A6B"])
        axes[1].set_title("Total Purchase Cost (incl. taxes)")
        axes[1].set_ylabel("\u20b9")
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: About
# ══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About the Dataset & Tax Rules":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ### Dataset
    This application is built on a **synthetic-but-realistic Indian property dataset**.
    Individual rows are generated, not scraped from real listings -- but the generation
    is anchored to **real 2025-26 market statistics** (average price-per-sqft by city,
    circle-rate-to-market-value ratios) sourced from public real-estate market reports
    (Knight Frank, Magicbricks, 99acres, Square Yards, NoBroker, Statista).

    This hybrid approach exists because clean, row-level, consistently-schemed Indian
    property transaction datasets are not freely available across municipalities --
    most government/municipal data is published as PDF bulletins, not structured CSVs.
    """)

    st.markdown("### Tax Rule Coverage")
    rule_summary = []
    for fname, data in tax_rule_files.items():
        if fname == "_default.json":
            continue
        rule_summary.append({
            "File": fname,
            "State": data.get("state", "?"),
            "System": data.get("system", "?"),
            "Financial Year": data.get("financial_year", "?"),
        })
    if rule_summary:
        st.dataframe(pd.DataFrame(rule_summary), use_container_width=True, hide_index=True)

    st.markdown("""
    For any state without a dedicated ruleset above, the engine falls back to a
    **generic, clearly-labelled estimate** (`tax_rules/_default.json`) so the app
    never crashes -- but those figures should not be treated as authoritative.
    """)

    with st.expander("📄 View raw tax rule JSON files"):
        selected_file = st.selectbox("Choose a file", list(tax_rule_files.keys()))
        st.json(tax_rule_files[selected_file])

    st.markdown("### Model Info")
    if feature_names:
        st.write(f"Production model: **XGBoost Regressor** trained on {len(feature_names)} features.")
    st.caption(
        "This is an academic / portfolio project. Tax and valuation figures are estimates "
        "for educational purposes and must be verified against official municipal and "
        "state government sources before any real financial decision."
    )
