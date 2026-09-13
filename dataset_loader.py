"""
dataset_loader.py
==================
Builds the Indian property dataset used to train and evaluate the
valuation models.

WHY SYNTHETIC + REAL ANCHORS (read this before changing the numbers):
-----------------------------------------------------------------------
Public, row-level Indian property transaction datasets with consistent
schemas across cities are not freely available (most municipal /
registration datasets are PDF bulletins, not clean CSVs). Building a
*believable* dataset for this project therefore uses a hybrid approach:

  1. REAL ANCHORS — average price-per-sqft by city/locality-tier, and
     circle-rate-to-market-value ratios, are taken from public 2025-26
     real-estate market reports (Knight Frank, Magicbricks, 99acres,
     Square Yards, NoBroker, Statista). These anchors are hard-coded
     in CITY_PROFILES below and are cited in tax_engine documentation
     and the project README.
  2. SYNTHETIC GENERATION — individual property rows (exact area,
     floor, age, amenities, etc.) are generated using statistical
     distributions (normal/lognormal/choice) *centered* on those real
     anchors, with engineered noise and realistic correlations
     (e.g. older properties -> lower price/sqft, metro proximity ->
     price premium). This produces a dataset that is large, varied,
     and "real-shaped" without us fabricating specific transactions
     and presenting them as real public records.

This is disclosed explicitly in the Streamlit UI ("About the Dataset"
tab) and in the README — nothing here pretends to be scraped
government data.
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── City profiles: real-world anchors (2025-26) ───────────────────────────
# avg_price_psqft: average residential market price per sqft (Rs), blended
#   across affordable/mid/luxury segments based on public 2025-26 reports.
# circle_rate_ratio: typical circle (guideline) rate as a fraction of
#   average market price (varies by state/city; circle rates are usually
#   below market price).
# state: for tax-rule lookup.
# tier: metro / tier2 / tier3, used for feature engineering.
CITY_PROFILES = {

    # ── Delhi / NCR ─────────────────────────────────────────────────────
    "Delhi":           {"state": "Delhi",             "avg_price_psqft": 14500, "circle_rate_ratio": 0.55, "tier": "metro",  "lat": 28.6139, "lon": 77.2090},
    "New Delhi":       {"state": "Delhi",             "avg_price_psqft": 18000, "circle_rate_ratio": 0.55, "tier": "metro",  "lat": 28.6328, "lon": 77.2197},
    "Noida":           {"state": "Uttar Pradesh",     "avg_price_psqft": 8600,  "circle_rate_ratio": 0.60, "tier": "metro",  "lat": 28.5355, "lon": 77.3910},
    "Greater Noida":   {"state": "Uttar Pradesh",     "avg_price_psqft": 6200,  "circle_rate_ratio": 0.58, "tier": "metro",  "lat": 28.4744, "lon": 77.5040},
    "Gurgaon":         {"state": "Haryana",           "avg_price_psqft": 11500, "circle_rate_ratio": 0.60, "tier": "metro",  "lat": 28.4595, "lon": 77.0266},
    "Faridabad":       {"state": "Haryana",           "avg_price_psqft": 5800,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 28.4089, "lon": 77.3178},
    "Ghaziabad":       {"state": "Uttar Pradesh",     "avg_price_psqft": 5600,  "circle_rate_ratio": 0.57, "tier": "tier2",  "lat": 28.6692, "lon": 77.4538},

    # ── Maharashtra ──────────────────────────────────────────────────────
    "Mumbai":          {"state": "Maharashtra",       "avg_price_psqft": 24000, "circle_rate_ratio": 0.70, "tier": "metro",  "lat": 19.0760, "lon": 72.8777},
    "Navi Mumbai":     {"state": "Maharashtra",       "avg_price_psqft": 13500, "circle_rate_ratio": 0.65, "tier": "metro",  "lat": 19.0330, "lon": 73.0297},
    "Thane":           {"state": "Maharashtra",       "avg_price_psqft": 11000, "circle_rate_ratio": 0.65, "tier": "metro",  "lat": 19.2183, "lon": 72.9781},
    "Pune":            {"state": "Maharashtra",       "avg_price_psqft": 12950, "circle_rate_ratio": 0.68, "tier": "metro",  "lat": 18.5204, "lon": 73.8567},
    "Nagpur":          {"state": "Maharashtra",       "avg_price_psqft": 5400,  "circle_rate_ratio": 0.65, "tier": "tier2",  "lat": 21.1458, "lon": 79.0882},
    "Nashik":          {"state": "Maharashtra",       "avg_price_psqft": 5000,  "circle_rate_ratio": 0.63, "tier": "tier2",  "lat": 20.0059, "lon": 73.7898},
    "Aurangabad":      {"state": "Maharashtra",       "avg_price_psqft": 4200,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 19.8762, "lon": 75.3433},
    "Solapur":         {"state": "Maharashtra",       "avg_price_psqft": 3800,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 17.6599, "lon": 75.9064},
    "Kolhapur":        {"state": "Maharashtra",       "avg_price_psqft": 4500,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 16.7050, "lon": 74.2433},

    # ── Karnataka ────────────────────────────────────────────────────────
    "Bengaluru":       {"state": "Karnataka",         "avg_price_psqft": 9800,  "circle_rate_ratio": 0.60, "tier": "metro",  "lat": 12.9716, "lon": 77.5946},
    "Mysore":          {"state": "Karnataka",         "avg_price_psqft": 4800,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 12.2958, "lon": 76.6394},
    "Mangalore":       {"state": "Karnataka",         "avg_price_psqft": 5200,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 12.9141, "lon": 74.8560},
    "Hubli":           {"state": "Karnataka",         "avg_price_psqft": 4000,  "circle_rate_ratio": 0.57, "tier": "tier2",  "lat": 15.3647, "lon": 75.1240},
    "Belgaum":         {"state": "Karnataka",         "avg_price_psqft": 3800,  "circle_rate_ratio": 0.56, "tier": "tier2",  "lat": 15.8497, "lon": 74.4977},

    # ── Tamil Nadu ───────────────────────────────────────────────────────
    "Chennai":         {"state": "Tamil Nadu",        "avg_price_psqft": 8700,  "circle_rate_ratio": 0.70, "tier": "metro",  "lat": 13.0827, "lon": 80.2707},
    "Coimbatore":      {"state": "Tamil Nadu",        "avg_price_psqft": 5500,  "circle_rate_ratio": 0.65, "tier": "tier2",  "lat": 11.0168, "lon": 76.9558},
    "Madurai":         {"state": "Tamil Nadu",        "avg_price_psqft": 4600,  "circle_rate_ratio": 0.62, "tier": "tier2",  "lat":  9.9252, "lon": 78.1198},
    "Tiruchirappalli": {"state": "Tamil Nadu",        "avg_price_psqft": 4200,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 10.7905, "lon": 78.7047},
    "Salem":           {"state": "Tamil Nadu",        "avg_price_psqft": 3900,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 11.6643, "lon": 78.1460},
    "Tiruppur":        {"state": "Tamil Nadu",        "avg_price_psqft": 4100,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 11.1085, "lon": 77.3411},

    # ── Telangana ────────────────────────────────────────────────────────
    "Hyderabad":       {"state": "Telangana",         "avg_price_psqft": 8200,  "circle_rate_ratio": 0.60, "tier": "metro",  "lat": 17.3850, "lon": 78.4867},
    "Warangal":        {"state": "Telangana",         "avg_price_psqft": 3500,  "circle_rate_ratio": 0.55, "tier": "tier2",  "lat": 17.9784, "lon": 79.5941},
    "Nizamabad":       {"state": "Telangana",         "avg_price_psqft": 3200,  "circle_rate_ratio": 0.53, "tier": "tier2",  "lat": 18.6725, "lon": 78.0940},

    # ── Andhra Pradesh ───────────────────────────────────────────────────
    "Visakhapatnam":   {"state": "Andhra Pradesh",    "avg_price_psqft": 4700,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 17.6868, "lon": 83.2185},
    "Vijayawada":      {"state": "Andhra Pradesh",    "avg_price_psqft": 4500,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 16.5062, "lon": 80.6480},
    "Guntur":          {"state": "Andhra Pradesh",    "avg_price_psqft": 4000,  "circle_rate_ratio": 0.57, "tier": "tier2",  "lat": 16.3067, "lon": 80.4365},
    "Tirupati":        {"state": "Andhra Pradesh",    "avg_price_psqft": 4300,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 13.6288, "lon": 79.4192},
    "Amaravati":       {"state": "Andhra Pradesh",    "avg_price_psqft": 5500,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 16.5125, "lon": 80.5160},

    # ── Gujarat ──────────────────────────────────────────────────────────
    "Ahmedabad":       {"state": "Gujarat",           "avg_price_psqft": 5800,  "circle_rate_ratio": 0.65, "tier": "tier2",  "lat": 23.0225, "lon": 72.5714},
    "Surat":           {"state": "Gujarat",           "avg_price_psqft": 5100,  "circle_rate_ratio": 0.65, "tier": "tier2",  "lat": 21.1702, "lon": 72.8311},
    "Vadodara":        {"state": "Gujarat",           "avg_price_psqft": 4800,  "circle_rate_ratio": 0.63, "tier": "tier2",  "lat": 22.3072, "lon": 73.1812},
    "Rajkot":          {"state": "Gujarat",           "avg_price_psqft": 4500,  "circle_rate_ratio": 0.62, "tier": "tier2",  "lat": 22.3039, "lon": 70.8022},
    "Gandhinagar":     {"state": "Gujarat",           "avg_price_psqft": 5000,  "circle_rate_ratio": 0.63, "tier": "tier2",  "lat": 23.2156, "lon": 72.6369},

    # ── Rajasthan ────────────────────────────────────────────────────────
    "Jaipur":          {"state": "Rajasthan",         "avg_price_psqft": 5200,  "circle_rate_ratio": 0.65, "tier": "tier2",  "lat": 26.9124, "lon": 75.7873},
    "Jodhpur":         {"state": "Rajasthan",         "avg_price_psqft": 4200,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 26.2389, "lon": 73.0243},
    "Udaipur":         {"state": "Rajasthan",         "avg_price_psqft": 4500,  "circle_rate_ratio": 0.62, "tier": "tier2",  "lat": 24.5854, "lon": 73.7125},
    "Kota":            {"state": "Rajasthan",         "avg_price_psqft": 3800,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 25.2138, "lon": 75.8648},
    "Ajmer":           {"state": "Rajasthan",         "avg_price_psqft": 3600,  "circle_rate_ratio": 0.57, "tier": "tier2",  "lat": 26.4499, "lon": 74.6399},

    # ── Uttar Pradesh ────────────────────────────────────────────────────
    "Lucknow":         {"state": "Uttar Pradesh",     "avg_price_psqft": 4900,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 26.8467, "lon": 80.9462},
    "Agra":            {"state": "Uttar Pradesh",     "avg_price_psqft": 4200,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 27.1767, "lon": 78.0081},
    "Varanasi":        {"state": "Uttar Pradesh",     "avg_price_psqft": 4600,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 25.3176, "lon": 82.9739},
    "Kanpur":          {"state": "Uttar Pradesh",     "avg_price_psqft": 4400,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 26.4499, "lon": 80.3319},
    "Prayagraj":       {"state": "Uttar Pradesh",     "avg_price_psqft": 4300,  "circle_rate_ratio": 0.57, "tier": "tier2",  "lat": 25.4358, "lon": 81.8463},
    "Meerut":          {"state": "Uttar Pradesh",     "avg_price_psqft": 4800,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 28.9845, "lon": 77.7064},

    # ── West Bengal ──────────────────────────────────────────────────────
    "Kolkata":         {"state": "West Bengal",       "avg_price_psqft": 6800,  "circle_rate_ratio": 0.60, "tier": "metro",  "lat": 22.5726, "lon": 88.3639},
    "Durgapur":        {"state": "West Bengal",       "avg_price_psqft": 3500,  "circle_rate_ratio": 0.55, "tier": "tier2",  "lat": 23.5204, "lon": 87.3119},
    "Siliguri":        {"state": "West Bengal",       "avg_price_psqft": 3800,  "circle_rate_ratio": 0.55, "tier": "tier2",  "lat": 26.7271, "lon": 88.3953},
    "Asansol":         {"state": "West Bengal",       "avg_price_psqft": 3200,  "circle_rate_ratio": 0.53, "tier": "tier2",  "lat": 23.6739, "lon": 86.9524},

    # ── Madhya Pradesh ───────────────────────────────────────────────────
    "Indore":          {"state": "Madhya Pradesh",    "avg_price_psqft": 4600,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 22.7196, "lon": 75.8577},
    "Bhopal":          {"state": "Madhya Pradesh",    "avg_price_psqft": 4200,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 23.2599, "lon": 77.4126},
    "Jabalpur":        {"state": "Madhya Pradesh",    "avg_price_psqft": 3600,  "circle_rate_ratio": 0.57, "tier": "tier2",  "lat": 23.1815, "lon": 79.9864},
    "Gwalior":         {"state": "Madhya Pradesh",    "avg_price_psqft": 3800,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 26.2183, "lon": 78.1828},

    # ── Punjab / Chandigarh / Haryana ────────────────────────────────────
    "Chandigarh":      {"state": "Chandigarh",        "avg_price_psqft": 7600,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 30.7333, "lon": 76.7794},
    "Amritsar":        {"state": "Punjab",            "avg_price_psqft": 4500,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 31.6340, "lon": 74.8723},
    "Ludhiana":        {"state": "Punjab",            "avg_price_psqft": 5000,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 30.9010, "lon": 75.8573},
    "Jalandhar":       {"state": "Punjab",            "avg_price_psqft": 4200,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 31.3260, "lon": 75.5762},
    "Mohali":          {"state": "Punjab",            "avg_price_psqft": 6500,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 30.7046, "lon": 76.7179},

    # ── Kerala ───────────────────────────────────────────────────────────
    "Kochi":           {"state": "Kerala",            "avg_price_psqft": 6200,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat":  9.9312, "lon": 76.2673},
    "Thiruvananthapuram": {"state": "Kerala",         "avg_price_psqft": 5500,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat":  8.5241, "lon": 76.9366},
    "Kozhikode":       {"state": "Kerala",            "avg_price_psqft": 5000,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 11.2588, "lon": 75.7804},
    "Thrissur":        {"state": "Kerala",            "avg_price_psqft": 5200,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 10.5276, "lon": 76.2144},
    "Kollam":          {"state": "Kerala",            "avg_price_psqft": 4600,  "circle_rate_ratio": 0.57, "tier": "tier2",  "lat":  8.8932, "lon": 76.6141},

    # ── Bihar / Jharkhand / Odisha / Chhattisgarh ────────────────────────
    "Patna":           {"state": "Bihar",             "avg_price_psqft": 4400,  "circle_rate_ratio": 0.55, "tier": "tier2",  "lat": 25.5941, "lon": 85.1376},
    "Ranchi":          {"state": "Jharkhand",         "avg_price_psqft": 4000,  "circle_rate_ratio": 0.55, "tier": "tier2",  "lat": 23.3441, "lon": 85.3096},
    "Jamshedpur":      {"state": "Jharkhand",         "avg_price_psqft": 3800,  "circle_rate_ratio": 0.54, "tier": "tier2",  "lat": 22.8046, "lon": 86.2029},
    "Bhubaneswar":     {"state": "Odisha",            "avg_price_psqft": 4500,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 20.2961, "lon": 85.8245},
    "Cuttack":         {"state": "Odisha",            "avg_price_psqft": 3800,  "circle_rate_ratio": 0.55, "tier": "tier2",  "lat": 20.4625, "lon": 85.8828},
    "Raipur":          {"state": "Chhattisgarh",      "avg_price_psqft": 3900,  "circle_rate_ratio": 0.57, "tier": "tier2",  "lat": 21.2514, "lon": 81.6296},
    "Bhilai":          {"state": "Chhattisgarh",      "avg_price_psqft": 3500,  "circle_rate_ratio": 0.55, "tier": "tier2",  "lat": 21.2090, "lon": 81.4285},

    # ── North East / Hill States ─────────────────────────────────────────
    "Guwahati":        {"state": "Assam",             "avg_price_psqft": 4200,  "circle_rate_ratio": 0.55, "tier": "tier2",  "lat": 26.1445, "lon": 91.7362},
    "Shillong":        {"state": "Meghalaya",         "avg_price_psqft": 4000,  "circle_rate_ratio": 0.53, "tier": "tier2",  "lat": 25.5788, "lon": 91.8933},
    "Agartala":        {"state": "Tripura",           "avg_price_psqft": 3200,  "circle_rate_ratio": 0.52, "tier": "tier2",  "lat": 23.8315, "lon": 91.2868},
    "Imphal":          {"state": "Manipur",           "avg_price_psqft": 3000,  "circle_rate_ratio": 0.50, "tier": "tier2",  "lat": 24.8170, "lon": 93.9368},
    "Aizawl":          {"state": "Mizoram",           "avg_price_psqft": 3000,  "circle_rate_ratio": 0.50, "tier": "tier2",  "lat": 23.7271, "lon": 92.7176},

    # ── North / Hill States ──────────────────────────────────────────────
    "Dehradun":        {"state": "Uttarakhand",       "avg_price_psqft": 5500,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 30.3165, "lon": 78.0322},
    "Haridwar":        {"state": "Uttarakhand",       "avg_price_psqft": 4200,  "circle_rate_ratio": 0.56, "tier": "tier2",  "lat": 29.9457, "lon": 78.1642},
    "Shimla":          {"state": "Himachal Pradesh",  "avg_price_psqft": 6000,  "circle_rate_ratio": 0.58, "tier": "tier2",  "lat": 31.1048, "lon": 77.1734},
    "Jammu":           {"state": "Jammu and Kashmir", "avg_price_psqft": 4800,  "circle_rate_ratio": 0.55, "tier": "tier2",  "lat": 32.7266, "lon": 74.8570},
    "Srinagar":        {"state": "Jammu and Kashmir", "avg_price_psqft": 4500,  "circle_rate_ratio": 0.55, "tier": "tier2",  "lat": 34.0837, "lon": 74.7973},

    # ── Other metros / UTs ───────────────────────────────────────────────
    "Panaji":          {"state": "Goa",               "avg_price_psqft": 7500,  "circle_rate_ratio": 0.62, "tier": "tier2",  "lat": 15.4909, "lon": 73.8278},
    "Margao":          {"state": "Goa",               "avg_price_psqft": 6800,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 15.2832, "lon": 73.9862},
    "Puducherry":      {"state": "Puducherry",        "avg_price_psqft": 5000,  "circle_rate_ratio": 0.60, "tier": "tier2",  "lat": 11.9416, "lon": 79.8083},
    "Solan":           {"state": "Himachal Pradesh",  "avg_price_psqft": 4500,  "circle_rate_ratio": 0.55, "tier": "tier2",  "lat": 30.9045, "lon": 77.0967},
}

PROPERTY_TYPES = [
    "Apartment", "Villa", "Independent House", "Plot", "Office", "Shop", "Warehouse"
]
# weight: residential dominates the dataset (mirrors real registration mix)
PROPERTY_TYPE_WEIGHTS = [0.45, 0.10, 0.18, 0.12, 0.06, 0.06, 0.03]

PROPERTY_CATEGORY_MAP = {
    "Apartment": "Residential", "Villa": "Residential",
    "Independent House": "Residential", "Plot": "Land",
    "Office": "Commercial", "Shop": "Commercial", "Warehouse": "Industrial",
}

FACING_DIRECTIONS = ["North", "South", "East", "West", "North-East", "North-West", "South-East", "South-West"]
CONSTRUCTION_QUALITY = ["Basic", "Standard", "Premium", "Luxury"]
OCCUPANCY_TYPES = ["Owner Occupied", "Rented", "Vacant"]


def _sample_property_type(n):
    return np.random.choice(PROPERTY_TYPES, size=n, p=PROPERTY_TYPE_WEIGHTS)


def _area_for_type(ptype, n):
    """Realistic area distributions (sqft) by property type."""
    if ptype == "Apartment":
        return np.clip(np.random.normal(1150, 420, n), 350, 4500)
    if ptype == "Villa":
        return np.clip(np.random.normal(3200, 900, n), 1500, 9000)
    if ptype == "Independent House":
        return np.clip(np.random.normal(1800, 700, n), 600, 6000)
    if ptype == "Plot":
        return np.clip(np.random.normal(2400, 1200, n), 600, 12000)
    if ptype == "Office":
        return np.clip(np.random.normal(1500, 800, n), 250, 8000)
    if ptype == "Shop":
        return np.clip(np.random.normal(650, 350, n), 120, 3500)
    if ptype == "Warehouse":
        return np.clip(np.random.normal(8000, 4000, n), 1500, 40000)
    return np.clip(np.random.normal(1200, 400, n), 300, 5000)


def generate_dataset(n_rows: int = 12000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate the full synthetic-but-realistic Indian property dataset.

    Parameters
    ----------
    n_rows : total number of property records to generate.
    seed   : RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame with one row per property and all engineering-ready columns.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    cities = list(CITY_PROFILES.keys())
    # Metro cities get proportionally more listings (mirrors real registration volume)
    city_weights = np.array([
        3.0 if CITY_PROFILES[c]["tier"] == "metro" else 1.0 for c in cities
    ])
    city_weights = city_weights / city_weights.sum()

    city_choice = np.random.choice(cities, size=n_rows, p=city_weights)
    ptype_choice = _sample_property_type(n_rows)

    rows = []
    for i in range(n_rows):
        city = city_choice[i]
        profile = CITY_PROFILES[city]
        ptype = ptype_choice[i]
        category = PROPERTY_CATEGORY_MAP[ptype]

        # ── Area ──────────────────────────────────────────────────────
        plot_or_built = _area_for_type(ptype, 1)[0]
        if ptype == "Plot":
            plot_area = plot_or_built
            built_up_area = 0.0
            carpet_area = 0.0
        else:
            built_up_area = plot_or_built
            carpet_area = built_up_area * np.random.uniform(0.78, 0.88)
            plot_area = built_up_area * np.random.uniform(1.0, 1.6) if ptype in (
                "Villa", "Independent House") else built_up_area

        # ── Construction / age ──────────────────────────────────────────
        construction_year = int(np.clip(np.random.normal(2012, 9), 1975, 2025))
        property_age = 2026 - construction_year

        # ── Floors / bedrooms / bathrooms ──────────────────────────────
        if ptype in ("Apartment",):
            num_floors = np.random.choice([1], p=[1.0])
            floor_number = np.random.randint(0, 25)
            bedrooms = np.random.choice([1, 2, 3, 4, 5], p=[0.18, 0.38, 0.30, 0.11, 0.03])
            bathrooms = max(1, bedrooms - np.random.choice([0, 1], p=[0.6, 0.4]))
        elif ptype in ("Villa", "Independent House"):
            num_floors = np.random.choice([1, 2, 3], p=[0.35, 0.5, 0.15])
            floor_number = 0
            bedrooms = np.random.choice([2, 3, 4, 5, 6], p=[0.10, 0.30, 0.35, 0.18, 0.07])
            bathrooms = max(1, bedrooms - np.random.choice([0, 1], p=[0.5, 0.5]))
        else:
            num_floors = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05])
            floor_number = np.random.randint(0, 8)
            bedrooms = 0
            bathrooms = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])

        # ── Amenities / location quality flags ──────────────────────────
        has_parking = np.random.choice([0, 1], p=[0.25, 0.75])
        road_width_ft = float(np.clip(np.random.normal(30, 12), 8, 100))
        nearby_metro = np.random.choice([0, 1], p=[0.55, 0.45]) if profile["tier"] == "metro" else np.random.choice([0, 1], p=[0.9, 0.1])
        nearby_school = np.random.choice([0, 1], p=[0.3, 0.7])
        nearby_hospital = np.random.choice([0, 1], p=[0.35, 0.65])
        nearby_market = np.random.choice([0, 1], p=[0.25, 0.75])
        nearby_airport = np.random.choice([0, 1], p=[0.85, 0.15])
        nearby_railway = np.random.choice([0, 1], p=[0.5, 0.5])
        corner_plot = np.random.choice([0, 1], p=[0.78, 0.22])
        facing = np.random.choice(FACING_DIRECTIONS)
        lift_available = np.random.choice([0, 1], p=[0.4, 0.6]) if ptype == "Apartment" and floor_number >= 3 else (
            np.random.choice([0, 1], p=[0.85, 0.15]))
        swimming_pool = np.random.choice([0, 1], p=[0.92, 0.08])
        solar_panels = np.random.choice([0, 1], p=[0.93, 0.07])
        rainwater_harvesting = np.random.choice([0, 1], p=[0.75, 0.25])
        green_certified = np.random.choice([0, 1], p=[0.94, 0.06])
        construction_quality = np.random.choice(CONSTRUCTION_QUALITY, p=[0.20, 0.45, 0.27, 0.08])
        occupancy = np.random.choice(OCCUPANCY_TYPES, p=[0.55, 0.30, 0.15])

        # ── Price modeling ───────────────────────────────────────────
        base_psqft = profile["avg_price_psqft"]
        quality_mult = {"Basic": 0.78, "Standard": 1.0, "Premium": 1.28, "Luxury": 1.75}[construction_quality]
        type_mult = {
            "Apartment": 1.0, "Villa": 1.15, "Independent House": 0.95,
            "Plot": 0.55, "Office": 1.05, "Shop": 1.4, "Warehouse": 0.35,
        }[ptype]
        age_mult = max(0.55, 1.0 - property_age * 0.012)
        metro_mult = 1.12 if nearby_metro else 1.0
        floor_mult = 1.0 + min(floor_number, 20) * 0.004 if ptype == "Apartment" else 1.0
        amenity_mult = 1.0 + 0.02 * sum([
            swimming_pool, solar_panels, rainwater_harvesting, green_certified, lift_available, has_parking
        ])
        noise = np.random.normal(1.0, 0.09)

        effective_psqft = (
            base_psqft * quality_mult * type_mult * age_mult *
            metro_mult * floor_mult * amenity_mult * noise
        )
        effective_psqft = max(800, effective_psqft)

        area_for_price = built_up_area if ptype != "Plot" else plot_area
        market_price = effective_psqft * area_for_price

        circle_rate_value = market_price * profile["circle_rate_ratio"] * np.random.uniform(0.92, 1.02)
        govt_valuation = circle_rate_value * np.random.uniform(0.95, 1.05)

        rental_yield_pct = (
            np.random.uniform(2.0, 3.5) if category == "Residential"
            else np.random.uniform(4.0, 7.5)
        )

        population_density = int(np.clip(np.random.normal(
            12000 if profile["tier"] == "metro" else 6000, 3500), 800, 35000))

        ward = f"Ward-{np.random.randint(1, 60)}"
        zone = np.random.choice(["North", "South", "East", "West", "Central"])
        municipality = f"{city} Municipal Corporation"
        pin_code = int(np.clip(np.random.normal(110001 if city in ("Delhi", "New Delhi") else 400001, 30000), 100001, 855117))

        lat_jitter = np.random.normal(0, 0.08)
        lon_jitter = np.random.normal(0, 0.08)

        rows.append({
            "state": profile["state"],
            "city": city,
            "district": city,
            "pin_code": pin_code,
            "latitude": round(profile["lat"] + lat_jitter, 5),
            "longitude": round(profile["lon"] + lon_jitter, 5),
            "property_type": ptype,
            "property_category": category,
            "built_up_area_sqft": round(built_up_area, 1),
            "carpet_area_sqft": round(carpet_area, 1),
            "plot_area_sqft": round(plot_area, 1),
            "construction_year": construction_year,
            "property_age": property_age,
            "num_floors": int(num_floors),
            "floor_number": int(floor_number),
            "bedrooms": int(bedrooms),
            "bathrooms": int(bathrooms),
            "parking": int(has_parking),
            "road_width_ft": round(road_width_ft, 1),
            "nearby_metro": int(nearby_metro),
            "nearby_school": int(nearby_school),
            "nearby_hospital": int(nearby_hospital),
            "nearby_market": int(nearby_market),
            "nearby_airport": int(nearby_airport),
            "nearby_railway_station": int(nearby_railway),
            "corner_plot": int(corner_plot),
            "facing_direction": facing,
            "lift_available": int(lift_available),
            "swimming_pool": int(swimming_pool),
            "solar_panels": int(solar_panels),
            "rainwater_harvesting": int(rainwater_harvesting),
            "green_certified": int(green_certified),
            "construction_quality": construction_quality,
            "occupancy": occupancy,
            "ward": ward,
            "zone": zone,
            "municipality": municipality,
            "population_density": population_density,
            "rental_yield_pct": round(rental_yield_pct, 2),
            "circle_rate_value": round(circle_rate_value, 0),
            "govt_valuation": round(govt_valuation, 0),
            "market_price": round(market_price, 0),
        })

    df = pd.DataFrame(rows)
    return df


def save_dataset(path: str = "data/india_property_data.csv", n_rows: int = 12000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = generate_dataset(n_rows=n_rows)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} rows -> {path}")
    return df


if __name__ == "__main__":
    save_dataset()
