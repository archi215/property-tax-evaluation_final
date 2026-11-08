"""ui_common.py -- shared Streamlit helpers: styling, cached resources, constants, small widgets."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import config
from dataset_loader import CITY_PROFILES
from market import load_market_frame
from utils import fmt_compact, fmt_inr
from valuation import ValuationModel

COLORS = {"indigo": "#1B3A6B", "saffron": "#FF8C42", "leaf": "#2E7D5B", "gold": "#C9A227", "plum": "#8E5572", "red": "#C0392B"}


PROPERTY_TYPES = ["Apartment", "Villa", "Independent House", "Plot", "Office", "Shop", "Warehouse"]
PROPERTY_CATEGORY_MAP = {"Apartment": "Residential", "Villa": "Residential", "Independent House": "Residential",
                         "Plot": "Land", "Office": "Commercial", "Shop": "Commercial", "Warehouse": "Industrial"}
FACING_DIRECTIONS = ["North", "South", "East", "West", "North-East", "North-West", "South-East", "South-West"]
CONSTRUCTION_QUALITY = ["Basic", "Standard", "Premium", "Luxury"]

OTHER_INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Himachal Pradesh", "Jharkhand",
    "Kerala", "Madhya Pradesh", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Jammu and Kashmir", "Ladakh", "Puducherry", "Andaman and Nicobar Islands",
    "Dadra and Nagar Haveli and Daman and Diu", "Lakshadweep", "Chandigarh", "Haryana", "Gujarat", "Karnataka",
    "Maharashtra", "Delhi",
]

STATE_CITY_MAP: dict = {}
for _city, _p in CITY_PROFILES.items():
    STATE_CITY_MAP.setdefault(_p["state"], []).append(_city)
ALL_STATES = sorted(set(STATE_CITY_MAP) | set(OTHER_INDIAN_STATES))


# ───────────────────────────── cached resources ─────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model() -> ValuationModel | None:
    return ValuationModel.try_load()


@st.cache_data(show_spinner=False)
def get_market() -> pd.DataFrame:
    return load_market_frame()


@st.cache_data(show_spinner=False)
def get_tax_rule_files() -> dict:
    files = {}
    if os.path.isdir(config.TAX_RULES_DIR):
        for fname in sorted(os.listdir(config.TAX_RULES_DIR)):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(config.TAX_RULES_DIR, fname), encoding="utf-8") as f:
                        files[fname] = json.load(f)
                except Exception:
                    pass
    return files


def require_model() -> ValuationModel:
    vm = get_model()
    if vm is None:
        st.error("No trained model found. Run `python preprocess.py` then `python 2_train.py` (or `python run_pipeline.py`).")
        st.stop()
    return vm


# ───────────────────────────── small widgets ────────────────────────────────
def metric_card(title: str, value: str, sub: str = "", tag: str = "h3"):
    st.markdown(f"<div class='metric-card'><{tag}>{value}</{tag}>{title}"
                f"{'<br><small>' + sub + '</small>' if sub else ''}</div>", unsafe_allow_html=True)


def tax_card(title: str, value: str, sub: str = ""):
    st.markdown(f"<div class='tax-card'><b>{title}</b><br><h4>{value}</h4>{'<small>' + sub + '</small>' if sub else ''}</div>",
                unsafe_allow_html=True)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def disclaimer(text: str):
    st.markdown(f"<div class='disclaimer-box'>{text}</div>", unsafe_allow_html=True)


def tax_rules_banner():
    if not get_tax_rule_files():
        st.warning("No `tax_rules/*.json` files were found, so tax figures will be zero. Copy your rule files "
                   "(Delhi.json, Maharashtra.json, Karnataka.json, _default.json) into the `tax_rules/` folder.")
