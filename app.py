"""
app.py -- Smart Indian Property Valuation & Tax Assessment System
Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Property valuation and tax", page_icon=":material/real_estate_agent:", layout="wide",
                   initial_sidebar_state="collapsed")

import theme  # noqa: E402
from views import bulk, data_pages, finance_pages, home, market_pages, simulator, studio, trust  # noqa: E402

theme.inject()
theme.persist("s_")  # keep Studio inputs when moving between pages

P = st.Page
pages = {
    "home": P(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
    "studio": P(studio.render, title="Valuation Studio", icon=":material/tune:", url_path="studio"),
    "simulator": P(simulator.render, title="Value-add simulator", icon=":material/construction:", url_path="simulator"),
    "bulk": P(bulk.render_bulk, title="Bulk valuation", icon=":material/table_view:", url_path="bulk"),
    "emi": P(finance_pages.render_emi, title="EMI lab", icon=":material/calculate:", url_path="emi"),
    "afford": P(finance_pages.render_afford, title="Affordability", icon=":material/account_balance_wallet:", url_path="afford"),
    "rentbuy": P(finance_pages.render_rentbuy, title="Rent or buy", icon=":material/compare_arrows:", url_path="rent-or-buy"),
    "investor": P(finance_pages.render_investor, title="Investor view", icon=":material/trending_up:", url_path="investor"),
    "map": P(market_pages.render_map, title="Market map", icon=":material/map:", url_path="map"),
    "faceoff": P(market_pages.render_faceoff, title="City face-off", icon=":material/location_city:", url_path="face-off"),
    "health": P(trust.render_health, title="Model health", icon=":material/monitor_heart:", url_path="model-health"),
    "conformal": P(trust.render_conformal, title="Confidence lab", icon=":material/science:", url_path="confidence-lab"),
    "how": P(trust.render_how, title="How it works", icon=":material/menu_book:", url_path="how-it-works"),
    "history": P(data_pages.render_history, title="History", icon=":material/history:", url_path="history"),
    "rules": P(data_pages.render_rules, title="Tax rules and about", icon=":material/gavel:", url_path="rules"),
}
theme.PAGES.update(pages)

nav = st.navigation({
    "": [pages["home"]],
    "Value": [pages["studio"], pages["simulator"], pages["bulk"]],
    "Loans": [pages["emi"], pages["afford"], pages["rentbuy"], pages["investor"]],
    "Markets": [pages["map"], pages["faceoff"]],
    "Trust": [pages["health"], pages["conformal"], pages["how"]],
    "Records": [pages["history"], pages["rules"]],
}, position="top")

with st.sidebar:
    st.caption("The dataset behind this app is synthetic, anchored to published 2025-26 market levels. Estimates are for learning and planning, not for filing or lending.")
nav.run()
