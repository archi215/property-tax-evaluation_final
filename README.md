# Smart Indian Property Valuation & Tax Assessment System  (v2)

Python + Streamlit + XGBoost. Predicts an Indian property's market value **with calibrated prediction intervals and a "why this
price" explanation**, then computes property tax, stamp duty, capital gains, home-loan economics and more.

> **Setup note:** copy your `tax_rules/*.json` files (Delhi, Maharashtra, Karnataka, `_default`) into `tax_rules/`. They were not part of the
> v2 hand-off, so the app shows a banner and tax figures stay zero until they are present.

```bash
pip install -r requirements.txt -r requirements-dev.txt
python run_pipeline.py          # data -> train -> evaluate -> 14-model comparison  (--skip-compare to save time)
python -m pytest -q tests       # 21 tests
streamlit run app.py
```
Or `docker build -t valuation . && docker run -p 8501:8501 valuation`.

## What's new in v2

**Model & ML pipeline**
- **Real prediction intervals**: split-conformal (90% / 80%) replace the fixed ±10%. Test coverage: 91.0% / 80.0%.
- **Exact SHAP drivers** ("Why this price?") per property, grouped (Location, Size, Quality, Age, Amenities...).
- **Honest evaluation**: 70/10/10/10 train/valid/calibration/test split assigned once in `preprocess.py`. The old script early-stopped on
  the *test* set, and the production model hit its 600-tree cap (still under-fitted).
- **Leak-free encoding**: city/state target encoding is fitted on train rows only (smoothed) and saved to `models/encoders.json`; the app
  no longer needs the raw CSV.
- **New features**: `log_effective_area`, `log_city_encoded`, `log_city_anchor`, `city_anchor_psqft`, `is_metro`; noise columns
  (`pin_code`, `population_density`, `rental_yield_pct`) removed from model inputs.
- Native `property_model.json` (version-robust) + metadata.json; optional `4_tune.py`; LightGBM (and CatBoost if installed) in the comparison.
- **Train/serve parity test** guarantees the app builds exactly the feature row the model was trained on.

**Frontend (v3, see below)** - the app is now topic-based and live. Earlier summary:  - Valuation & Tax Calculator (tabs: valuation + local market percentile, drivers, taxes, loan & tax benefits, future
scenarios, insights, export) · **Value-Add Simulator** (improvement ROI + sensitivity curves) · **Loan / Affordability / Rent-vs-Buy / Rental ROI**
· **Bulk CSV valuation** · **Market Explorer** (map, city stats) · City comparison (2-4 cities) · Model Health · **Persistent SQLite history** · About.
Plus **PDF report export**, results that survive reruns (the old app lost them on any button click), `width="stretch"` API migration and
OS-aware dark mode (the old toggle was a stub).

**Tax engine (small, safe fixes)** - falls back to the generic path when a state file is missing (previously crashed into a warning);
enforces the **23 July 2024 indexation cut-off** your README described but the code did not.

## Findings you should know about
Current model (XGBoost, held-out test): **R² 0.967, MAPE 8.9%, median APE 7.2%, MAE ≈ ₹9.1 lakh** (previous version on the same data: R² 0.945 / MAPE ~10%).

1. **The README's explanation for linear models was wrong.** Linear/Ridge scored negative R² not because they are "structurally unable", but
   because price = ₹/sqft × area is multiplicative and only *raw* area was supplied. With `log` area features Ridge goes from **R² -1.0 → 0.979**
   (MAPE 7.4%) and now *beats* XGBoost here. The synthetic generator is log-linear plus ~9% noise, so Ridge is already near the noise floor.
   XGBoost stays as production for exact SHAP and real-world non-linearity, but **accuracy on this dataset overstates real-world performance**.
2. Warehouses (MAPE ≈ 20%) and shops (≈ 11%) are the weakest segments (few rows) - see Model Health.
3. A small 8-iteration random search *did not* beat the defaults on test, so tuned params are not applied; run `python 4_tune.py --iters 50` for a real search.
4. Value-Add Simulator uplifts reflect the synthetic generator's flat ~2% per amenity - illustrative, not market evidence. Cost inputs are placeholders.

## Rules modelled (verified against FY 2025-26 / 2026-27 guidance)
Sec 24(b) interest (≤ ₹2 L, self-occupied) and Sec 80C (≤ ₹1.5 L incl. stamp duty/registration) are **old-regime only**; the new regime gives
neither for a self-occupied home. LTCG 12.5% (no indexation) or 20% with indexation for property acquired before 23 Jul 2024. The Income-tax Act 2025
renumbers sections from FY 2026-27 - confirm with a CA. **Nothing here is authoritative for filing.**

## Not yet built / next ideas
Full 28-state rule coverage, OpenStreetMap amenity features, admin UI for rule JSONs, AI tax chatbot, fraud/outlier flags, a Ridge+XGBoost blend,
and validation on *real* transaction data (the single most valuable next step).

## Layout
`app.py` router · `views/` pages · `ui_common.py` · `valuation.py` (model logic) · `finance.py` · `insights.py` · `market.py` · `storage.py` ·
`report.py` · `tax_engine.py` · `preprocess.py` / `2_train.py` / `3_evaluate.py` / `4_tune.py` / `model_comparison.py` · `run_pipeline.py` · `tests/`

## Research paper experiments
`python extended_experiments.py` reproduces the stability (5 splits, bootstrap CIs), conditional-coverage, calibration-size, learning-curve,
leakage, blend and latency experiments reported in the paper; results are saved to `outputs/extended_results.json`.

## Frontend v3: topic-based and interactive
The interface is organised as a **top navigation with five topics**: Value (Valuation Studio, Value-add simulator, Bulk valuation), Loans (EMI lab,
Affordability, Rent or buy, Investor view), Markets (Market map, City face-off), Trust (Model health, Confidence lab, How it works) and Records
(History, Tax rules). A **Home hub** offers a 10-second valuation and links to every topic.

- **Live Valuation Studio**: no submit button. Every input (sliders, pills, toggles) recomputes the value, calibrated range, SHAP waterfall, taxes,
  loan and projections immediately; pin scenarios to compare them, save to history, export a PDF.
- **Interactive Plotly charts** everywhere (hover, zoom, selection): waterfall of price drivers, EMI heat-map, rent-vs-buy sensitivity grid,
  affordability by city, click-to-explore market map, scenario fans.
- **Confidence lab**: choose a confidence level and watch the range and its measured coverage on unseen properties change.
- **Design**: a survey-sheet look (`theme.py`) with one signature component, the valuation ruler; light/dark aware; Streamlit >= 1.50 (`st.navigation`, `st.pills`).
- `tests/test_ui_smoke.py` renders every page headlessly.

## Custom web frontend (no Streamlit)
A fully custom interface lives in `web/index.html`, served by a FastAPI backend (`server.py`, `service.py`) that reuses the same model, tax engine
and finance code:

```bash
python -m uvicorn server:app --port 8000      # then open http://localhost:8000
```
Screens:
- **Studio**: colour-blocked control panel, oversized live price, survey ruler, SHAP waterfall, costs and tax, loan, advice, a **Negotiate** tab
  (where an asking price sits among plausible values, with a suggested opening-offer band), **Save to history** and **PDF report**.
- **Compare**: your Studio property against a second one, with a verdict on value per sq ft.
- **Money**: loan lab (EMI heat-map, prepayment), affordability (stress test, which cities fit your budget), rent-or-buy (crossover and sensitivity grid).
- **Portfolio**: bulk CSV valuation with template and download, plus saved valuations.
- **Market** (click-through map) and **Confidence** (live calibration lab).

The page loads fonts and Plotly.js from CDNs, so it needs an internet connection. The Streamlit app (`app.py`) remains available.

## Group-conditional intervals
`conformal.py` builds Mondrian split-conformal quantiles per property type (>= 30 calibration rows), then per category (>= 80), then global.
`python recalibrate.py` rebuilds them for an already-trained model and prints test coverage by type. Measured 90% coverage by type (test split):
apartment 91.9%, independent house 92.2%, office 87.9%, plot 97.8%, shop 92.3%, villa 95.5%, warehouse 74.4% (up from 53.8% with the global
quantile; only 33 warehouse calibration rows exist, so it remains below target). Small groups need more data, not a cleverer method.
