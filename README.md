# Smart Indian Property Valuation & Tax Assessment System

**Stage 1 build** — Indian dataset, XGBoost valuation model, 11-model comparison,
and a real, researched property-tax engine for Delhi, Maharashtra (Mumbai & Pune),
and Karnataka (Bengaluru), with a safe generic fallback for every other state/UT.

This extends an existing Python + Streamlit + XGBoost US-housing price predictor
into an Indian property valuation + tax calculator, while preserving every original
feature (prediction, model comparison, mortgage calculator, confidence interval,
feature importance, evaluation metrics, prediction history).

---

## What's real vs. what's modeled

Being upfront about this matters more than looking complete.

### Dataset (`dataset_loader.py`)
There is no free, row-level, consistently-schemed Indian property transaction
dataset spanning multiple cities (most municipal/government data is published as
PDF bulletins, not CSVs). So this project uses a **hybrid approach**:

- **Real anchors**: average price-per-sqft by city, and circle-rate-to-market-value
  ratios, are taken from public 2025-26 market reports (Knight Frank, Magicbricks,
  99acres, Square Yards, NoBroker, Statista). These are hard-coded in
  `CITY_PROFILES` inside `dataset_loader.py`.
- **Synthetic rows**: individual property records (exact area, age, amenities,
  floor, etc.) are generated from statistical distributions centered on those real
  anchors, with realistic correlations (older properties → lower price/sqft, metro
  proximity → premium, etc.)

This is disclosed in the app's "About the Dataset & Tax Rules" page. Nothing here
is presented as scraped real transactions.

**Important modeling note**: `circle_rate_value` and `govt_valuation` are
deliberately **excluded from the model's predictive features**. In the synthetic
generator they're derived as `market_price × ratio + small noise`, which is a
near-deterministic function of the prediction target — including them let the
model "predict" price by trivially inverting that formula (verified: they alone
explained ~99% of feature importance in an earlier version of this build). They
remain in the raw dataset and are used legitimately by `tax_engine.py` for stamp
duty calculations, where circle rate is a genuine real-world input.

### Tax rules (`tax_rules/*.json`)
Three states have **researched, cited rulesets**, reconstructed from public
2025-26 municipal guides (each JSON file lists its sources):

| File | Coverage | System |
|---|---|---|
| `Delhi.json` | Delhi (MCD) | Unit Area System, categories A-H |
| `Maharashtra.json` | Mumbai (BMC), Pune (PMC), + generic fallback for other MH cities | Capital Value System |
| `Karnataka.json` | Bengaluru (BBMP) | Unit Area Value System, zones A-F |

Every other state/UT uses `tax_rules/_default.json` — a clearly-labelled generic
Annual Rateable Value estimate. The app always tells you when this fallback is
being used (`is_fallback_used` flag), and the JSON itself carries a disclaimer.

**Capital gains tax** uses the actual current rules (FY 2025-26): 12.5% LTCG
without indexation, or 20% with indexation (CII = 376) for property acquired
before 23 July 2024 — the engine computes both and picks whichever is lower, as
the law allows.

Stamp duty, registration fee percentages/caps, and rebate structures (lump-sum
payment rebates, senior-citizen/women/disabled rebates, green-building rebates,
self-occupancy rebates) are all sourced from public guides current as of FY
2025-26 and cited inside each JSON file's `"sources"` key.

**None of this should be treated as authoritative for an actual tax filing.**
Always verify against your municipal corporation's official portal.

---

## Project structure

```
project/
├── app.py                  # Streamlit app (5 pages: calculator, model comparison,
│                           #   history, city/state comparison, about)
├── dataset_loader.py        # Indian property dataset generator
├── preprocess.py            # Cleaning, encoding, feature engineering
├── 2_train.py                # Trains the production XGBoost model
├── 3_evaluate.py              # Evaluation metrics + diagnostic plots
├── model_comparison.py      # Trains & compares 12 regression models
├── tax_engine.py             # Core tax calculation engine (all logic)
├── tax_rules/
│   ├── Delhi.json
│   ├── Maharashtra.json
│   ├── Karnataka.json
│   └── _default.json        # generic fallback for unmapped states
├── data/                     # generated CSVs (gitignored in a real repo)
├── models/                   # trained model + feature list (.pkl)
├── outputs/                  # evaluation charts, comparison results
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

# 1. Generate the dataset + clean it
python preprocess.py          # auto-generates data/india_property_data.csv if missing

# 2. Train the production model
python 2_train.py

# 3. Evaluate it
python 3_evaluate.py

# 4. (Optional) Compare 12 models
python model_comparison.py

# 5. Launch the app
streamlit run app.py
```

## Current model performance (XGBoost, production model)

- R²: 0.945
- MAPE: ~9.8%
- MAE: ~₹16.4 lakh (on a dataset where city-level median prices range ~₹54L-₹3Cr)

Linear/Ridge/Lasso/ElasticNet score **near zero or negative R²** on this dataset —
this is an honest, expected result, not a bug: property price here is a
*multiplicative* function of city tier, area, quality, and amenities, which linear
models structurally can't capture. Tree-based ensembles (Gradient Boosting, XGBoost,
Random Forest) dominate, which mirrors how real-world automated valuation models
are built in practice.

## What's next (not yet built)

This is Stage 1 of a larger spec. Not yet implemented: full 28-state tax rule
coverage, OpenStreetMap integration, PDF/Excel/QR exports, admin panel for
editing tax rules without code, AI tax chatbot, fraud detection, and the full
documentation set (ER diagrams, viva questions, etc.) Each of these is a
substantial standalone piece of work and should be tackled incrementally with
review at each step, the same way Stage 1 was.
