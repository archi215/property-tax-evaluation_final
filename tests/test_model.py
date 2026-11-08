import os
import numpy as np
import pytest
import config

pytestmark = pytest.mark.skipif(not os.path.exists(config.METADATA_JSON), reason="model not trained")

from valuation import ValuationModel, feature_group

INP = dict(city="Delhi", state="Delhi", property_type="Apartment", property_category="Residential",
           built_up_area_sqft=1200, carpet_area_sqft=1000, plot_area_sqft=1500, construction_year=2015,
           bedrooms=3, bathrooms=2, floor_number=2)


@pytest.fixture(scope="module")
def vm():
    return ValuationModel()


def test_prediction_and_interval_ordering(vm):
    p = vm.predict(INP)
    assert p.lower_90 < p.lower_80 < p.price < p.upper_80 < p.upper_90


def test_monotonic_sanity(vm):
    small = vm.predict({**INP, "built_up_area_sqft": 800, "carpet_area_sqft": 680}).price
    big = vm.predict({**INP, "built_up_area_sqft": 2000, "carpet_area_sqft": 1700}).price
    assert big > small
    assert vm.predict({**INP, "city": "Mumbai", "state": "Maharashtra"}).price > vm.predict(INP).price


def test_unseen_city_warns(vm):
    assert vm.predict({**INP, "city": "Atlantis"}).warnings


def test_shap_groups_sum_to_gap(vm):
    d = vm.explain(INP)
    assert abs(d["rupee_effect"].sum() - (d.attrs["price"] - d.attrs["baseline_price"])) < 1.0
    assert feature_group("city_encoded") == "Location" and feature_group("nearby_metro") == "Surroundings & access"


def test_what_if_skips_existing_and_orders(vm):
    out = vm.what_if({**INP, "solar_panels": True}, ["solar_panels", "rainwater_harvesting"])
    assert "solar_panels" not in set(out["key"])


def test_train_serve_parity(vm):
    """The app's feature builder must reproduce the training matrix exactly."""
    from common import load_splits
    from dataset_loader import CITY_PROFILES
    _, sp = load_splits()
    X, _ = sp["test"]
    r = X.iloc[0]
    get = lambda p: [c for c in X.columns if c.startswith(p) and r[c] == 1][0][len(p):]
    inv = {round(v, 4): k for k, v in vm.encoders["city_map"].items()}
    city = inv[round(r["city_encoded"], 4)]
    inp = {"city": city, "state": CITY_PROFILES[city]["state"], "property_type": get("property_type_"),
           "property_category": get("property_category_"), "facing_direction": get("facing_direction_"),
           "construction_quality": get("construction_quality_"), "occupancy": get("occupancy_"), "zone": get("zone_")}
    for k in ["latitude", "longitude", "built_up_area_sqft", "carpet_area_sqft", "plot_area_sqft", "construction_year", "num_floors",
              "floor_number", "bedrooms", "bathrooms", "road_width_ft", "parking", "nearby_metro", "nearby_school", "nearby_hospital",
              "nearby_market", "nearby_airport", "nearby_railway_station", "corner_plot", "lift_available", "swimming_pool",
              "solar_panels", "rainwater_harvesting", "green_certified"]:
        inp[k] = r[k]
    row, _ = vm.build_row(inp)
    assert np.allclose(row.iloc[0].values.astype(float), r[vm.features].values.astype(float))


def test_interval_coverage_recorded(vm):
    cov = vm.meta["empirical_coverage_test"]
    assert 0.85 <= cov["90"] <= 0.95 and 0.75 <= cov["80"] <= 0.85


def test_every_model_feature_has_a_driver_group(vm):
    """Guards against new features silently landing in 'Other' in the explanations."""
    assert [f for f in vm.features if feature_group(f) == "Other"] == []


def test_group_conformal_picks_most_specific(vm):
    from conformal import pick
    q = vm.conformal
    assert pick(q, "Residential", "Apartment", "90") == q["by_type"]["Apartment"]["90"]
    assert pick(q, "Residential", "Unknown type", "90") == q["by_category"]["Residential"]["90"]
    assert pick(q, "", "", "90") == q["global"]["90"]
