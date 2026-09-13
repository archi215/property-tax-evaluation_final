"""
test_tax_engine.py
===================
Regression checks for tax_engine.py, anchored to worked examples so a future
change to the JSON rules or the formulas can't silently reintroduce a bug.

Run with: pytest test_tax_engine.py -v
     or:  python test_tax_engine.py   (plain assert runner, no pytest needed)
"""

from tax_engine import PropertyInput, calculate_taxes


def approx(a, b, tol=1.0):
    return abs(a - b) <= tol


def test_bbmp_applies_20pct_base_tax_rate():
    """BBMP formula is (taxable value) x 20%, then cess on top -- not
    taxable value taken directly as the base tax (that overcharged ~4.5x)."""
    r = calculate_taxes(PropertyInput(
        state="Karnataka", city="Bengaluru",
        built_up_area_sqft=1200, market_value=11_000_000,
        occupancy="Owner Occupied", pays_in_lump_sum=False,
    ))
    # zone C self-occupied: 1200 * 1.8/sqft/mo * 10 months = 21,600 gross annual value
    # default construction_year=2015 -> age 11 in this engine's 2026 clock -> 10% depreciation band
    # taxable_value = 21,600 * 0.90 = 19,440
    # base_tax = 19,440 * 20% = 3,888; cess = 3,888 * 24% = 933.12; gross = 4,821.12
    assert approx(r.annual_property_tax, 4821.12, tol=1.0), r.annual_property_tax
    assert r.property_tax_breakdown["gross_tax"] < 6000


def test_bmc_exempts_on_carpet_area_alone():
    """BMC's <=500 sqft exemption is on CARPET area alone; requiring built-up
    area to also be <=500 sqft made the exemption almost never fire."""
    r = calculate_taxes(PropertyInput(
        state="Maharashtra", city="Mumbai",
        built_up_area_sqft=620, carpet_area_sqft=480,  # built-up > 500, carpet <= 500
        market_value=12_000_000,
    ))
    assert r.annual_property_tax == 0.0
    assert r.property_tax_breakdown.get("exempt") is True


def test_bmc_does_not_exempt_above_carpet_threshold():
    r = calculate_taxes(PropertyInput(
        state="Maharashtra", city="Mumbai",
        built_up_area_sqft=800, carpet_area_sqft=650,
        market_value=25_000_000,
    ))
    assert r.annual_property_tax > 0


def test_no_phantom_capital_gains_without_purchase_price():
    """A plain valuation (no purchase_price given) must not invent a capital
    gains tax bill by assuming the property is being sold at a 35% markup."""
    r = calculate_taxes(PropertyInput(state="Delhi", city="Delhi", market_value=15_000_000))
    assert r.capital_gains_tax_estimate == 0.0
    assert "Original Purchase Price" in r.capital_gains_method


def test_capital_gains_computed_when_purchase_price_given():
    r = calculate_taxes(PropertyInput(
        state="Delhi", city="Delhi", market_value=15_000_000,
        purchase_price=9_000_000, holding_period_months=60,
    ))
    assert r.capital_gains_tax_estimate > 0


def test_short_term_gain_no_purchase_price_still_zero():
    r = calculate_taxes(PropertyInput(
        state="Delhi", city="Delhi", market_value=15_000_000, holding_period_months=6,
    ))
    assert r.capital_gains_tax_estimate == 0.0


def test_no_state_ever_crashes_the_engine():
    """The generic fallback must always produce a usable, warning-free result."""
    r = calculate_taxes(PropertyInput(state="Kerala", city="Kochi", market_value=8_000_000))
    assert r.is_fallback_used is True
    assert r.warnings == []
    assert r.annual_property_tax > 0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
