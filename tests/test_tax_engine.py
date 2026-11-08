import os
import pytest
import config
from tax_engine import PropertyInput, calculate_taxes, TaxEngine

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(config.TAX_RULES_DIR, "_default.json")), reason="no tax rules")


def test_unknown_state_uses_fallback():
    r = calculate_taxes(PropertyInput(state="Goa", city="Panaji", market_value=1e7))
    assert r.is_fallback_used and r.annual_property_tax > 0 and r.total_purchase_cost > 1e7


def test_missing_state_file_never_crashes():
    r = calculate_taxes(PropertyInput(state="Maharashtra", city="Pune", market_value=1e7))
    assert not r.warnings


def test_indexation_cutoff():
    assert TaxEngine._indexation_allowed(PropertyInput(holding_period_months=120))
    assert not TaxEngine._indexation_allowed(PropertyInput(holding_period_months=24))
