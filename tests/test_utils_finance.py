import math
import finance as f
from utils import fmt_inr, fmt_compact


def test_fmt_inr_grouping():
    assert fmt_inr(0) == "\u20b90"
    assert fmt_inr(999) == "\u20b9999"
    assert fmt_inr(1000) == "\u20b91,000"
    assert fmt_inr(12345678) == "\u20b91,23,45,678"
    assert fmt_inr(-150000) == "-\u20b91,50,000"
    assert fmt_inr("abc") == "N/A"


def test_fmt_compact():
    assert fmt_compact(12_500_000) == "\u20b91.25 Cr"
    assert fmt_compact(450_000) == "\u20b94.50 L"


def test_emi_known_value():
    assert math.isclose(f.emi(5_000_000, 8.5, 20), 43391.16, rel_tol=1e-4)
    assert f.emi(1200, 0, 1) == 100


def test_schedule_closes_and_sums():
    s = f.amortization_schedule(5_000_000, 8.5, 20)
    assert len(s) == 240 and s["balance"].iloc[-1] < 1
    assert math.isclose(s["principal"].sum(), 5_000_000, rel_tol=1e-6)


def test_prepayment_saves_interest_and_time():
    r = f.prepayment_impact(5_000_000, 8.5, 20, monthly_prepay=5000)
    assert r["interest_saved"] > 0 and r["months_saved"] > 0


def test_tax_benefit_caps_and_regimes():
    b = f.home_loan_tax_benefit(420_000, 100_000, 300_000, True, "old", 30, first_year=True)
    assert b.interest_deduction == 200_000 and b.sec80c_deduction == 150_000
    assert math.isclose(b.estimated_tax_saved, 350_000 * 0.30 * 1.04)
    assert f.home_loan_tax_benefit(420_000, 100_000, 0, True, "new").total_deduction == 0
    assert f.home_loan_tax_benefit(420_000, 0, 0, False, "new").interest_deduction == 420_000


def test_affordability_monotonic():
    a = f.affordability(150_000, 0, 8.5, 20, 20)
    assert a["max_emi"] == 67_500 and a["stressed_max_loan"] < a["max_loan"]
    assert f.affordability(150_000, 30_000, 8.5, 20, 20)["max_loan"] < a["max_loan"]


def test_rent_vs_buy_shape():
    d = f.rent_vs_buy(8e6, 5e5, 20, 8.5, 20, 25000, 5, 6, 10, 15)
    assert len(d) == 15 and d["loan_balance"].is_monotonic_decreasing
