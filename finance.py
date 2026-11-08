"""
finance.py
==========
Pure financial calculations (no Streamlit, no ML): EMI, amortisation with
prepayments, home-loan tax benefits, affordability, rent-vs-buy and rental ROI.

All tax-benefit figures follow the OLD income-tax regime rules (Sec 24(b)
interest up to Rs 2 lakh for a self-occupied home, Sec 80C up to Rs 1.5 lakh
for principal + stamp duty/registration). The NEW regime allows neither for a
self-occupied property. These are estimates -- confirm with a tax professional.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEC_24B_CAP_SELF_OCCUPIED = 200_000
SEC_80C_CAP = 150_000
CESS = 0.04


# ───────────────────────── EMI & amortisation ─────────────────────────────
def emi(principal: float, annual_rate_pct: float, years: float) -> float:
    n = int(round(years * 12))
    if principal <= 0 or n <= 0:
        return 0.0
    r = annual_rate_pct / 1200.0
    if r == 0:
        return principal / n
    f = (1 + r) ** n
    return principal * r * f / (f - 1)


def calculate_mortgage(principal: float, annual_rate_pct: float, years: int):
    """Returns (emi, total_payment, total_interest). Kept for back-compat with the old app."""
    e = emi(principal, annual_rate_pct, years)
    total = e * years * 12
    return e, total, total - principal if principal > 0 else 0.0


def amortization_schedule(principal: float, annual_rate_pct: float, years: int,
                          monthly_prepay: float = 0.0, lump_sum: float = 0.0,
                          lump_sum_month: int = 12) -> pd.DataFrame:
    """Month-by-month schedule. Prepayments reduce tenure (EMI stays constant)."""
    base_emi = emi(principal, annual_rate_pct, years)
    r = annual_rate_pct / 1200.0
    bal = float(principal)
    rows, m = [], 0
    max_months = int(years * 12)
    while bal > 0.5 and m < max_months:
        m += 1
        interest = bal * r
        princ = min(bal, base_emi - interest) if base_emi > interest else bal
        extra = monthly_prepay
        if lump_sum and m == lump_sum_month:
            extra += lump_sum
        extra = min(extra, max(bal - princ, 0.0))
        bal = bal - princ - extra
        rows.append({"month": m, "year": (m - 1) // 12 + 1, "emi": princ + interest,
                     "interest": interest, "principal": princ, "prepayment": extra,
                     "balance": max(bal, 0.0)})
    return pd.DataFrame(rows)


def prepayment_impact(principal, rate, years, monthly_prepay=0.0, lump_sum=0.0, lump_sum_month=12) -> dict:
    base = amortization_schedule(principal, rate, years)
    new = amortization_schedule(principal, rate, years, monthly_prepay, lump_sum, lump_sum_month)
    return {
        "base_interest": float(base["interest"].sum()), "new_interest": float(new["interest"].sum()),
        "interest_saved": float(base["interest"].sum() - new["interest"].sum()),
        "base_months": len(base), "new_months": len(new), "months_saved": len(base) - len(new),
    }


def yearly_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    g = schedule.groupby("year").agg(interest=("interest", "sum"), principal=("principal", "sum"),
                                     prepayment=("prepayment", "sum"), closing_balance=("balance", "last"))
    return g.reset_index()


# ───────────────────────── Tax benefits (old regime) ──────────────────────
@dataclass
class TaxBenefit:
    interest_deduction: float
    sec80c_deduction: float
    total_deduction: float
    estimated_tax_saved: float
    note: str


def home_loan_tax_benefit(year_interest: float, year_principal: float, stamp_and_reg: float = 0.0,
                          self_occupied: bool = True, regime: str = "old",
                          marginal_rate_pct: float = 30.0, other_80c_used: float = 0.0,
                          first_year: bool = False) -> TaxBenefit:
    if regime != "old" and self_occupied:
        return TaxBenefit(0, 0, 0, 0, "New regime: no Sec 24(b)/80C deduction for a self-occupied home.")
    if self_occupied:
        interest_ded = min(year_interest, SEC_24B_CAP_SELF_OCCUPIED)
    else:
        # let-out: interest fully deductible against rent; loss set-off vs other income capped at Rs 2L
        interest_ded = year_interest
    c80 = 0.0
    if regime == "old":
        pool = year_principal + (stamp_and_reg if first_year else 0.0)
        c80 = min(pool, max(SEC_80C_CAP - other_80c_used, 0.0))
    total = interest_ded + c80
    saved = total * marginal_rate_pct / 100.0 * (1 + CESS)
    note = "Old-regime estimate; assumes deductions fully absorbed by taxable income."
    if not self_occupied:
        note += " For a let-out home, set-off of the resulting loss against other income is capped at Rs 2 lakh/yr."
    return TaxBenefit(interest_ded, c80, total, saved, note)


# ───────────────────────── Affordability ──────────────────────────────────
def affordability(monthly_income: float, existing_emis: float, rate_pct: float, years: int,
                  down_payment_pct: float, foir_pct: float = 45.0, stress_bump_pct: float = 2.0) -> dict:
    """Max EMI = FOIR * income - existing EMIs (FOIR = fixed obligations to income ratio)."""
    max_emi = max(monthly_income * foir_pct / 100.0 - existing_emis, 0.0)

    def loan_for(e, rate):
        n = years * 12
        r = rate / 1200.0
        if e <= 0:
            return 0.0
        if r == 0:
            return e * n
        return e * ((1 + r) ** n - 1) / (r * (1 + r) ** n)

    max_loan = loan_for(max_emi, rate_pct)
    stressed_loan = loan_for(max_emi, rate_pct + stress_bump_pct)
    dp = down_payment_pct / 100.0
    price = max_loan / (1 - dp) if dp < 1 else 0.0
    return {"max_emi": max_emi, "max_loan": max_loan, "max_property_price": price,
            "stressed_max_loan": stressed_loan, "stressed_max_price": stressed_loan / (1 - dp) if dp < 1 else 0.0}


def emi_to_income_ratio(emi_amount: float, monthly_income: float, existing_emis: float = 0.0) -> float:
    return (emi_amount + existing_emis) / monthly_income * 100.0 if monthly_income > 0 else float("inf")


# ───────────────────────── Rent vs buy ────────────────────────────────────
def rent_vs_buy(price: float, closing_costs: float, down_payment_pct: float, loan_rate_pct: float,
                tenure_years: int, monthly_rent: float, rent_growth_pct: float,
                appreciation_pct: float, investment_return_pct: float, years: int,
                annual_maintenance: float = 0.0, annual_property_tax: float = 0.0,
                selling_cost_pct: float = 2.0) -> pd.DataFrame:
    """Compares net worth after each year: owning (equity) vs renting + investing the difference."""
    loan = price * (1 - down_payment_pct / 100.0)
    e = emi(loan, loan_rate_pct, tenure_years)
    sched = amortization_schedule(loan, loan_rate_pct, tenure_years)
    balance_by_month = {0: loan, **dict(zip(sched["month"], sched["balance"]))}
    r_inv = (1 + investment_return_pct / 100.0) ** (1 / 12) - 1
    corpus = price * down_payment_pct / 100.0 + closing_costs
    rent = monthly_rent
    rows = []
    for m in range(1, years * 12 + 1):
        if m > 1 and (m - 1) % 12 == 0:
            rent *= 1 + rent_growth_pct / 100.0
        owning_outflow = (e if m <= tenure_years * 12 else 0.0) + (annual_maintenance + annual_property_tax) / 12
        corpus = corpus * (1 + r_inv) + (owning_outflow - rent)
        if m % 12 == 0:
            y = m // 12
            value = price * (1 + appreciation_pct / 100.0) ** y
            bal = balance_by_month.get(m, 0.0)
            equity = value * (1 - selling_cost_pct / 100.0) - bal
            rows.append({"year": y, "property_value": value, "loan_balance": bal,
                         "buy_net_worth": equity, "rent_net_worth": corpus,
                         "monthly_rent": rent, "advantage_buy": equity - corpus})
    return pd.DataFrame(rows)


def breakeven_year(df: pd.DataFrame):
    pos = df[df["advantage_buy"] > 0]
    return int(pos["year"].iloc[0]) if not pos.empty else None


# ───────────────────────── Rental ROI & appreciation ─────────────────────
def rental_roi(price: float, closing_costs: float, annual_rent: float, annual_property_tax: float,
               annual_maintenance: float, vacancy_pct: float = 5.0) -> dict:
    effective_rent = annual_rent * (1 - vacancy_pct / 100.0)
    net = effective_rent - annual_property_tax - annual_maintenance
    invested = price + closing_costs
    return {"gross_yield_pct": annual_rent / price * 100 if price else 0.0,
            "net_yield_pct": net / invested * 100 if invested else 0.0,
            "net_annual_income": net,
            "payback_years": invested / net if net > 0 else float("inf")}


def project_value(price: float, years: int, cagr_pct: float) -> float:
    return price * (1 + cagr_pct / 100.0) ** years


def appreciation_scenarios(price: float, horizon: int = 15, bear=3.0, base=6.0, bull=9.0) -> pd.DataFrame:
    yrs = np.arange(0, horizon + 1)
    return pd.DataFrame({"year": yrs, "bear": price * (1 + bear / 100) ** yrs,
                         "base": price * (1 + base / 100) ** yrs, "bull": price * (1 + bull / 100) ** yrs})
