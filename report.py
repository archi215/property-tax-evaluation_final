"""
report.py -- one-click PDF valuation report (reportlab). No Streamlit dependency.

NOTE: reportlab's built-in Helvetica has no rupee glyph, so amounts are printed
as "Rs." in the PDF (the app UI itself uses the rupee symbol).
"""
from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from utils import fmt_inr

INDIGO, SAFFRON = colors.HexColor("#1B3A6B"), colors.HexColor("#FF8C42")


def _rs(v) -> str:
    return fmt_inr(v, symbol="Rs. ")


def _table(rows, col_widths=None, header=True):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style = [("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F2E9")])]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), INDIGO), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def build_pdf_report(*, inputs: dict, prediction, tax, loan: dict, drivers=None, recommendations=None,
                     scores: dict | None = None, model_meta: dict | None = None) -> bytes:
    """Returns PDF bytes. `prediction` = valuation.Prediction, `tax` = tax_engine.TaxBreakdown."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Property Valuation & Tax Report")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=INDIGO, fontSize=20, alignment=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=INDIGO, fontSize=13, spaceBefore=10)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, textColor=colors.grey)
    S = []

    S.append(Paragraph("Property Valuation &amp; Tax Report", h1))
    S.append(Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M} &nbsp;|&nbsp; "
                       f"{inputs.get('city', '')}, {inputs.get('state', '')} &nbsp;|&nbsp; "
                       f"{inputs.get('property_type', '')} ({inputs.get('property_category', '')})", body))
    S.append(Spacer(1, 6))

    S.append(Paragraph("1. Estimated market value", h2))
    p = prediction
    area = inputs.get("built_up_area_sqft") or inputs.get("plot_area_sqft") or 1
    S.append(_table([
        ["Metric", "Value"],
        ["Point estimate", _rs(p.price)],
        ["90% prediction interval", f"{_rs(p.lower_90)}  -  {_rs(p.upper_90)}"],
        ["80% prediction interval", f"{_rs(p.lower_80)}  -  {_rs(p.upper_80)}"],
        ["Implied price per sq.ft", _rs(p.price / max(area, 1))],
    ], [60 * mm, 110 * mm]))
    for w in p.warnings:
        S.append(Paragraph(f"Warning: {w}", small))

    S.append(Paragraph("2. Property summary", h2))
    keys = [("Built-up area (sq.ft)", "built_up_area_sqft"), ("Carpet area (sq.ft)", "carpet_area_sqft"),
            ("Construction year", "construction_year"), ("Bedrooms / Bathrooms", None),
            ("Construction quality", "construction_quality"), ("Occupancy", "occupancy")]
    rows = [["Attribute", "Value"]]
    for label, k in keys:
        rows.append([label, f"{inputs.get('bedrooms', '-')} / {inputs.get('bathrooms', '-')}" if k is None else str(inputs.get(k, "-"))])
    S.append(_table(rows, [60 * mm, 110 * mm]))

    if drivers is not None and len(drivers):
        S.append(Paragraph("3. What drives this price", h2))
        rows = [["Driver group", "Effect on price", "Approx. amount"]]
        for _, r in drivers.iterrows():
            rows.append([r["group"], f"{r['pct_effect']:+.1f}%", _rs(r["rupee_effect"])])
        S.append(_table(rows, [70 * mm, 40 * mm, 60 * mm]))
        S.append(Paragraph("Model attributions (SHAP) are relative to a baseline average property; they describe the model, not causal effects.", small))

    S.append(Paragraph("4. Taxes & government charges", h2))
    S.append(Paragraph(f"Rule source: <b>{tax.rule_source}</b>" + (" (generic fallback)" if tax.is_fallback_used else ""), body))
    S.append(_table([
        ["Item", "Amount"],
        ["Annual property tax", _rs(tax.annual_property_tax)], ["Recurring annual total (incl. cess/charges)", _rs(tax.recurring_annual_total)],
        ["Stamp duty", _rs(tax.stamp_duty)], ["Registration charges", _rs(tax.registration_charges)],
        ["Mutation charges", _rs(tax.mutation_charges)], ["One-time total", _rs(tax.one_time_total)],
        ["Total purchase cost", _rs(tax.total_purchase_cost)],
        ["Est. capital gains tax", _rs(tax.capital_gains_tax_estimate)],
    ], [100 * mm, 70 * mm]))

    S.append(Paragraph("5. Home-loan summary", h2))
    S.append(_table([
        ["Item", "Value"], ["Loan amount", _rs(loan.get("loan_amount", 0))],
        ["Rate / tenure", f"{loan.get('rate', 0):.2f}% / {loan.get('years', 0)} yrs"],
        ["Monthly EMI", _rs(loan.get("emi", 0))], ["Total interest", _rs(loan.get("total_interest", 0))],
        ["Est. yearly tax benefit (old regime, yr 1)", _rs(loan.get("tax_benefit_year1", 0))],
    ], [100 * mm, 70 * mm]))

    if scores:
        S.append(Paragraph("6. Scores", h2))
        S.append(Paragraph(f"Investment score: <b>{scores.get('investment', '-')}/100</b> &nbsp; Risk score: <b>{scores.get('risk', '-')}/100</b>", body))
    if recommendations:
        S.append(Paragraph("7. Recommendations", h2))
        for r in recommendations:
            S.append(Paragraph("&bull; " + r.replace("**", ""), body))

    S.append(Spacer(1, 10))
    meta = model_meta or {}
    S.append(Paragraph(
        "DISCLAIMER: Academic/portfolio estimate. The valuation model is trained on a synthetic-but-realistic dataset anchored to public 2025-26 "
        "market reports; it is not a certified valuation. Tax figures are estimates from rule files and must be verified with your municipal "
        f"corporation / a qualified professional. Model: XGBoost, test R2 = {meta.get('test_metrics', {}).get('r2', float('nan')):.3f}.", small))
    doc.build(S)
    return buf.getvalue()
