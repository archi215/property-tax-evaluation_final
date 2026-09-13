"""
tax_engine.py
=============
Core Indian Property Tax Calculation Engine.

Loads state/city-specific rules from tax_rules/*.json and computes every
applicable tax for a given property. Falls back to tax_rules/_default.json
(a clearly-labelled generic approximation) for any state without a
dedicated rule file, so the application never crashes on unmapped
locations.

Supported "real" rulesets (researched from public sources, see each
JSON's "sources" key):
    - Delhi.json        -> MCD Unit Area System
    - Maharashtra.json  -> BMC (Mumbai) + PMC (Pune) Capital Value System
    - Karnataka.json    -> BBMP Unit Area Value System

Every other state uses the generic fallback in _default.json, which is
clearly disclaimed in the JSON itself and surfaced to the user in the UI.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tax_engine")

TAX_RULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tax_rules")

# "Today" for age/holding-period math. Kept as a fixed date (matching the 2026
# hardcoded elsewhere in this module, e.g. age_years = 2026 - construction_year)
# rather than date.today(), so results stay reproducible across runs.
_CURRENT_DATE = date(2026, 1, 1)


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    """Parse a 'YYYY-MM-DD' string from a tax_rules JSON; None if missing/invalid."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────
# Input data structure
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class PropertyInput:
    """All fields the tax engine needs. Sensible defaults are provided so
    partially-filled forms from the Streamlit UI never raise KeyErrors."""
    state: str = "Delhi"
    city: str = "Delhi"
    property_type: str = "Apartment"          # Apartment/Villa/Independent House/Plot/Office/Shop/Warehouse
    property_category: str = "Residential"    # Residential/Commercial/Industrial/Land
    built_up_area_sqft: float = 1000.0
    carpet_area_sqft: float = 0.0
    plot_area_sqft: float = 0.0
    construction_year: int = 2015
    market_value: float = 5000000.0
    circle_rate_value: Optional[float] = None
    govt_guideline_value: Optional[float] = None
    occupancy: str = "Owner Occupied"          # Owner Occupied / Rented / Vacant
    buyer_gender: str = "Male"                 # Male / Female / Joint
    construction_quality: str = "Standard"     # Basic/Standard/Premium/Luxury
    structure_type: str = "RCC"                # RCC / Non-RCC / Load Bearing / Chawl ...
    is_senior_citizen: bool = False
    is_woman_owner: bool = False
    is_disabled_or_exserviceman: bool = False
    has_solar_panels: bool = False
    has_rainwater_harvesting: bool = False
    is_green_certified: bool = False
    zone_or_category: Optional[str] = None     # MCD A-H / BBMP A-F / Maharashtra zone key, optional override
    pays_in_lump_sum: bool = True
    annual_rental_income: float = 0.0
    holding_period_months: int = 36
    purchase_price: Optional[float] = None     # for capital gains; defaults to market_value if absent
    inflation_rate_pct: float = 6.0            # user-adjustable for projections


@dataclass
class TaxBreakdown:
    """Structured result returned by TaxEngine.calculate()."""
    state: str = ""
    city: str = ""
    rule_source: str = ""               # which JSON file / fallback was used
    is_fallback_used: bool = False
    disclaimer: str = ""

    annual_property_tax: float = 0.0
    monthly_property_tax: float = 0.0
    property_tax_breakdown: dict = field(default_factory=dict)

    water_tax: float = 0.0
    sewerage_tax: float = 0.0
    solid_waste_management_charge: float = 0.0
    education_cess: float = 0.0
    fire_cess: float = 0.0
    health_cess: float = 0.0
    urban_development_cess: float = 0.0
    luxury_property_tax: float = 0.0
    green_tax_rebate: float = 0.0

    recurring_annual_total: float = 0.0

    stamp_duty: float = 0.0
    registration_charges: float = 0.0
    transfer_charges: float = 0.0
    mutation_charges: float = 0.0
    one_time_total: float = 0.0

    capital_gains_tax_estimate: float = 0.0
    capital_gains_method: str = ""
    rental_income_tax_estimate: float = 0.0

    rebates_applied: dict = field(default_factory=dict)
    total_rebate_amount: float = 0.0

    total_purchase_cost: float = 0.0
    projections: dict = field(default_factory=dict)

    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


# ──────────────────────────────────────────────────────────────────────────
# Rule loading
# ──────────────────────────────────────────────────────────────────────────
class TaxRuleRepository:
    """Loads and caches tax_rules/*.json files."""

    def __init__(self, rules_dir: str = TAX_RULES_DIR):
        self.rules_dir = rules_dir
        self._cache: dict[str, dict] = {}

    def _load_file(self, filename: str) -> dict:
        if filename in self._cache:
            return self._cache[filename]
        path = os.path.join(self.rules_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._cache[filename] = data
            return data
        except FileNotFoundError:
            logger.warning("Tax rule file not found: %s", path)
            return {}
        except json.JSONDecodeError as e:
            logger.error("Malformed JSON in %s: %s", path, e)
            return {}

    def get_rules_for_state(self, state: str) -> tuple[dict, bool, str]:
        """
        Returns (rules_dict, is_fallback_used, source_filename).
        Looks for tax_rules/<state>.json (case-insensitive, spaces stripped);
        falls back to _default.json if not found.
        """
        if not state:
            state = "_default_"
        normalized = state.strip().replace(" ", "")
        # try a few common filename variants
        candidates = [f"{state.strip()}.json", f"{normalized}.json"]
        for candidate in candidates:
            for existing in os.listdir(self.rules_dir) if os.path.isdir(self.rules_dir) else []:
                if existing.lower() == candidate.lower():
                    data = self._load_file(existing)
                    if data:
                        return data, False, existing
        # fallback
        data = self._load_file("_default.json")
        return data, True, "_default.json"


# ──────────────────────────────────────────────────────────────────────────
# Core engine
# ──────────────────────────────────────────────────────────────────────────
class TaxEngine:
    def __init__(self, rules_dir: str = TAX_RULES_DIR):
        self.repo = TaxRuleRepository(rules_dir)

    # ---- public API -----------------------------------------------------
    def calculate(self, prop: PropertyInput) -> TaxBreakdown:
        result = TaxBreakdown(state=prop.state, city=prop.city)

        rules, is_fallback, source = self.repo.get_rules_for_state(prop.state)
        result.rule_source = source
        result.is_fallback_used = is_fallback
        result.disclaimer = rules.get("disclaimer", "")

        if not rules:
            result.warnings.append(
                f"No tax rules could be loaded for state '{prop.state}'. "
                "Returning zeroed estimates -- please verify inputs."
            )
            return result

        try:
            self._compute_property_tax(prop, rules, result)
        except Exception as e:
            logger.exception("property_tax computation failed")
            result.warnings.append(f"Property tax calculation failed safely: {e}")

        try:
            self._compute_recurring_charges(prop, rules, result)
        except Exception as e:
            logger.exception("recurring charges computation failed")
            result.warnings.append(f"Recurring charges calculation failed safely: {e}")

        try:
            self._compute_one_time_charges(prop, rules, result)
        except Exception as e:
            logger.exception("one-time charges computation failed")
            result.warnings.append(f"One-time charges calculation failed safely: {e}")

        try:
            self._compute_capital_gains(prop, rules, result)
        except Exception as e:
            logger.exception("capital gains computation failed")
            result.warnings.append(f"Capital gains calculation failed safely: {e}")

        try:
            self._compute_rental_income_tax(prop, rules, result)
        except Exception as e:
            logger.exception("rental income tax computation failed")
            result.warnings.append(f"Rental income tax calculation failed safely: {e}")

        try:
            self._compute_total_purchase_cost(prop, result)
        except Exception as e:
            logger.exception("total purchase cost computation failed")
            result.warnings.append(f"Total cost calculation failed safely: {e}")

        try:
            self._compute_projections(prop, result)
        except Exception as e:
            logger.exception("projection computation failed")
            result.warnings.append(f"Projection calculation failed safely: {e}")

        return result

    # ---- property tax -----------------------------------------------------
    def _compute_property_tax(self, prop: PropertyInput, rules: dict, result: TaxBreakdown):
        state = prop.state.strip().lower()
        city = prop.city.strip().lower()

        if state == "delhi":
            self._property_tax_delhi(prop, rules, result)
        elif state == "maharashtra":
            self._property_tax_maharashtra(prop, rules, result, city)
        elif state == "karnataka":
            self._property_tax_karnataka(prop, rules, result)
        else:
            self._property_tax_generic(prop, rules, result)

        result.annual_property_tax = round(result.annual_property_tax, 2)
        result.monthly_property_tax = round(result.annual_property_tax / 12, 2)

    def _age_factor(self, age_years: int, bands: list[dict]) -> float:
        for band in sorted(bands, key=lambda b: b["max_age_years"]):
            if age_years <= band["max_age_years"]:
                return band["factor"]
        return bands[-1]["factor"] if bands else 1.0

    def _property_tax_delhi(self, prop: PropertyInput, rules: dict, result: TaxBreakdown):
        pt = rules["property_tax"]
        category_key = prop.zone_or_category or pt.get("default_category", "C")
        category_key = category_key.upper() if category_key.upper() in pt["categories"] else pt.get("default_category", "C")
        cat = pt["categories"][category_key]

        uav = cat["unit_area_value_per_sqm"]
        # For vacant land/plots, built_up_area_sqft is typically 0 -- use plot_area instead.
        effective_area_sqft = prop.built_up_area_sqft if prop.built_up_area_sqft > 0 else prop.plot_area_sqft
        area_sqm = max(effective_area_sqft, 1) * 0.0929  # sqft -> sqm

        is_commercial = prop.property_category in ("Commercial",)
        is_industrial = prop.property_category in ("Industrial",)
        use_factor = pt["use_factor"].get(prop.property_category, 1.0)
        structure_factor = pt["structure_factor"].get(prop.structure_type, 1.0)
        occupancy_factor = pt["occupancy_factor"].get(prop.occupancy, 1.0)

        age_years = max(0, 2026 - prop.construction_year)
        age_factor = self._age_factor(age_years, pt["age_factor_bands"])

        annual_value = uav * area_sqm * age_factor * use_factor * structure_factor * occupancy_factor

        if is_industrial:
            rate_pct = pt["industrial_rate_pct_of_annual_value"]
        elif is_commercial:
            rate_pct = cat["commercial_rate_pct"]
        elif prop.property_category == "Land":
            rate_pct = pt["vacant_land_rate_pct_of_annual_value"]
        else:
            rate_pct = cat["residential_rate_pct"]

        gross_tax = annual_value * rate_pct / 100.0

        # rebates
        rebates = pt["rebates"]
        rebate_amount = 0.0
        applied = {}
        if prop.pays_in_lump_sum:
            amt = gross_tax * rebates["lump_sum_first_quarter_pct"] / 100.0
            rebate_amount += amt
            applied["lump_sum_payment_rebate"] = round(amt, 2)
        if (prop.is_senior_citizen or prop.is_woman_owner or prop.is_disabled_or_exserviceman) \
                and prop.occupancy == "Owner Occupied":
            amt = gross_tax * rebates["senior_citizen_women_disabled_exservicemen_pct"] / 100.0
            rebate_amount += amt
            applied["senior_citizen_women_disabled_rebate"] = round(amt, 2)

        net_tax = max(0.0, gross_tax - rebate_amount)

        result.annual_property_tax = net_tax
        result.property_tax_breakdown = {
            "category": category_key,
            "unit_area_value_per_sqm": uav,
            "area_sqm": round(area_sqm, 2),
            "age_factor": age_factor,
            "use_factor": use_factor,
            "structure_factor": structure_factor,
            "occupancy_factor": occupancy_factor,
            "annual_value": round(annual_value, 2),
            "tax_rate_pct": rate_pct,
            "gross_tax": round(gross_tax, 2),
        }
        result.rebates_applied.update(applied)
        result.total_rebate_amount += rebate_amount

    def _property_tax_maharashtra(self, prop: PropertyInput, rules: dict, result: TaxBreakdown, city: str):
        cities = rules.get("cities", {})
        city_key = None
        for k in cities:
            if k.lower() == city:
                city_key = k
                break

        if city_key == "Mumbai":
            self._property_tax_bmc(prop, cities["Mumbai"]["property_tax"], result)
        elif city_key == "Pune":
            self._property_tax_pmc(prop, cities["Pune"]["property_tax"], result)
        else:
            # generic Maharashtra fallback for other cities (Nagpur, Thane, Navi Mumbai, etc.)
            fb = rules["default_city_fallback"]
            area = prop.built_up_area_sqft if prop.built_up_area_sqft > 0 else max(prop.plot_area_sqft, 1)
            base = fb["base_value_per_sqft"]
            age_years = max(0, 2026 - prop.construction_year)
            age_factor = max(0.6, 1.0 - age_years * 0.01)
            if prop.property_category == "Land":
                rate_key = "industrial"  # closest available band; land has no dedicated key in fallback
            elif prop.property_category == "Commercial":
                rate_key = "commercial"
            elif prop.property_category == "Industrial":
                rate_key = "industrial"
            elif prop.occupancy == "Owner Occupied":
                rate_key = "self_occupied_residential"
            else:
                rate_key = "rented_residential"
            rate_pct = fb["tax_rate_pct"][rate_key]
            capital_value = base * area * age_factor
            tax = capital_value * rate_pct / 100.0
            result.annual_property_tax = tax
            result.property_tax_breakdown = {
                "method": "maharashtra_generic_fallback",
                "base_value_per_sqft": base,
                "capital_value": round(capital_value, 2),
                "tax_rate_pct": rate_pct,
            }

    def _property_tax_bmc(self, prop: PropertyInput, pt: dict, result: TaxBreakdown):
        zone_key = prop.zone_or_category or pt.get("default_zone", "C_Central_Suburbs")
        zone_values = pt["zone_capital_value_per_sqft"]
        zone_key = zone_key if zone_key in zone_values else pt.get("default_zone", "C_Central_Suburbs")
        zone_csqft = zone_values[zone_key]

        area = prop.built_up_area_sqft if prop.built_up_area_sqft > 0 else max(prop.plot_area_sqft, 1)
        # BMC exempts on CARPET area alone (<=500 sqft), regardless of built-up area.
        if prop.carpet_area_sqft and prop.carpet_area_sqft <= pt.get("carpet_area_exemption_sqft", 500):
            result.annual_property_tax = 0.0
            result.property_tax_breakdown = {"exempt": True, "reason": "Carpet area <= 500 sqft BMC exemption"}
            return

        building_factor = pt["building_factor"].get(prop.structure_type, 1.0)
        age_years = max(0, 2026 - prop.construction_year)
        max_dep = pt["age_depreciation_max_pct"] / 100.0
        full_at = pt["age_depreciation_full_at_years"]
        age_depreciation = min(max_dep, (age_years / full_at) * max_dep) if full_at else 0.0
        floor_factor = 1.0  # apartment floor number not modelled separately here; conservative default

        capital_value = area * zone_csqft * building_factor * floor_factor * (1 - age_depreciation)

        if prop.property_category == "Commercial":
            rate_pct = pt["tax_rate_pct"]["commercial"]
        elif prop.property_category == "Industrial":
            rate_pct = pt["tax_rate_pct"]["industrial"]
        elif prop.occupancy == "Owner Occupied":
            rate_pct = pt["tax_rate_pct"]["self_occupied_residential"]
        else:
            rate_pct = pt["tax_rate_pct"]["rented_residential"]

        tax = capital_value * rate_pct / 100.0
        result.annual_property_tax = tax
        result.property_tax_breakdown = {
            "zone": zone_key,
            "zone_capital_value_per_sqft": zone_csqft,
            "building_factor": building_factor,
            "age_depreciation_pct": round(age_depreciation * 100, 2),
            "capital_value": round(capital_value, 2),
            "tax_rate_pct": rate_pct,
        }

    def _property_tax_pmc(self, prop: PropertyInput, pt: dict, result: TaxBreakdown):
        zone_key = prop.zone_or_category or pt.get("default_zone", "Pune_West")
        base_values = pt["base_value_per_sqft"]
        zone_key = zone_key if zone_key in base_values else pt.get("default_zone", "Pune_West")
        base_value = base_values[zone_key]

        area = prop.built_up_area_sqft if prop.built_up_area_sqft > 0 else max(prop.plot_area_sqft, 1)
        use_factor = pt["use_category_factor"].get(prop.property_category, 1.0)
        building_factor = pt["building_type_factor"].get(prop.structure_type, 1.0)
        age_years = max(0, 2026 - prop.construction_year)
        age_factor = self._age_factor(age_years, pt["age_factor_bands"])

        capital_value = base_value * area * use_factor * building_factor * age_factor

        if prop.property_category == "Commercial":
            rate_pct = pt["tax_rate_pct"]["commercial"]
        elif prop.property_category == "Industrial":
            rate_pct = pt["tax_rate_pct"]["industrial"]
        elif prop.occupancy == "Owner Occupied":
            rate_pct = pt["tax_rate_pct"]["self_occupied_residential"]
        else:
            rate_pct = pt["tax_rate_pct"]["rented_residential"]

        gross_tax = capital_value * rate_pct / 100.0

        rebate_amount = 0.0
        applied = {}
        if prop.occupancy == "Owner Occupied" and prop.property_category == "Residential":
            amt = gross_tax * pt["self_occupancy_rebate_pct"] / 100.0
            rebate_amount += amt
            applied["self_occupancy_rebate"] = round(amt, 2)

        annual_value_for_slab = gross_tax  # used only to pick early-payment slab
        if prop.pays_in_lump_sum:
            for slab in pt["early_payment_rebate"]:
                if annual_value_for_slab <= slab["max_annual_value"]:
                    amt = (gross_tax - rebate_amount) * slab["rebate_pct"] / 100.0
                    rebate_amount += amt
                    applied["early_payment_rebate"] = round(amt, 2)
                    break

        green_features = sum([prop.has_solar_panels, prop.has_rainwater_harvesting, prop.is_green_certified])
        if green_features > 0:
            pct = min(pt["green_rebate_max_pct"], green_features * pt["green_rebate_pct_per_eco_feature"])
            amt = (gross_tax - rebate_amount) * pct / 100.0
            rebate_amount += amt
            applied["green_building_rebate"] = round(amt, 2)

        net_tax = max(0.0, gross_tax - rebate_amount)

        result.annual_property_tax = net_tax
        result.property_tax_breakdown = {
            "zone": zone_key,
            "base_value_per_sqft": base_value,
            "use_factor": use_factor,
            "building_factor": building_factor,
            "age_factor": age_factor,
            "capital_value": round(capital_value, 2),
            "tax_rate_pct": rate_pct,
            "gross_tax": round(gross_tax, 2),
        }
        result.rebates_applied.update(applied)
        result.total_rebate_amount += rebate_amount

    def _property_tax_karnataka(self, prop: PropertyInput, rules: dict, result: TaxBreakdown):
        pt = rules["property_tax"]
        zone_key = prop.zone_or_category or pt.get("default_zone", "C")
        zone_key = zone_key.upper() if zone_key.upper() in pt["zones"] else pt.get("default_zone", "C")
        zone = pt["zones"][zone_key]

        area = prop.built_up_area_sqft if prop.built_up_area_sqft > 0 else max(prop.plot_area_sqft, 1)
        rate_per_sqft_month = zone["self_occupied_per_sqft_month"] if prop.occupancy == "Owner Occupied" \
            else zone["tenanted_per_sqft_month"]

        multiplier = 1.0
        if prop.property_category == "Commercial":
            multiplier = pt["commercial_multiplier"]
        elif prop.property_category == "Industrial":
            multiplier = pt["industrial_multiplier"]
        elif prop.property_category == "Land":
            multiplier = pt["vacant_land_use_factor"]

        gross_annual_value = area * rate_per_sqft_month * multiplier * pt["months_multiplier"]

        age_years = max(0, 2026 - prop.construction_year)
        depreciation_pct = 0
        for band in sorted(pt["depreciation_by_age"], key=lambda b: b["max_age_years"]):
            if age_years <= band["max_age_years"]:
                depreciation_pct = band["depreciation_pct"]
                break
        else:
            depreciation_pct = pt["depreciation_by_age"][-1]["depreciation_pct"]

        depreciation_amount = gross_annual_value * depreciation_pct / 100.0
        taxable_value = gross_annual_value - depreciation_amount

        # BBMP formula (public guides): Property Tax = (taxable value) x 20%, then cess on top.
        # The 20% was previously omitted, overcharging every Bengaluru estimate ~5x.
        base_tax = taxable_value * pt.get("base_tax_rate_pct", 20) / 100.0
        cess_pct = pt["cess_pct_of_base_tax"]
        cess_amount = base_tax * cess_pct / 100.0

        gross_tax = base_tax + cess_amount

        rebate_amount = 0.0
        applied = {}
        if prop.pays_in_lump_sum:
            amt = gross_tax * pt["rebate_full_payment_pct"] / 100.0
            rebate_amount += amt
            applied["full_payment_rebate"] = round(amt, 2)

        net_tax = max(0.0, gross_tax - rebate_amount)

        result.annual_property_tax = net_tax
        result.property_tax_breakdown = {
            "zone": zone_key,
            "rate_per_sqft_month": rate_per_sqft_month,
            "gross_annual_value": round(gross_annual_value, 2),
            "depreciation_pct": depreciation_pct,
            "taxable_value": round(taxable_value, 2),
            "cess_pct": cess_pct,
            "cess_amount": round(cess_amount, 2),
            "gross_tax": round(gross_tax, 2),
        }
        result.rebates_applied.update(applied)
        result.total_rebate_amount += rebate_amount
        result.health_cess = round(base_tax * pt["cess_breakdown"]["health_cess_pct"] / 100.0, 2)

    def _property_tax_generic(self, prop: PropertyInput, rules: dict, result: TaxBreakdown):
        pt = rules["property_tax"]
        arv_pct = pt["annual_rateable_value_pct_of_market_price"]
        annual_rateable_value = prop.market_value * arv_pct / 100.0

        age_years = max(0, 2026 - prop.construction_year)
        age_factor = self._age_factor(age_years, pt["age_factor_bands"])

        if prop.property_category == "Commercial":
            rate_key = "commercial"
        elif prop.property_category == "Industrial":
            rate_key = "industrial"
        elif prop.property_category == "Land":
            rate_key = "land"
        elif prop.occupancy == "Owner Occupied":
            rate_key = "self_occupied_residential"
        else:
            rate_key = "rented_residential"
        rate_pct = pt["tax_rate_pct"][rate_key]

        gross_tax = annual_rateable_value * age_factor * rate_pct / 100.0

        rebate_amount = 0.0
        applied = {}
        if prop.pays_in_lump_sum:
            amt = gross_tax * pt["rebate_full_payment_pct"] / 100.0
            rebate_amount += amt
            applied["full_payment_rebate"] = round(amt, 2)

        net_tax = max(0.0, gross_tax - rebate_amount)
        result.annual_property_tax = net_tax
        result.property_tax_breakdown = {
            "method": "generic_fallback_arv",
            "annual_rateable_value": round(annual_rateable_value, 2),
            "age_factor": age_factor,
            "tax_rate_pct": rate_pct,
            "gross_tax": round(gross_tax, 2),
        }
        result.rebates_applied.update(applied)
        result.total_rebate_amount += rebate_amount

    # ---- recurring municipal charges ---------------------------------------
    def _compute_recurring_charges(self, prop: PropertyInput, rules: dict, result: TaxBreakdown):
        base = result.annual_property_tax

        def pct_of_base(key, default=0.0):
            block = rules.get(key, {})
            return base * block.get("flat_annual_estimate_pct_of_property_tax", default) / 100.0

        result.water_tax = round(pct_of_base("water_tax"), 2)
        result.sewerage_tax = round(pct_of_base("sewerage_tax"), 2)
        result.solid_waste_management_charge = round(rules.get("solid_waste_management_charge_flat_annual", 0.0), 2)
        result.education_cess = round(base * rules.get("education_cess_pct_of_property_tax", 0) / 100.0, 2)
        result.fire_cess = round(base * rules.get("fire_cess_pct_of_property_tax", 0) / 100.0, 2)
        result.urban_development_cess = round(base * rules.get("urban_development_cess_pct_of_property_tax", 0) / 100.0, 2)

        luxury = rules.get("luxury_property_tax", {})
        if prop.market_value >= luxury.get("threshold_market_value", float("inf")):
            result.luxury_property_tax = round(prop.market_value * luxury.get("additional_cess_pct", 0) / 100.0, 2)

        green = rules.get("green_tax", {})
        green_rebate_pct = green.get("rebate_pct_with_solar_or_rainwater", 0)
        if green_rebate_pct and (prop.has_solar_panels or prop.has_rainwater_harvesting):
            result.green_tax_rebate = round(base * green_rebate_pct / 100.0, 2)

        result.recurring_annual_total = round(
            result.annual_property_tax + result.water_tax + result.sewerage_tax +
            result.solid_waste_management_charge + result.education_cess + result.fire_cess +
            result.health_cess + result.urban_development_cess + result.luxury_property_tax -
            result.green_tax_rebate,
            2
        )

    # ---- one-time charges (purchase-time) ----------------------------------
    def _compute_one_time_charges(self, prop: PropertyInput, rules: dict, result: TaxBreakdown):
        state = prop.state.strip().lower()
        sd = rules.get("stamp_duty", {})
        value_for_duty = max(prop.market_value, prop.circle_rate_value or 0, prop.govt_guideline_value or 0)

        if state == "maharashtra":
            # use city-specific stamp duty if available
            cities = rules.get("cities", {})
            city_key = next((k for k in cities if k.lower() == prop.city.strip().lower()), None)
            if city_key:
                csd = cities[city_key]["stamp_duty_pct"]
                rate = csd.get({"Male": "male", "Female": "female", "Joint": "joint"}.get(prop.buyer_gender, "male"), csd.get("male", 6))
                extra_pct = csd.get("metro_cess_pct", 0) + csd.get("local_body_tax_pct", 0)
                stamp_duty_pct = rate + extra_pct
            else:
                stamp_duty_pct = sd.get("municipal_corporation_area_pct", 6)
                if prop.buyer_gender == "Female":
                    stamp_duty_pct -= sd.get("female_concession_pct", 0)
            registration_pct = sd.get("registration_fee_pct", 1)
            registration_cap = sd.get("registration_fee_cap")
            stamp_duty = value_for_duty * stamp_duty_pct / 100.0
            registration_charges = value_for_duty * registration_pct / 100.0
            if registration_cap:
                registration_charges = min(registration_charges, registration_cap)

        elif state == "karnataka":
            slabs = sd.get("slabs", [])
            stamp_duty_pct = next((s["rate_pct"] for s in slabs if value_for_duty <= s["max_value"]), slabs[-1]["rate_pct"] if slabs else 5)
            stamp_duty = value_for_duty * stamp_duty_pct / 100.0
            registration_pct = sd.get("registration_fee_pct", 2)
            registration_charges = value_for_duty * registration_pct / 100.0

        elif state == "delhi":
            gender_key = {"Male": "male_pct", "Female": "female_pct", "Joint": "joint_pct"}.get(prop.buyer_gender, "male_pct")
            stamp_duty_pct = sd.get(gender_key, sd.get("male_pct", 6))
            stamp_duty = value_for_duty * stamp_duty_pct / 100.0
            registration_pct = sd.get("registration_fee_pct", 1)
            registration_charges = value_for_duty * registration_pct / 100.0

        else:
            gender_key = {"Male": "male_pct", "Female": "female_pct", "Joint": "joint_pct"}.get(prop.buyer_gender, "male_pct")
            stamp_duty_pct = sd.get(gender_key, sd.get("male_pct", 6))
            stamp_duty = value_for_duty * stamp_duty_pct / 100.0
            registration_pct = sd.get("registration_fee_pct", 1)
            registration_charges = value_for_duty * registration_pct / 100.0
            cap = sd.get("registration_fee_cap")
            if cap:
                registration_charges = min(registration_charges, cap)

        # transfer and mutation charges -- standard small flat/percent costs seen
        # across most Indian municipalities, not tied to a single state's rules.
        transfer_charges = value_for_duty * 0.0  # most states fold transfer duty into stamp duty already
        mutation_charges = min(25000.0, max(1000.0, value_for_duty * 0.001))

        result.stamp_duty = round(stamp_duty, 2)
        result.registration_charges = round(registration_charges, 2)
        result.transfer_charges = round(transfer_charges, 2)
        result.mutation_charges = round(mutation_charges, 2)
        result.one_time_total = round(stamp_duty + registration_charges + transfer_charges + mutation_charges, 2)

    # ---- capital gains ------------------------------------------------------
    def _compute_capital_gains(self, prop: PropertyInput, rules: dict, result: TaxBreakdown):
        cg = rules.get("capital_gains", {})

        # CGT only applies to an actual sale. Without a real purchase price this
        # used to assume one (market_value * 0.65), which invented a tax bill for
        # every plain valuation -- including someone who isn't selling at all.
        if not prop.purchase_price:
            result.capital_gains_tax_estimate = 0.0
            result.capital_gains_method = (
                "Not calculated -- enter an Original Purchase Price to estimate capital gains tax "
                "(this only applies if you are selling, not for a valuation/holding estimate)."
            )
            return

        purchase_price = prop.purchase_price
        sale_price = prop.market_value
        gain = max(0.0, sale_price - purchase_price)

        long_term_months = cg.get("long_term_holding_months", 24)
        is_long_term = prop.holding_period_months >= long_term_months

        if not is_long_term:
            # Short-term: taxed at slab rate; we show a representative 20% estimate
            estimate = gain * 0.20
            result.capital_gains_tax_estimate = round(estimate, 2)
            result.capital_gains_method = "Short-term (held < 24 months) - estimated at representative 20% slab rate; actual tax follows your income tax slab."
            return

        rate_no_indexation = cg.get("ltcg_rate_pct_no_indexation", 12.5)
        tax_no_indexation = gain * rate_no_indexation / 100.0

        # Indexation (20% route) is only available for property acquired BEFORE the
        # cutoff date (23 July 2024 per Finance Act 2024); anything bought after that
        # only gets the 12.5%-no-indexation route. We only have holding_period_months,
        # so approximate the purchase date from "today" minus that holding period --
        # consistent with the 2026 "current year" used elsewhere in this engine.
        cutoff = _parse_iso_date(cg.get("indexation_eligible_if_acquired_before"))
        approx_purchase_date = _CURRENT_DATE - timedelta(days=prop.holding_period_months * 30.44)
        indexation_eligible = cutoff is None or approx_purchase_date < cutoff

        if not indexation_eligible:
            result.capital_gains_tax_estimate = round(tax_no_indexation, 2)
            result.capital_gains_method = (
                "Long-term capital gains, 12.5% WITHOUT indexation -- the only route available for property "
                "acquired on/after 23 July 2024 (indexation was withdrawn by the Finance Act 2024)."
            )
            return

        # Simplified indexation estimate using CII ratio assumption (purchase ~10 yrs ago -> moderate uplift)
        cii_current = cg.get("cii_fy_2025_26", 376)
        rate_with_indexation = cg.get("ltcg_rate_pct_with_indexation", 20)
        # Approximate an indexed cost using a flat assumed historical CII based on holding period
        years_held = max(1, prop.holding_period_months / 12)
        assumed_cii_at_purchase = max(100, cii_current / (1.06 ** years_held))  # ~6% avg CII growth assumption
        indexed_cost = purchase_price * (cii_current / assumed_cii_at_purchase)
        indexed_gain = max(0.0, sale_price - indexed_cost)
        tax_with_indexation = indexed_gain * rate_with_indexation / 100.0

        if tax_with_indexation < tax_no_indexation:
            result.capital_gains_tax_estimate = round(tax_with_indexation, 2)
            result.capital_gains_method = (
                f"Long-term capital gains, 20% WITH indexation chosen (lower of the two options). "
                f"Indexed cost estimate: Rs {indexed_cost:,.0f}."
            )
        else:
            result.capital_gains_tax_estimate = round(tax_no_indexation, 2)
            result.capital_gains_method = (
                "Long-term capital gains, 12.5% WITHOUT indexation chosen (lower of the two options)."
            )

    # ---- rental income tax ---------------------------------------------------
    def _compute_rental_income_tax(self, prop: PropertyInput, rules: dict, result: TaxBreakdown):
        if prop.annual_rental_income <= 0:
            result.rental_income_tax_estimate = 0.0
            return
        rit = rules.get("rental_income_tax", {})
        std_deduction_pct = rit.get("standard_deduction_pct", 30)
        net_annual_value = prop.annual_rental_income - result.annual_property_tax
        net_annual_value = max(0.0, net_annual_value)
        taxable_income_from_property = net_annual_value * (1 - std_deduction_pct / 100.0)
        # representative estimate at a 20% marginal slab (actual depends on total income)
        result.rental_income_tax_estimate = round(max(0.0, taxable_income_from_property) * 0.20, 2)

    # ---- total purchase cost --------------------------------------------------
    def _compute_total_purchase_cost(self, prop: PropertyInput, result: TaxBreakdown):
        result.total_purchase_cost = round(prop.market_value + result.one_time_total, 2)

    # ---- future projections -----------------------------------------------
    def _compute_projections(self, prop: PropertyInput, result: TaxBreakdown):
        rate = prop.inflation_rate_pct / 100.0
        base_annual = result.recurring_annual_total
        projections = {}
        for years in (5, 10, 15):
            projected_annual = base_annual * ((1 + rate) ** years)
            cumulative = sum(base_annual * ((1 + rate) ** y) for y in range(1, years + 1))
            projections[f"{years}_year"] = {
                "projected_annual_tax": round(projected_annual, 2),
                "cumulative_tax_paid": round(cumulative, 2),
            }
        result.projections = projections


# ──────────────────────────────────────────────────────────────────────────
# Convenience function for simple call sites (e.g. Streamlit app)
# ──────────────────────────────────────────────────────────────────────────
_engine_singleton: Optional[TaxEngine] = None


def get_engine() -> TaxEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = TaxEngine()
    return _engine_singleton


def calculate_taxes(prop: PropertyInput) -> TaxBreakdown:
    return get_engine().calculate(prop)
