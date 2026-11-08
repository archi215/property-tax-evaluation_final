import os
import pytest
import config

pytestmark = pytest.mark.skipif(not os.path.exists(config.METADATA_JSON), reason="model not trained")
from fastapi.testclient import TestClient
import server

client = TestClient(server.app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "Land" in r.text


def test_meta_and_market():
    assert len(client.get("/api/meta").json()["cities"]) == 90
    assert len(client.get("/api/market").json()) > 100


def test_valuate_shape_and_monotonic():
    base = {"city": "Pune", "state": "Maharashtra", "property_type": "Apartment", "built_up_area_sqft": 1000}
    a = client.post("/api/valuate", json=base).json()
    b = client.post("/api/valuate", json={**base, "built_up_area_sqft": 2000}).json()
    assert b["pred"]["price"] > a["pred"]["price"]
    assert a["pred"]["lower_90"] < a["pred"]["price"] < a["pred"]["upper_90"]
    assert len(a["loan"]["yearly"]["year"]) == 20 and a["drivers"]["groups"]


def test_residuals_for_confidence_lab():
    r = client.get("/api/residuals").json()
    assert len(r["calib"]["abs_log"]) == len(r["test"]["ratio"]) == 1176


def test_negotiate_percentile_rises_with_asking():
    base = {"city": "Pune", "state": "Maharashtra", "property_type": "Apartment", "built_up_area_sqft": 1100}
    price = client.post("/api/valuate", json=base).json()["pred"]["price"]
    lo = client.post("/api/negotiate", json={**base, "asking": price * 0.9}).json()
    hi = client.post("/api/negotiate", json={**base, "asking": price * 1.2}).json()
    assert 0 <= lo["percentile"] < hi["percentile"] <= 100
    assert lo["quantiles"]["10"] < lo["quantiles"]["50"] < lo["quantiles"]["90"]


def test_money_endpoints():
    l = client.post("/api/loan", json={"price": 8e6, "down": 20, "rate": 8.5, "years": 20, "prepay": 5000, "lump": 0}).json()
    assert l["impact"]["months_saved"] > 0 and len(l["heat"]["z"]) == 6
    a = client.post("/api/afford", json={"income": 150000, "foir": 45, "down": 20, "rate": 8.5, "years": 20}).json()
    assert a["result"]["max_emi"] == 67500 and a["result"]["stressed_max_loan"] < a["result"]["max_loan"]
    r = client.post("/api/rentbuy", json={"price": 8e6, "down": 20, "rate": 8.5, "years": 20, "rent": 20000, "rg": 5, "ap": 6, "inv": 9, "horizon": 10}).json()
    assert len(r["series"]["year"]) == 10 and len(r["heat"]["z"]) == 6


def test_bulk_and_errors():
    ok = client.post("/api/bulk", json={"csv": "state,city,property_type,built_up_area_sqft,construction_year\nDelhi,Delhi,Apartment,1200,2015\n"}).json()
    assert ok["n"] == 1 and ok["rows"][0][ok["columns"].index("predicted_price")] > 0
    assert "error" in client.post("/api/bulk", json={"csv": "a,b\n1,2"}).json()


def test_pdf_and_history(tmp_path, monkeypatch):
    import storage
    monkeypatch.setattr(storage, "_conn", lambda path=str(tmp_path / "h.db"): __import__("sqlite3").connect(path) if False else _mk(path))
    pdf = client.post("/api/report", json={"city": "Pune", "state": "Maharashtra"})
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


def _mk(path):
    import sqlite3
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE IF NOT EXISTS valuations (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, label TEXT, state TEXT, city TEXT,
        property_type TEXT, area_sqft REAL, predicted_price REAL, lower_90 REAL, upper_90 REAL, annual_property_tax REAL, stamp_duty REAL,
        total_purchase_cost REAL, emi REAL, inputs_json TEXT)""")
    return con
