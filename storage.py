"""storage.py -- persistent valuation history in SQLite (replaces session-only history)."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

import pandas as pd

import config

_COLS = ["id", "timestamp", "label", "state", "city", "property_type", "area_sqft", "predicted_price",
         "lower_90", "upper_90", "annual_property_tax", "stamp_duty", "total_purchase_cost", "emi", "inputs_json"]


def _conn(path: str = config.HISTORY_DB):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE IF NOT EXISTS valuations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, label TEXT, state TEXT, city TEXT,
        property_type TEXT, area_sqft REAL, predicted_price REAL, lower_90 REAL, upper_90 REAL,
        annual_property_tax REAL, stamp_duty REAL, total_purchase_cost REAL, emi REAL, inputs_json TEXT)""")
    return con


def save_valuation(record: dict, inputs: dict | None = None, label: str = "", path: str = config.HISTORY_DB) -> int:
    con = _conn(path)
    with con:
        cur = con.execute(
            "INSERT INTO valuations (timestamp,label,state,city,property_type,area_sqft,predicted_price,lower_90,upper_90,"
            "annual_property_tax,stamp_duty,total_purchase_cost,emi,inputs_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), label, record.get("state"), record.get("city"),
             record.get("property_type"), record.get("area_sqft"), record.get("predicted_price"),
             record.get("lower_90"), record.get("upper_90"), record.get("annual_property_tax"),
             record.get("stamp_duty"), record.get("total_purchase_cost"), record.get("emi"),
             json.dumps(inputs or {}, default=str)))
    rid = cur.lastrowid
    con.close()
    return rid


def load_history(path: str = config.HISTORY_DB) -> pd.DataFrame:
    con = _conn(path)
    df = pd.read_sql_query("SELECT * FROM valuations ORDER BY id DESC", con)
    con.close()
    return df


def delete_valuation(rid: int, path: str = config.HISTORY_DB) -> None:
    con = _conn(path)
    with con:
        con.execute("DELETE FROM valuations WHERE id=?", (rid,))
    con.close()


def clear_history(path: str = config.HISTORY_DB) -> None:
    con = _conn(path)
    with con:
        con.execute("DELETE FROM valuations")
    con.close()
