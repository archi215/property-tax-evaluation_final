"""
server.py -- custom web frontend + JSON API.   Run:  python -m uvicorn server:app --port 8000
Open http://localhost:8000
"""
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response

import service

app = FastAPI(title="Property valuation and tax")
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def _json(obj):
    return Response(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)), media_type="application/json")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


@app.get("/api/meta")
def meta():
    return _json(service.meta())


@app.post("/api/valuate")
async def valuate(request: Request):
    return _json(service.valuate(await request.json()))


@app.post("/api/negotiate")
async def negotiate(request: Request):
    return _json(service.negotiate(await request.json()))


@app.post("/api/loan")
async def loan(request: Request):
    return _json(service.loan_lab(await request.json()))


@app.post("/api/afford")
async def afford(request: Request):
    return _json(service.afford(await request.json()))


@app.post("/api/rentbuy")
async def rentbuy(request: Request):
    return _json(service.rent_buy(await request.json()))


@app.post("/api/bulk")
async def bulk(request: Request):
    return _json(service.bulk((await request.json()).get("csv", "")))


@app.post("/api/save")
async def save(request: Request):
    return _json({"id": service.save(await request.json())})


@app.get("/api/history")
def history():
    return _json(service.history())


@app.post("/api/history/delete")
async def history_delete(request: Request):
    service.delete_history((await request.json())["id"])
    return _json({"ok": True})


@app.post("/api/report")
async def report(request: Request):
    return Response(service.report_pdf(await request.json()), media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=valuation_report.pdf"})


@app.get("/api/market")
def market():
    return _json(service.market_rows())


@app.get("/api/residuals")
def residuals():
    return _json(service.residuals())
