import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL as _SAURL
from dotenv import load_dotenv, dotenv_values

ROOT = Path(__file__).parent.parent
_ENV = dotenv_values(ROOT / ".env")
load_dotenv(ROOT / ".env", override=True)

SEC_PATH   = ROOT / "data" / "sec" / "prologis_financials.json"
PRESS_PATH = ROOT / "data" / "press_releases.json"


def _get_engine():
    url = _SAURL.create(
        "postgresql+psycopg2",
        username=_ENV.get("POSTGRES_USER") or os.getenv("POSTGRES_USER", "postgres"),
        password=_ENV.get("POSTGRES_PASSWORD") or os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=_ENV.get("POSTGRES_HOST") or os.getenv("POSTGRES_HOST", "localhost"),
        port=int(_ENV.get("POSTGRES_PORT") or os.getenv("POSTGRES_PORT", "5432")),
        database=_ENV.get("POSTGRES_DB") or os.getenv("POSTGRES_DB", "postgres"),
    )
    return create_engine(url)


def query_postgres(
    metro_area: Optional[str] = None,
    property_type: Optional[str] = None,
    min_revenue: Optional[float] = None,
    limit: int = 20,
) -> dict:
    """Query properties + financials from Postgres. Returns up to `limit` rows."""
    engine = _get_engine()
    sql = """
        SELECT p.property_id, p.address, p.metro_area, p.sq_footage,
               p.property_type, f.revenue, f.net_income, f.expenses
        FROM properties p
        LEFT JOIN financials f ON p.property_id = f.property_id
        WHERE 1=1
    """
    params: dict = {}
    if metro_area:
        sql += " AND LOWER(p.metro_area) = LOWER(:metro)"
        params["metro"] = metro_area
    if property_type:
        sql += " AND LOWER(p.property_type) = LOWER(:ptype)"
        params["ptype"] = property_type
    if min_revenue is not None:
        sql += " AND f.revenue >= :minrev"
        params["minrev"] = min_revenue
    sql += " ORDER BY f.revenue DESC NULLS LAST LIMIT :lim"
    params["lim"] = limit

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if df.empty:
        return {"count": 0, "properties": [], "summary": {}}

    for col in ["revenue", "net_income", "expenses"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return {
        "count": len(df),
        "properties": df.to_dict("records"),
        "summary": {
            "total_revenue":    float(df["revenue"].sum()),
            "total_net_income": float(df["net_income"].sum()),
            "avg_revenue":      float(df["revenue"].mean()),
            "metros":           sorted(df["metro_area"].unique().tolist()),
            "types":            sorted(df["property_type"].unique().tolist()),
        },
    }


def query_sec_edgar(metric: Optional[str] = None, period: str = "annual") -> dict:
    """Look up Prologis financial metrics from cached SEC EDGAR data."""
    if not SEC_PATH.exists():
        return {"error": "SEC data not found. Run scripts/fetch_sec.py first."}

    data = json.loads(SEC_PATH.read_text())
    metrics = data.get("metrics", {})

    if metric and metric not in metrics:
        return {"error": f"Unknown metric '{metric}'.", "available_metrics": list(metrics.keys())}

    target_metrics = [metric] if metric else list(metrics.keys())
    out = {"company": data.get("company", "Prologis"), "results": []}

    for m in target_metrics:
        entry = metrics.get(m, {}).get("latest_annual" if period == "annual" else "latest_quarterly")
        if entry:
            out["results"].append({
                "metric":        m,
                "value_usd":     entry.get("val"),
                "period_end":    entry.get("end"),
                "form":          entry.get("form"),
                "fiscal_year":   entry.get("fy"),
                "fiscal_period": entry.get("fp"),
            })
    return out


def query_press_releases(
    keywords: Optional[list] = None,
    category: Optional[str] = None,
    limit: int = 5,
) -> dict:
    """Search press releases by keyword (title + content) and/or category."""
    limit = int(limit)
    if not PRESS_PATH.exists():
        return {"error": "Press releases file not found."}

    releases = json.loads(PRESS_PATH.read_text())

    def match(r: dict) -> bool:
        if category and r.get("category", "").lower() != category.lower():
            return False
        if keywords:
            blob = (r.get("title", "") + " " + r.get("content", "")).lower()
            if not all(kw.lower() in blob for kw in keywords):
                return False
        return True

    matched = sorted([r for r in releases if match(r)], key=lambda r: r.get("date", ""), reverse=True)
    return {"count": len(matched), "releases": matched[:limit]}
