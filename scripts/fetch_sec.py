"""
Fetch Prologis financial metrics from the SEC EDGAR XBRL Company Facts API.

Prologis CIK: 0001045609

Metrics retrieved:
  - Revenue (Revenues)
  - Net Income (NetIncomeLoss)
  - Operating Expenses (OperatingExpenses)
  - Total Assets (Assets)
  - Total Liabilities (Liabilities)

Output: data/sec/prologis_financials.json

Run: python scripts/fetch_sec.py
NOTE: Edit the User-Agent email below before running.
"""
import json
from datetime import datetime
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
CIK = "0001045609"
COMPANY = "Prologis, Inc."
# SEC requires a real User-Agent with contact info
USER_AGENT = "FinancialAssistant your_email@example.com"

CONCEPTS = {
    "revenue":             ("us-gaap", "Revenues"),
    "net_income":          ("us-gaap", "NetIncomeLoss"),
    "operating_expenses":  ("us-gaap", "OperatingExpenses"),
    "total_assets":        ("us-gaap", "Assets"),
    "total_liabilities":   ("us-gaap", "Liabilities"),
}

OUT_PATH = Path(__file__).parent.parent / "data" / "sec" / "prologis_financials.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

BASE_URL = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"

# ── Fetch all company facts ───────────────────────────────────────────────────
print(f"Fetching SEC company facts for {COMPANY} (CIK {CIK})...")
resp = requests.get(BASE_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
resp.raise_for_status()
facts = resp.json()

# ── Extract metrics ───────────────────────────────────────────────────────────
result = {
    "company": COMPANY,
    "cik": CIK,
    "ticker": "PLD",
    "fetched_at": datetime.utcnow().isoformat() + "Z",
    "metrics": {},
}

for key, (taxonomy, concept) in CONCEPTS.items():
    print(f"  Processing: {key} ({taxonomy}/{concept})")
    try:
        units = facts["facts"][taxonomy][concept]["units"]["USD"]
    except KeyError:
        print(f"    WARNING: {concept} not found — skipping")
        result["metrics"][key] = {}
        continue

    # Filter 10-K (annual) filings
    annual = [
        u for u in units
        if u.get("form") in ("10-K",) and u.get("fp") == "FY"
    ]
    annual.sort(key=lambda u: u.get("end", ""), reverse=True)

    # Filter 10-Q (quarterly) filings
    quarterly = [
        u for u in units
        if u.get("form") in ("10-Q",) and u.get("fp") != "FY"
    ]
    quarterly.sort(key=lambda u: u.get("end", ""), reverse=True)

    result["metrics"][key] = {
        "label": facts["facts"][taxonomy][concept].get("label", concept),
        "latest_annual":    annual[0] if annual else None,
        "latest_quarterly": quarterly[0] if quarterly else None,
        "annual_history":   annual[:5],
        "quarterly_history": quarterly[:8],
    }

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_PATH.write_text(json.dumps(result, indent=2))
print(f"\nSaved to {OUT_PATH}")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\nLatest annual figures:")
for key, data in result["metrics"].items():
    entry = data.get("latest_annual")
    if entry:
        val = entry.get("val", 0)
        print(f"  {key:<25} ${val:>20,.0f}  ({entry.get('end')})")
