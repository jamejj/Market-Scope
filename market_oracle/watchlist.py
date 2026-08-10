from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WATCHLIST_PATH = DATA_DIR / "watchlist.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _date_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)[:10]


def _make_id(symbol: str, horizon: int, created_at: str, salt: str = "") -> str:
    raw = f"{symbol.upper()}|{int(horizon)}|{created_at}|{salt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def load_watchlist(path: Path = WATCHLIST_PATH) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, dict):
        items = payload.get("items", [])
    else:
        items = payload
    return [item for item in items if isinstance(item, dict)]


def save_watchlist(items: list[dict], path: Path = WATCHLIST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "items": items}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def find_active_duplicate(items: list[dict], symbol: str, horizon: int) -> dict | None:
    normalized = str(symbol or "").upper()
    for item in items:
        if str(item.get("status") or "ACTIVE").upper() != "ACTIVE":
            continue
        if str(item.get("symbol") or "").upper() != normalized:
            continue
        try:
            item_horizon = int(item.get("horizon"))
        except (TypeError, ValueError):
            continue
        if item_horizon == int(horizon):
            return item
    return None


def upsert_watch_item(item: dict, path: Path = WATCHLIST_PATH) -> tuple[dict, bool]:
    items = load_watchlist(path)
    symbol = str(item.get("symbol") or "").upper().strip()
    if not symbol:
        raise ValueError("Watchlist item requires symbol")
    horizon = int(item.get("horizon") or 0)
    if horizon <= 0:
        raise ValueError("Watchlist item requires positive horizon")
    duplicate = find_active_duplicate(items, symbol, horizon)
    if duplicate:
        return duplicate, False

    created_at = str(item.get("created_at") or _now_iso())
    normalized = {
        **item,
        "id": item.get("id") or _make_id(symbol, horizon, created_at, str(len(items))),
        "symbol": symbol,
        "horizon": horizon,
        "created_at": created_at,
        "status": str(item.get("status") or "ACTIVE").upper(),
    }
    items.append(normalized)
    save_watchlist(items, path)
    return normalized, True


def archive_watch_item(item_id: str, path: Path = WATCHLIST_PATH, archived_at: str | None = None) -> bool:
    items = load_watchlist(path)
    changed = False
    for item in items:
        if str(item.get("id")) == str(item_id):
            item["status"] = "ARCHIVED"
            item["archived_at"] = archived_at or _now_iso()
            changed = True
            break
    if changed:
        save_watchlist(items, path)
    return changed


def watchlist_summary(items: list[dict]) -> dict:
    active = [item for item in items if str(item.get("status") or "ACTIVE").upper() == "ACTIVE"]
    archived = [item for item in items if str(item.get("status") or "").upper() == "ARCHIVED"]
    symbols = {str(item.get("symbol") or "").upper() for item in active if item.get("symbol")}
    return {
        "total": len(items),
        "active": len(active),
        "archived": len(archived),
        "symbols": len(symbols),
    }


def watch_item_from_analysis(
    result: dict,
    report: dict,
    source_context: dict | None = None,
    *,
    source: str = "FULL_ANALYSIS",
    origin: str = "full_analysis",
) -> dict:
    source_context = source_context or {}
    symbol = str(result.get("symbol") or report.get("symbol") or "").upper()
    horizon = int(report.get("primary_horizon") or 20)
    forecasts = result.get("forecasts") or {}
    forecast = forecasts.get(horizon) or forecasts.get(str(horizon)) or {}
    verdict = report.get("verdict") or {}
    evidence = report.get("evidence") or []
    counterpoints = report.get("counterpoints") or []
    freshness = report.get("freshness") or {}
    return {
        "symbol": symbol,
        "horizon": horizon,
        "source": source,
        "origin": origin,
        "status": "ACTIVE",
        "verdict": verdict.get("reason") or "UNKNOWN",
        "verdict_label": verdict.get("label") or "—",
        "verdict_decision": verdict.get("decision"),
        "probability_up": _safe_float(forecast.get("probability_up")),
        "expected_return": _safe_float(forecast.get("expected_return")),
        "quality": str(forecast.get("quality") or "—"),
        "reason": str(evidence[0] if evidence else report.get("headline") or "—"),
        "thesis": str(report.get("headline") or "—"),
        "risk_note": str(counterpoints[0] if counterpoints else "Brak głównej kontrtezy w raporcie."),
        "data_as_of": _date_text(result.get("last_date") or freshness.get("analysis")),
        "radar_as_of": freshness.get("radar") or source_context.get("radar_updated_at") or "—",
        "benchmark": result.get("benchmark") or freshness.get("benchmark") or "—",
        "last_price": _safe_float(result.get("last_price")),
    }
