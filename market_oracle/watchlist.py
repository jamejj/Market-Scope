from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from market_oracle.signals import DEFAULT_SIGNAL_THRESHOLD, SignalInputs, signal_verdict


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


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def watchlist_analysis_matches_selection(
    saved: dict | None,
    selected: dict | None,
    current_years: int | None = None,
) -> bool:
    """Return True only when a cached watchlist analysis belongs to the selected item."""
    if not isinstance(saved, dict) or not isinstance(selected, dict):
        return False
    if current_years is not None and saved.get("years") != current_years:
        return False

    selected_id = str(selected.get("id") or "")
    saved_id = str(saved.get("item_id") or "")
    if selected_id and saved_id != selected_id:
        return False

    selected_symbol = str(selected.get("symbol") or "").upper()
    result = saved.get("result") if isinstance(saved.get("result"), dict) else {}
    saved_symbol = str(saved.get("symbol") or result.get("symbol") or "").upper()
    if selected_symbol and saved_symbol != selected_symbol:
        return False

    selected_horizon = _safe_int(selected.get("horizon"))
    saved_horizon = _safe_int(saved.get("horizon"))
    if selected_horizon is not None and saved_horizon is not None and selected_horizon != saved_horizon:
        return False
    return True


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


def _forecast_for_horizon(result: dict, horizon: int) -> dict:
    forecasts = result.get("forecasts") if isinstance(result, dict) else {}
    if not isinstance(forecasts, dict):
        return {}
    return forecasts.get(horizon) or forecasts.get(str(horizon)) or {}


def _direction_from_snapshot(snapshot: dict | None) -> int:
    if not isinstance(snapshot, dict):
        return 0
    decision = _safe_int(snapshot.get("verdict_decision"))
    if decision in (-1, 0, 1):
        return decision

    label = str(snapshot.get("verdict_label") or snapshot.get("label") or "").upper()
    reason = str(snapshot.get("verdict") or snapshot.get("reason") or "").upper()
    if label == "LONG" or reason == "LONG_CONFIRMED":
        return 1
    if label == "SHORT" or reason == "SHORT_CONFIRMED":
        return -1
    return 0


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _business_days_between(start: date, end: date) -> int:
    if end <= start:
        return 0
    days = 0
    current = start + timedelta(days=1)
    while current <= end:
        if current.weekday() < 5:
            days += 1
        current += timedelta(days=1)
    return days


def watch_item_lifecycle(item: dict, now: date | datetime | str | None = None) -> dict:
    """Return a lifecycle status for the saved observation horizon.

    This is deliberately separate from the model verdict: an observation can reach
    its planned horizon even if the current model still confirms the thesis.
    """
    created = _parse_date(item.get("created_at"))
    current = _parse_date(now) if now is not None else datetime.now(timezone.utc).astimezone().date()
    horizon = _safe_int(item.get("horizon"))
    if created is None or current is None or horizon is None or horizon <= 0:
        return {
            "status": "UNKNOWN",
            "label": "Cykl obserwacji nieznany",
            "elapsed": None,
            "remaining": None,
            "is_expired": False,
        }

    elapsed = _business_days_between(created, current)
    remaining = max(horizon - elapsed, 0)
    expired = elapsed >= horizon
    return {
        "status": "HORIZON_ENDED" if expired else "ACTIVE",
        "label": "Horyzont obserwacji zakończony" if expired else "Horyzont obserwacji nadal trwa",
        "elapsed": elapsed,
        "remaining": remaining,
        "is_expired": expired,
    }


def watch_item_current_snapshot(result: dict, item: dict) -> dict:
    """Build the current comparable snapshot for the same symbol and horizon.

    The current verdict uses the shared MarketScope signal gate and never changes
    the immutable "then" snapshot saved in the watchlist item.
    """
    symbol = str(item.get("symbol") or "").upper()
    result_symbol = str(result.get("symbol") or "").upper() if isinstance(result, dict) else ""
    horizon = _safe_int(item.get("horizon"))
    if horizon is None or horizon <= 0:
        return {"available": False, "reason": "INVALID_HORIZON", "symbol": result_symbol or symbol, "horizon": horizon}
    if symbol and result_symbol and symbol != result_symbol:
        return {"available": False, "reason": "SYMBOL_MISMATCH", "symbol": result_symbol, "horizon": horizon}

    forecast = _forecast_for_horizon(result, horizon)
    if not forecast:
        return {"available": False, "reason": "HORIZON_NOT_AVAILABLE", "symbol": result_symbol or symbol, "horizon": horizon}

    probability = _safe_float(forecast.get("probability_up"))
    expected_return = _safe_float(forecast.get("expected_return"))
    auc = _safe_float(forecast.get("auc"))
    brier = _safe_float(forecast.get("brier"))
    quality = str(forecast.get("quality") or "NISKA — BRAK PRZEWAGI")
    verdict = signal_verdict(
        SignalInputs(
            probability=0.5 if probability is None else probability,
            expected_return=0.0 if expected_return is None else expected_return,
            quality=quality,
            auc=auc,
            brier=brier,
            source="ML",
        ),
        threshold=DEFAULT_SIGNAL_THRESHOLD,
    )
    return {
        "available": True,
        "symbol": result_symbol or symbol,
        "horizon": horizon,
        "calculated_at": _now_iso(),
        "data_as_of": _date_text(result.get("last_date")),
        "verdict": verdict.reason,
        "verdict_label": verdict.label,
        "verdict_decision": verdict.decision,
        "probability_up": probability,
        "expected_return": expected_return,
        "quality": quality,
        "auc": auc,
        "brier": brier,
        "last_price": _safe_float(result.get("last_price")),
    }


def compare_watch_item_to_current(item: dict, current: dict | None, now: date | datetime | str | None = None) -> dict:
    """Compare immutable watchlist snapshot with a current same-horizon analysis."""
    lifecycle = watch_item_lifecycle(item, now=now)
    if not isinstance(current, dict) or not current.get("available"):
        return {
            "comparison_status": "NO_CURRENT_ANALYSIS",
            "label": "Uruchom aktualną analizę",
            "verdict_transition": None,
            "lifecycle": lifecycle,
            "delta_probability": None,
            "delta_expected_return": None,
            "quality_change": None,
            "reasons": ["Najpierw uruchom pełną analizę tej obserwacji, żeby porównać ją z zapisem z dnia dodania."],
        }

    item_symbol = str(item.get("symbol") or "").upper()
    current_symbol = str(current.get("symbol") or "").upper()
    item_horizon = _safe_int(item.get("horizon"))
    current_horizon = _safe_int(current.get("horizon"))
    if item_symbol and current_symbol and item_symbol != current_symbol:
        return {
            "comparison_status": "MISMATCH",
            "label": "Analiza dotyczy innego symbolu",
            "verdict_transition": None,
            "lifecycle": lifecycle,
            "delta_probability": None,
            "delta_expected_return": None,
            "quality_change": None,
            "reasons": [f"Wybrano {item_symbol}, ale aktualna analiza dotyczy {current_symbol}."],
        }
    if item_horizon is not None and current_horizon is not None and item_horizon != current_horizon:
        return {
            "comparison_status": "MISMATCH",
            "label": "Analiza dotyczy innego horyzontu",
            "verdict_transition": None,
            "lifecycle": lifecycle,
            "delta_probability": None,
            "delta_expected_return": None,
            "quality_change": None,
            "reasons": [f"Obserwacja ma horyzont {item_horizon}, a aktualna analiza {current_horizon}."],
        }

    then_direction = _direction_from_snapshot(item)
    now_direction = _direction_from_snapshot(current)
    if then_direction == 0 and now_direction == 0:
        status, label = "UNCHANGED", "Bez zmiany statusu"
    elif then_direction == 0 and now_direction != 0:
        status, label = "GAINED_CONFIRMATION", "Teza zyskała potwierdzenie"
    elif then_direction != 0 and now_direction == then_direction:
        status, label = "STILL_CONFIRMED", "Teza nadal potwierdzona"
    elif then_direction != 0 and now_direction == 0:
        status, label = "WEAKENED", "Teza osłabła"
    else:
        status, label = "REVERSED", "Kierunek zanegowany"

    then_prob = _safe_float(item.get("probability_up"))
    now_prob = _safe_float(current.get("probability_up"))
    then_return = _safe_float(item.get("expected_return"))
    now_return = _safe_float(current.get("expected_return"))
    delta_probability = None if then_prob is None or now_prob is None else now_prob - then_prob
    delta_expected_return = None if then_return is None or now_return is None else now_return - then_return
    then_quality = str(item.get("quality") or "—")
    now_quality = str(current.get("quality") or "—")
    quality_change = "bez zmiany" if then_quality == now_quality else f"{then_quality} → {now_quality}"

    reasons = [
        f"Status bramki: {item.get('verdict_label') or '—'} → {current.get('verdict_label') or '—'} na tym samym horyzoncie {item_horizon or current_horizon}.",
    ]
    if delta_probability is not None:
        reasons.append(f"P(wzrost) zmieniło się o {delta_probability * 100:+.1f} pp.")
    if delta_expected_return is not None:
        reasons.append(f"Oczekiwany ruch zmienił się o {delta_expected_return * 100:+.1f} pp.")
    if quality_change != "bez zmiany":
        reasons.append(f"Jakość modelu: {quality_change}.")
    if lifecycle["is_expired"]:
        reasons.append("Horyzont obserwacji minął — to status cyklu życia, nie osobny verdict modelu.")

    return {
        "comparison_status": status,
        "label": label,
        "verdict_transition": f"{item.get('verdict_label') or '—'} → {current.get('verdict_label') or '—'}",
        "lifecycle": lifecycle,
        "delta_probability": delta_probability,
        "delta_expected_return": delta_expected_return,
        "quality_change": quality_change,
        "reasons": reasons,
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
