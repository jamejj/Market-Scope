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


def _easter_sunday(year: int) -> date:
    """Return Gregorian Easter Sunday for exchange-holiday calculations."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nyse_holidays(year: int) -> set[date]:
    try:
        from market_oracle.auto_forward import nyse_full_holidays

        holidays: set[date] = set()
        for item in nyse_full_holidays(year):
            try:
                holidays.add(date.fromisoformat(str(item)[:10]))
            except ValueError:
                continue
        return holidays
    except Exception:
        return set()


def _gpw_holidays(year: int) -> set[date]:
    easter = _easter_sunday(year)
    return {
        date(year, 1, 1),
        date(year, 1, 6),
        easter + timedelta(days=1),
        date(year, 5, 1),
        date(year, 5, 3),
        easter + timedelta(days=60),
        date(year, 8, 15),
        date(year, 11, 1),
        date(year, 11, 11),
        date(year, 12, 25),
        date(year, 12, 26),
    }


def _infer_asset_class(symbol: str, result: dict | None = None, source_context: dict | None = None) -> str:
    for payload in (source_context, result):
        if not isinstance(payload, dict):
            continue
        for key in ("asset_class", "assetClass", "class", "klasa", "market"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value

    normalized = str(symbol or "").upper()
    if normalized.endswith("-USD"):
        return "Krypto"
    if normalized.endswith(".WA"):
        return "GPW"
    return "USA / ETF"


def _infer_calendar_kind(symbol: str, asset_class: str | None = None, benchmark: str | None = None) -> str:
    text = " ".join(str(value or "").upper() for value in (symbol, asset_class, benchmark))
    normalized = str(symbol or "").upper()
    if normalized.endswith("-USD") or "KRYPTO" in text or "CRYPTO" in text:
        return "CRYPTO_24_7"
    if normalized.endswith(".WA") or "GPW" in text or "WIG" in text or "ETFBW" in text:
        return "GPW"
    if normalized and "." not in normalized and "-" not in normalized:
        return "NYSE"
    if "USA" in text or "ETF" in text or "^GSPC" in text:
        return "NYSE"
    return "UNKNOWN"


def _calendar_label(calendar_kind: str) -> str:
    labels = {
        "CRYPTO_24_7": "kalendarz crypto 24/7",
        "NYSE": "sesje USA/NYSE",
        "GPW": "sesje GPW",
        "UNKNOWN": "kalendarz nieznany",
    }
    return labels.get(str(calendar_kind or "UNKNOWN").upper(), "kalendarz nieznany")


def _calendar_unit(calendar_kind: str) -> str:
    kind = str(calendar_kind or "UNKNOWN").upper()
    if kind == "CRYPTO_24_7":
        return "dni"
    if kind in {"NYSE", "GPW"}:
        return "sesji"
    return "okresów"


def _is_counted_period(day: date, calendar_kind: str) -> bool | None:
    kind = str(calendar_kind or "UNKNOWN").upper()
    if kind == "CRYPTO_24_7":
        return True
    if kind == "NYSE":
        return day.weekday() < 5 and day not in _nyse_holidays(day.year)
    if kind == "GPW":
        return day.weekday() < 5 and day not in _gpw_holidays(day.year)
    return None


def _calendar_periods_between(start: date, end: date, calendar_kind: str) -> int | None:
    if end <= start:
        return 0
    periods = 0
    current = start + timedelta(days=1)
    while current <= end:
        counted = _is_counted_period(current, calendar_kind)
        if counted is None:
            return None
        if counted:
            periods += 1
        current += timedelta(days=1)
    return periods


def watch_item_lifecycle(item: dict, now: date | datetime | str | None = None) -> dict:
    """Return a lifecycle status for the saved observation horizon.

    This is deliberately separate from the model verdict: an observation can reach
    its planned horizon even if the current model still confirms the thesis.
    """
    data_anchor = _parse_date(item.get("data_as_of"))
    created = _parse_date(item.get("created_at"))
    anchor = data_anchor or created
    anchor_source = "data_as_of" if data_anchor is not None else "created_at"
    current = _parse_date(now) if now is not None else datetime.now(timezone.utc).astimezone().date()
    horizon = _safe_int(item.get("horizon"))
    symbol = str(item.get("symbol") or "").upper()
    asset_class = str(item.get("asset_class") or "")
    benchmark = str(item.get("benchmark") or "")
    calendar_kind = str(
        item.get("calendar_kind") or _infer_calendar_kind(symbol, asset_class, benchmark)
    ).upper()
    base = {
        "calendar_kind": calendar_kind,
        "calendar_label": _calendar_label(calendar_kind),
        "unit": _calendar_unit(calendar_kind),
        "anchor_date": anchor.isoformat() if anchor is not None else None,
        "anchor_source": anchor_source,
        "is_approximate": False,
    }
    if anchor is None or current is None or horizon is None or horizon <= 0:
        return {
            **base,
            "status": "UNKNOWN",
            "label": "Cykl obserwacji nieznany",
            "elapsed": None,
            "remaining": None,
            "is_expired": False,
            "is_approximate": True,
        }

    elapsed = _calendar_periods_between(anchor, current, calendar_kind)
    if elapsed is None:
        return {
            **base,
            "status": "UNKNOWN",
            "label": "Cykl obserwacji nieznany",
            "elapsed": None,
            "remaining": None,
            "is_expired": False,
            "is_approximate": True,
        }

    remaining = max(horizon - elapsed, 0)
    expired = elapsed >= horizon
    return {
        **base,
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
    asset_class = _infer_asset_class(symbol, result, source_context)
    benchmark = result.get("benchmark") or freshness.get("benchmark") or "—"
    calendar_kind = _infer_calendar_kind(symbol, asset_class, benchmark)
    return {
        "symbol": symbol,
        "horizon": horizon,
        "asset_class": asset_class,
        "calendar_kind": calendar_kind,
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
        "benchmark": benchmark,
        "last_price": _safe_float(result.get("last_price")),
    }
