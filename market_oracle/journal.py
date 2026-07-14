from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import download_history


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JOURNAL_PATH = DATA_DIR / "signal_journal.json"
BULLISH_LABELS = {"SILNY KANDYDAT WZROSTOWY", "KANDYDAT WZROSTOWY"}
BEARISH_LABELS = {"SILNE RYZYKO SPADKU", "RYZYKO SPADKU"}


def load_journal(path: Path = JOURNAL_PATH) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def save_journal(entries: list[dict], path: Path = JOURNAL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def signal_direction(row: dict) -> str | None:
    label = row.get("Ocena")
    if label in BULLISH_LABELS:
        return "LONG"
    if label in BEARISH_LABELS:
        return "SHORT"
    return None


def _signal_date(row: dict, snapshot: dict) -> str:
    value = row.get("Data") or snapshot.get("updated_at") or snapshot.get("started_at")
    return str(pd.Timestamp(value).date())


def _entry_id(row: dict, snapshot: dict, direction: str) -> str:
    return f"{_signal_date(row, snapshot)}|{row.get('Symbol')}|{int(row.get('Horyzont', 0))}|{direction}"


def record_snapshot_signals(snapshot: dict, path: Path = JOURNAL_PATH) -> int:
    """Persist directional signals from a completed market scan. Returns number of new entries."""
    if not snapshot or snapshot.get("status") != "complete":
        return 0

    entries = load_journal(path)
    known_ids = {entry.get("id") for entry in entries}
    created_at = snapshot.get("updated_at") or datetime.now(timezone.utc).isoformat()
    added = 0

    for row in snapshot.get("records", []):
        direction = signal_direction(row)
        if direction is None:
            continue
        try:
            horizon = int(row.get("Horyzont", 0))
            entry_price = float(row.get("Cena"))
        except (TypeError, ValueError):
            continue
        entry_id = _entry_id(row, snapshot, direction)
        if entry_id in known_ids:
            continue
        entries.append({
            "id": entry_id,
            "created_at": created_at,
            "signal_date": _signal_date(row, snapshot),
            "symbol": str(row.get("Symbol", "")).upper(),
            "asset_class": row.get("Klasa", "—"),
            "horizon": horizon,
            "direction": direction,
            "entry_price": entry_price,
            "setup": row.get("Setup", "—"),
            "label": row.get("Ocena", "—"),
            "probability_up": row.get("P(wzrost)"),
            "expected_return": row.get("Oczekiwany ruch"),
            "auc": row.get("AUC walidacji"),
            "brier": row.get("Brier"),
            "quality": row.get("Jakość modelu"),
            "score": row.get("Score"),
            "status": "open",
            "bars_elapsed": 0,
            "bars_remaining": horizon,
            "target_date": None,
            "target_price": None,
            "underlying_return": None,
            "strategy_return": None,
            "hit": None,
            "evaluated_at": None,
        })
        known_ids.add(entry_id)
        added += 1

    if added:
        save_journal(entries, path)
    return added


def _evaluate_entry(entry: dict, history: pd.DataFrame) -> dict:
    close = history["Close"].astype(float)
    signal_date = pd.Timestamp(entry["signal_date"])
    start_idx = int(close.index.searchsorted(signal_date, side="left"))
    if start_idx >= len(close):
        entry["bars_elapsed"] = 0
        entry["bars_remaining"] = int(entry["horizon"])
        return entry

    horizon = int(entry["horizon"])
    target_idx = start_idx + horizon
    bars_elapsed = max(0, len(close) - 1 - start_idx)
    entry["bars_elapsed"] = min(horizon, bars_elapsed)
    entry["bars_remaining"] = max(0, horizon - bars_elapsed)
    if target_idx >= len(close):
        entry["status"] = "open"
        return entry

    target_price = float(close.iloc[target_idx])
    entry_price = float(entry["entry_price"])
    underlying_return = target_price / entry_price - 1
    strategy_return = underlying_return if entry["direction"] == "LONG" else -underlying_return
    entry.update({
        "status": "closed",
        "target_date": str(close.index[target_idx].date()),
        "target_price": target_price,
        "underlying_return": float(underlying_return),
        "strategy_return": float(strategy_return),
        "hit": bool(strategy_return > 0),
        "bars_elapsed": horizon,
        "bars_remaining": 0,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    })
    return entry


def refresh_journal_results(path: Path = JOURNAL_PATH, years: int = 3) -> tuple[list[dict], dict[str, str]]:
    entries = load_journal(path)
    errors: dict[str, str] = {}
    histories: dict[str, pd.DataFrame] = {}

    for entry in entries:
        if entry.get("status") == "closed":
            continue
        symbol = entry.get("symbol")
        if not symbol:
            continue
        try:
            if symbol not in histories:
                histories[symbol] = download_history(symbol, years=years)
            _evaluate_entry(entry, histories[symbol])
        except Exception as exc:
            errors[str(symbol)] = str(exc)

    save_journal(entries, path)
    return entries, errors


def journal_summary(entries: list[dict]) -> dict:
    closed = [entry for entry in entries if entry.get("status") == "closed"]
    open_entries = [entry for entry in entries if entry.get("status") != "closed"]
    if not closed:
        return {
            "total": len(entries), "closed": 0, "open": len(open_entries),
            "hit_rate": None, "average_return": None, "median_return": None,
            "best_return": None, "worst_return": None, "profit_factor": None,
            "payoff_ratio": None, "expectancy": None, "max_drawdown": None,
        }
    returns = pd.Series([entry["strategy_return"] for entry in closed], dtype=float)
    hits = pd.Series([entry["hit"] for entry in closed], dtype=bool)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses.sum())) if not losses.empty else 0.0
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    average_win = float(wins.mean()) if not wins.empty else None
    average_loss = float(abs(losses.mean())) if not losses.empty else None
    return {
        "total": len(entries), "closed": len(closed), "open": len(open_entries),
        "hit_rate": float(hits.mean()), "average_return": float(returns.mean()),
        "median_return": float(returns.median()), "best_return": float(returns.max()),
        "worst_return": float(returns.min()),
        "profit_factor": None if gross_loss == 0 else gross_profit / gross_loss,
        "payoff_ratio": None if not average_loss else (average_win or 0.0) / average_loss,
        "expectancy": float(returns.mean()),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else None,
    }
