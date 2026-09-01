from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import download_history


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JOURNAL_PATH = DATA_DIR / "signal_journal.json"
NON_DIRECTIONAL_REASONS = {
    "EXPECTED_RETURN_CONFLICT",
    "EXPECTED_RETURN_TOO_SMALL",
    "INCOMPLETE_FORECAST",
    "LOW_QUALITY",
    "PROBABILITY_INSIDE_BAND",
}


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


def valid_machine_decision_contract(row: dict) -> bool:
    decision = row.get("Decision")
    reason = row.get("DecisionReason")
    if type(decision) is not int or decision not in {-1, 0, 1} or not isinstance(reason, str):
        return False
    if decision == 1:
        return reason == "LONG_CONFIRMED"
    if decision == -1:
        return reason == "SHORT_CONFIRMED"
    return reason in NON_DIRECTIONAL_REASONS


def signal_direction(row: dict) -> str | None:
    if not valid_machine_decision_contract(row):
        return None
    if row["Decision"] == 1:
        return "LONG"
    if row["Decision"] == -1:
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
        if row.get("Tryb analizy") != "ML":
            continue
        direction = signal_direction(row)
        if direction is None:
            continue
        try:
            horizon = int(row.get("Horyzont", 0))
            signal_price = float(row.get("Cena"))
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
            "execution": "NEXT_OPEN",
            "signal_price": signal_price,
            "entry_date": None,
            "entry_price": None,
            "setup": row.get("Setup", "—"),
            "label": row.get("Ocena", "—"),
            "decision": row["Decision"],
            "decision_reason": row["DecisionReason"],
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
    open_price = history["Open"].astype(float) if "Open" in history else close
    signal_date = pd.Timestamp(entry["signal_date"])
    start_idx = int(close.index.searchsorted(signal_date, side="left"))
    if start_idx >= len(close):
        entry["bars_elapsed"] = 0
        entry["bars_remaining"] = int(entry["horizon"])
        return entry

    horizon = int(entry["horizon"])
    use_next_open = entry.get("execution") == "NEXT_OPEN" or entry.get("entry_price") is None
    entry_idx = start_idx + 1 if use_next_open else start_idx
    if entry_idx >= len(open_price):
        entry["status"] = "open"
        entry["bars_elapsed"] = 0
        entry["bars_remaining"] = horizon
        return entry

    if use_next_open:
        entry["entry_date"] = str(open_price.index[entry_idx].date())
        entry["entry_price"] = float(open_price.iloc[entry_idx])
        entry["execution"] = "NEXT_OPEN"

    target_idx = entry_idx + horizon if use_next_open else start_idx + horizon
    bars_elapsed = max(0, len(close) - 1 - entry_idx)
    entry["bars_elapsed"] = min(horizon, bars_elapsed)
    entry["bars_remaining"] = max(0, horizon - bars_elapsed)
    if target_idx >= len(open_price):
        entry["status"] = "open"
        return entry

    target_price = float(open_price.iloc[target_idx]) if use_next_open else float(close.iloc[target_idx])
    entry_price = float(entry["entry_price"])
    underlying_return = target_price / entry_price - 1
    strategy_return = underlying_return if entry["direction"] == "LONG" else -underlying_return
    entry.update({
        "status": "closed",
        "target_date": str(open_price.index[target_idx].date()) if use_next_open else str(close.index[target_idx].date()),
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


def paper_portfolio(
    entries: list[dict],
    starting_capital: float = 10_000.0,
    position_fraction: float = 0.10,
    round_trip_cost_bps: float = 20.0,
) -> tuple[pd.DataFrame, dict]:
    """Simulate a simple sequential paper portfolio from closed journal signals."""
    closed = [
        entry for entry in entries
        if entry.get("status") == "closed" and entry.get("strategy_return") is not None
    ]
    if not closed:
        return pd.DataFrame(), {
            "trades": 0,
            "final_capital": float(starting_capital),
            "total_return": 0.0,
            "max_drawdown": None,
            "profit_factor": None,
            "hit_rate": None,
            "average_trade": None,
            "expectancy_capital": None,
        }

    closed = sorted(
        closed,
        key=lambda entry: (
            str(entry.get("target_date") or entry.get("signal_date") or ""),
            str(entry.get("symbol") or ""),
            int(entry.get("horizon") or 0),
        ),
    )
    capital = float(starting_capital)
    peak = capital
    cost = float(round_trip_cost_bps) / 10_000
    fraction = max(0.0, min(float(position_fraction), 1.0))
    rows: list[dict] = []

    for index, entry in enumerate(closed, start=1):
        raw_return = float(entry.get("strategy_return") or 0.0)
        net_return = raw_return - cost
        capital_before = capital
        position_value = capital_before * fraction
        pnl = position_value * net_return
        capital = max(0.0, capital_before + pnl)
        peak = max(peak, capital)
        drawdown = capital / peak - 1 if peak else 0.0
        rows.append({
            "Nr": index,
            "Data sygnału": entry.get("signal_date"),
            "Data oceny": entry.get("target_date"),
            "Symbol": entry.get("symbol"),
            "Klasa": entry.get("asset_class", "—"),
            "Horyzont": entry.get("horizon"),
            "Kierunek": entry.get("direction"),
            "Zwrot brutto": raw_return,
            "Zwrot netto": net_return,
            "Pozycja": fraction,
            "Kapitał przed": capital_before,
            "P&L": pnl,
            "Kapitał": capital,
            "Drawdown": drawdown,
        })

    curve = pd.DataFrame(rows)
    pnl_series = curve["P&L"].astype(float)
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses.sum())) if not losses.empty else 0.0
    summary = {
        "trades": int(len(curve)),
        "final_capital": float(capital),
        "total_return": float(capital / starting_capital - 1) if starting_capital else 0.0,
        "max_drawdown": float(curve["Drawdown"].min()) if not curve.empty else None,
        "profit_factor": None if gross_loss == 0 else gross_profit / gross_loss,
        "hit_rate": float((pnl_series > 0).mean()) if not pnl_series.empty else None,
        "average_trade": float(curve["Zwrot netto"].mean()) if not curve.empty else None,
        "expectancy_capital": float(pnl_series.mean()) if not pnl_series.empty else None,
    }
    return curve, summary
