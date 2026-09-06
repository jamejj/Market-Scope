from __future__ import annotations

import copy
import fcntl
import json
import math
from contextlib import contextmanager
from datetime import date, datetime, timezone
from numbers import Real
from pathlib import Path

import pandas as pd

from .data import download_history
from .product_verdict import (
    MachineDecisionState,
    persisted_machine_decision_state,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JOURNAL_PATH = DATA_DIR / "signal_journal.json"
JOURNAL_LOCK_PATH = DATA_DIR / "signal_journal.lock"
JOURNAL_ERROR_CODES = frozenset({
    "JOURNAL_CORRUPT",
    "JOURNAL_READ_FAILED",
    "JOURNAL_WRITE_FAILED",
    "JOURNAL_LOCK_FAILED",
    "JOURNAL_INVALID_SIGNAL",
    "JOURNAL_UNEXPECTED",
})
REFRESH_CONFLICT_FIELDS = (
    "symbol", "signal_date", "horizon", "direction", "execution",
    "entry_date", "entry_price", "status", "bars_elapsed", "bars_remaining",
    "target_date", "target_price", "underlying_return", "strategy_return",
    "hit", "evaluated_at",
)
REFRESH_LIFECYCLE_FIELDS = (
    "entry_date", "entry_price", "execution", "status", "bars_elapsed",
    "bars_remaining", "target_date", "target_price", "underlying_return",
    "strategy_return", "hit", "evaluated_at",
)


class JournalIntegrityError(RuntimeError):
    """Controlled Journal storage or input-integrity failure."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def journal_error_code(exc: BaseException) -> str:
    if isinstance(exc, JournalIntegrityError) and exc.code in JOURNAL_ERROR_CODES:
        return exc.code
    return "JOURNAL_UNEXPECTED"


def _refresh_conflict_token(entry: dict) -> tuple:
    return tuple(entry.get(field) for field in REFRESH_CONFLICT_FIELDS)


def _reject_nonfinite_json(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _has_nonfinite_number(value) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, list):
        return any(_has_nonfinite_number(item) for item in value)
    if isinstance(value, dict):
        return any(_has_nonfinite_number(item) for item in value.values())
    return False


def load_journal(path: Path = JOURNAL_PATH) -> list[dict]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise JournalIntegrityError("JOURNAL_READ_FAILED") from exc
    try:
        data = json.loads(raw, parse_constant=_reject_nonfinite_json)
    except (json.JSONDecodeError, ValueError) as exc:
        raise JournalIntegrityError("JOURNAL_CORRUPT") from exc
    if (
        not isinstance(data, list)
        or any(not isinstance(entry, dict) for entry in data)
        or _has_nonfinite_number(data)
    ):
        raise JournalIntegrityError("JOURNAL_CORRUPT")
    return data


def safe_load_journal(path: Path = JOURNAL_PATH) -> tuple[list[dict] | None, str | None]:
    """Return UI-safe Journal state without turning failure into an empty Journal."""
    try:
        return load_journal(path), None
    except Exception as exc:
        return None, journal_error_code(exc)


def _journal_lock_path(path: Path) -> Path:
    if path.resolve() == JOURNAL_PATH.resolve():
        return JOURNAL_LOCK_PATH
    return path.with_suffix(path.suffix + ".lock")


@contextmanager
def _journal_lock(path: Path):
    lock_path = _journal_lock_path(path)
    lock_file = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("a+")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    except OSError as exc:
        if lock_file is not None:
            lock_file.close()
        raise JournalIntegrityError("JOURNAL_LOCK_FAILED") from exc
    try:
        yield
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def _save_journal_unlocked(entries: list[dict], path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            raise ValueError("Journal root must be a list of objects")
        serialized = json.dumps(entries, ensure_ascii=False, indent=2, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise JournalIntegrityError("JOURNAL_WRITE_FAILED") from exc


def save_journal(entries: list[dict], path: Path = JOURNAL_PATH) -> None:
    with _journal_lock(path):
        _save_journal_unlocked(entries, path)


def valid_machine_decision_contract(row: dict) -> bool:
    return persisted_machine_decision_state(row) is not MachineDecisionState.INVALID


def signal_direction(row: dict) -> str | None:
    state = persisted_machine_decision_state(row)
    if state is MachineDecisionState.LONG:
        return "LONG"
    if state is MachineDecisionState.SHORT:
        return "SHORT"
    return None


def _signal_date(row: dict) -> str:
    value = row.get("Data")
    if isinstance(value, str):
        text = value.strip()
        try:
            if len(text) == 10:
                return date.fromisoformat(text).isoformat()
            if len(text) > 10 and text[10] in {"T", " "}:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError as exc:
            raise JournalIntegrityError("JOURNAL_INVALID_SIGNAL") from exc
        raise JournalIntegrityError("JOURNAL_INVALID_SIGNAL")
    if isinstance(value, (pd.Timestamp, datetime, date)) and pd.isna(value):
        raise JournalIntegrityError("JOURNAL_INVALID_SIGNAL")
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise JournalIntegrityError("JOURNAL_INVALID_SIGNAL")


def _strict_finite_real(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise JournalIntegrityError("JOURNAL_INVALID_SIGNAL")
    number = float(value)
    if not math.isfinite(number):
        raise JournalIntegrityError("JOURNAL_INVALID_SIGNAL")
    return number


def _entry_id(signal_date: str, symbol: str, horizon: int, direction: str) -> str:
    return f"{signal_date}|{symbol}|{horizon}|{direction}"


def _directional_entry(row: dict, created_at: str) -> dict | None:
    if not isinstance(row, dict):
        raise JournalIntegrityError("JOURNAL_INVALID_SIGNAL")
    if row.get("Tryb analizy") != "ML":
        return None

    state = persisted_machine_decision_state(row)
    if state is MachineDecisionState.INVALID:
        raise JournalIntegrityError("JOURNAL_INVALID_SIGNAL")
    if state is MachineDecisionState.NEUTRAL:
        return None

    probability = _strict_finite_real(row.get("P(wzrost)"))
    expected_return = _strict_finite_real(row.get("Oczekiwany ruch"))
    horizon = row.get("Horyzont")
    price = row.get("Cena")
    symbol = row.get("Symbol")
    if not 0.0 <= probability <= 1.0:
        raise JournalIntegrityError("JOURNAL_INVALID_SIGNAL")
    if type(horizon) is not int or horizon <= 0:
        raise JournalIntegrityError("JOURNAL_INVALID_SIGNAL")
    if isinstance(price, bool) or not isinstance(price, Real) or not math.isfinite(float(price)):
        raise JournalIntegrityError("JOURNAL_INVALID_SIGNAL")
    if not isinstance(symbol, str) or not symbol.strip():
        raise JournalIntegrityError("JOURNAL_INVALID_SIGNAL")

    direction = state.value
    signal_date = _signal_date(row)
    normalized_symbol = symbol.strip().upper()
    return {
        "id": _entry_id(signal_date, normalized_symbol, horizon, direction),
        "created_at": created_at,
        "signal_date": signal_date,
        "symbol": normalized_symbol,
        "asset_class": row.get("Klasa", "—"),
        "horizon": horizon,
        "direction": direction,
        "execution": "NEXT_OPEN",
        "signal_price": float(price),
        "entry_date": None,
        "entry_price": None,
        "setup": row.get("Setup", "—"),
        "label": row.get("Ocena", "—"),
        "decision": row["Decision"],
        "decision_reason": row["DecisionReason"],
        "probability_up": probability,
        "expected_return": expected_return,
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
    }


def record_snapshot_signals(snapshot: dict, path: Path = JOURNAL_PATH) -> int:
    """Persist directional signals from a completed market scan. Returns number of new entries."""
    if not snapshot or snapshot.get("status") != "complete":
        return 0

    created_at = snapshot.get("updated_at") or datetime.now(timezone.utc).isoformat()
    candidates = [
        entry
        for row in (snapshot.get("records") or [])
        if (entry := _directional_entry(row, created_at)) is not None
    ]

    with _journal_lock(path):
        entries = load_journal(path)
        known_ids = {entry.get("id") for entry in entries}
        additions = []
        for entry in candidates:
            if entry["id"] in known_ids:
                continue
            additions.append(entry)
            known_ids.add(entry["id"])
        if additions:
            _save_journal_unlocked(entries + additions, path)
        return len(additions)


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
    with _journal_lock(path):
        entries = copy.deepcopy(load_journal(path))
    errors: dict[str, str] = {}
    histories: dict[str, pd.DataFrame] = {}
    evaluated_entries: list[dict] = []

    for entry in entries:
        if entry.get("status") == "closed":
            continue
        symbol = entry.get("symbol")
        if not symbol:
            continue
        candidate = copy.deepcopy(entry)
        try:
            if symbol not in histories:
                histories[symbol] = download_history(symbol, years=years)
            _evaluate_entry(candidate, histories[symbol])
        except Exception as exc:
            errors[str(symbol)] = str(exc)
        else:
            evaluated_entries.append(candidate)

    def entry_key(entry: dict):
        entry_id = entry.get("id")
        if isinstance(entry_id, str) and entry_id:
            return ("id", entry_id)
        return (
            "legacy",
            entry.get("signal_date"),
            entry.get("symbol"),
            entry.get("horizon"),
            entry.get("direction"),
        )

    baseline_by_key = {entry_key(entry): entry for entry in entries}
    evaluated_by_key = {entry_key(entry): entry for entry in evaluated_entries}
    with _journal_lock(path):
        latest_entries = load_journal(path)
        changed = False
        for latest in latest_entries:
            key = entry_key(latest)
            baseline = baseline_by_key.get(key)
            evaluated = evaluated_by_key.get(key)
            if (
                baseline is None
                or evaluated is None
                or latest.get("status") == "closed"
                or _refresh_conflict_token(latest) != _refresh_conflict_token(baseline)
            ):
                continue
            lifecycle = {
                field: evaluated[field]
                for field in REFRESH_LIFECYCLE_FIELDS
                if field in evaluated
            }
            if any(latest.get(field) != value for field, value in lifecycle.items()):
                latest.update(lifecycle)
                changed = True
        if changed:
            _save_journal_unlocked(latest_entries, path)
        return latest_entries, errors


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
