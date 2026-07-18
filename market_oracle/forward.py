from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from .data import download_history


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
CANDIDATE_MANIFEST_PATH = CONFIG_DIR / "marketscope_20d_long_candidate_v1.json"
UNSEEN_UNIVERSE_PATH = CONFIG_DIR / "unseen_usa_etf_v1.json"
FORWARD_LEDGER_PATH = DATA_DIR / "forward_ledger_candidate_v1.jsonl"

SIGNAL_EVENT = "SIGNAL_OBSERVED"
ENTRY_EVENT = "ENTRY_FILLED"
CLOSE_EVENT = "POSITION_CLOSED"
LONG_LABELS = {"KANDYDAT WZROSTOWY", "SILNY KANDYDAT WZROSTOWY"}
HASH_FIELDS_TO_SKIP = {"manifest_hash", "universe_hash", "event_id"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def canonical_json(payload: Any) -> str:
    return json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def frozen_hash(payload: dict[str, Any]) -> str:
    return sha256_payload({key: value for key, value in payload.items() if key not in HASH_FIELDS_TO_SKIP})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate_manifest(path: Path = CANDIDATE_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_unseen_universe(path: Path = UNSEEN_UNIVERSE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_hash(payload: dict[str, Any], field: str) -> bool:
    expected = payload.get(field)
    return isinstance(expected, str) and expected == frozen_hash(payload)


def current_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_forward_events(path: Path = FORWARD_LEDGER_PATH) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Nieprawidłowy JSONL w {path}:{line_number}") from exc
    return events


def append_forward_event(event: dict[str, Any], path: Path = FORWARD_LEDGER_PATH) -> dict[str, Any]:
    payload = json_safe(event)
    payload.setdefault("event_time_utc", utc_now_iso())
    payload.setdefault("code_commit", current_commit())
    payload.setdefault("event_id", sha256_payload({
        key: value for key, value in payload.items() if key != "event_id"
    })[:24])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")
    return payload


def _text(row: dict[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return default if value is None else str(value)


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _date_text(value: Any) -> str | None:
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).tz_localize(None).date().isoformat()


def _symbol_is_candidate_scope(symbol: str, asset_class: str, manifest: dict[str, Any]) -> bool:
    symbol = symbol.strip().upper()
    asset_class = asset_class.strip()
    accepted_classes = set(manifest.get("scope", {}).get("accepted_snapshot_asset_classes") or ["USA / ETF"])
    if symbol.endswith("-USD") or symbol.endswith(".WA") or symbol.startswith("^") or "." in symbol:
        return False
    if asset_class in accepted_classes:
        return True
    return asset_class.upper() in {"USA", "ETF", "USA/ETF", "USA / ETF"}


def is_candidate_row(row: dict[str, Any], manifest: dict[str, Any] | None = None) -> bool:
    manifest = manifest or load_candidate_manifest()
    contract = manifest.get("decision_contract", {})
    scope = manifest.get("scope", {})
    symbol = _text(row, "Symbol").strip().upper()
    if not symbol:
        return False
    try:
        horizon = int(row.get("Horyzont"))
    except (TypeError, ValueError):
        return False
    if horizon != int(scope.get("horizon_sessions", 20)):
        return False
    if _text(row, "Tryb analizy").upper() != str(contract.get("required_analysis_mode", "ML")).upper():
        return False
    label = _text(row, "Ocena").upper()
    accepted_labels = {str(value).upper() for value in contract.get("accepted_labels", LONG_LABELS)}
    if label not in accepted_labels:
        return False
    probability = _float(row, "P(wzrost)", 0.5)
    expected = _float(row, "Oczekiwany ruch", 0.0)
    quality = _text(row, "Jakość modelu")
    if contract.get("reject_low_quality", True) and quality.startswith("NISKA"):
        return False
    if probability < float(contract.get("threshold", 0.55)):
        return False
    if expected < float(contract.get("min_expected_return", 0.0)):
        return False
    return _symbol_is_candidate_scope(symbol, _text(row, "Klasa"), manifest)


def forward_signal_id(row: dict[str, Any], manifest: dict[str, Any], signal_date: str) -> str:
    payload = {
        "candidate_id": manifest["candidate_id"],
        "symbol": _text(row, "Symbol").strip().upper(),
        "signal_date": signal_date,
        "horizon": int(row.get("Horyzont")),
        "direction": manifest.get("scope", {}).get("direction", "LONG"),
    }
    return sha256_payload(payload)[:24]


def _observed_signal_event(
    row: dict[str, Any],
    snapshot: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    signal_date = _date_text(row.get("Data") or snapshot.get("updated_at"))
    if signal_date is None:
        return None
    signal_id = forward_signal_id(row, manifest, signal_date)
    return {
        "event_type": SIGNAL_EVENT,
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_hash": manifest.get("manifest_hash") or frozen_hash(manifest),
        "signal_id": signal_id,
        "status": "PENDING",
        "symbol": _text(row, "Symbol").strip().upper(),
        "asset_class": _text(row, "Klasa"),
        "horizon": int(row.get("Horyzont")),
        "direction": manifest.get("scope", {}).get("direction", "LONG"),
        "execution": manifest.get("scope", {}).get("execution", "NEXT_OPEN"),
        "signal_date": signal_date,
        "source_snapshot_updated_at": snapshot.get("updated_at"),
        "source_snapshot_schema_version": snapshot.get("schema_version"),
        "analysis_mode": _text(row, "Tryb analizy"),
        "decision_label": _text(row, "Ocena"),
        "decision_reason": row.get("DecisionReason") or "LONG_CONFIRMED",
        "signal_price": _float(row, "Cena"),
        "probability_up": _float(row, "P(wzrost)", 0.5),
        "expected_return": _float(row, "Oczekiwany ruch", 0.0),
        "quality": _text(row, "Jakość modelu"),
        "auc": _float(row, "AUC walidacji", 0.5),
        "brier": _float(row, "Brier", 0.25),
        "score": _float(row, "Score", 0.0),
        "setup": row.get("Setup"),
        "raw_snapshot_row": row,
    }


def record_snapshot_forward_signals(
    snapshot: dict[str, Any],
    *,
    path: Path = FORWARD_LEDGER_PATH,
    manifest_path: Path = CANDIDATE_MANIFEST_PATH,
) -> int:
    """Append new Candidate v1 signals from a completed monitor snapshot.

    Existing events are never edited. A signal is de-duplicated by candidate,
    symbol, signal date, horizon and direction so a repeated dashboard refresh
    does not create a second pending trade.
    """
    if snapshot.get("status") != "complete":
        return 0
    manifest = load_candidate_manifest(manifest_path)
    if manifest.get("manifest_hash") and not verify_frozen_hash(manifest, "manifest_hash"):
        raise ValueError(f"Candidate manifest hash mismatch: {manifest_path}")
    events = load_forward_events(path)
    observed_ids = {
        event.get("signal_id")
        for event in events
        if event.get("event_type") == SIGNAL_EVENT
    }
    added = 0
    for row in snapshot.get("records") or []:
        if not isinstance(row, dict) or not is_candidate_row(row, manifest):
            continue
        event = _observed_signal_event(row, snapshot, manifest)
        if event is None or event["signal_id"] in observed_ids:
            continue
        appended = append_forward_event(event, path)
        observed_ids.add(appended["signal_id"])
        added += 1
    return added


def reconstruct_forward_state(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(events):
        signal_id = event.get("signal_id")
        if not signal_id:
            continue
        current = state.setdefault(str(signal_id), {"events": []})
        current["events"].append(event.get("event_type"))
        current["last_event_index"] = index
        event_type = event.get("event_type")
        if event_type == SIGNAL_EVENT and "symbol" not in current:
            current.update(event)
        elif event_type == ENTRY_EVENT:
            current.update({
                "status": "OPEN",
                "entry_date": event.get("entry_date"),
                "entry_price": event.get("entry_price"),
                "entry_event_id": event.get("event_id"),
            })
        elif event_type == CLOSE_EVENT:
            current.update({
                "status": "CLOSED",
                "exit_date": event.get("exit_date"),
                "exit_price": event.get("exit_price"),
                "gross_return": event.get("gross_return"),
                "strategy_return": event.get("strategy_return"),
                "hit": event.get("hit"),
                "close_event_id": event.get("event_id"),
            })
        current.setdefault("status", "PENDING")
    return state


def _event_exists(events: list[dict[str, Any]], signal_id: str, event_type: str) -> bool:
    return any(event.get("signal_id") == signal_id and event.get("event_type") == event_type for event in events)


def _open_series(history: pd.DataFrame) -> pd.Series:
    column = "Open" if "Open" in history else "Close"
    series = history[column].astype(float).copy()
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    return series[~series.index.duplicated(keep="last")].dropna().sort_index()


def _next_open_execution(
    signal: dict[str, Any],
    history: pd.DataFrame,
) -> dict[str, Any]:
    opens = _open_series(history)
    if opens.empty:
        return {"ready_for_entry": False, "ready_for_exit": False}
    signal_date = pd.Timestamp(signal["signal_date"]).normalize()
    entry_position = int(opens.index.searchsorted(signal_date, side="right"))
    if entry_position >= len(opens):
        return {"ready_for_entry": False, "ready_for_exit": False}
    horizon = int(signal.get("horizon") or 20)
    entry_date = pd.Timestamp(opens.index[entry_position]).normalize()
    entry_price = float(opens.iloc[entry_position])
    exit_position = entry_position + horizon
    payload: dict[str, Any] = {
        "ready_for_entry": True,
        "entry_date": entry_date.date().isoformat(),
        "entry_price": entry_price,
        "ready_for_exit": False,
    }
    if exit_position < len(opens):
        exit_date = pd.Timestamp(opens.index[exit_position]).normalize()
        exit_price = float(opens.iloc[exit_position])
        payload.update({
            "ready_for_exit": True,
            "exit_date": exit_date.date().isoformat(),
            "exit_price": exit_price,
        })
    return payload


def _history_for(
    symbol: str,
    histories: dict[str, pd.DataFrame] | None,
    years: int,
) -> pd.DataFrame:
    if histories is not None and symbol in histories:
        return histories[symbol]
    return download_history(symbol, years)


def refresh_forward_ledger(
    *,
    path: Path = FORWARD_LEDGER_PATH,
    manifest_path: Path = CANDIDATE_MANIFEST_PATH,
    years: int = 3,
    histories: dict[str, pd.DataFrame] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    """Append entry/close events when next-open execution data is available."""
    manifest = load_candidate_manifest(manifest_path)
    events = load_forward_events(path)
    state = reconstruct_forward_state(events)
    errors: dict[str, str] = {}
    total_cost = float(manifest.get("execution_contract", {}).get("round_trip_cost", 0.0015))

    for signal_id in sorted(state):
        signal = state[signal_id]
        if signal.get("candidate_id") != manifest.get("candidate_id"):
            continue
        if signal.get("status") == "CLOSED":
            continue
        symbol = str(signal.get("symbol", "")).upper()
        if not symbol:
            continue
        try:
            execution = _next_open_execution(signal, _history_for(symbol, histories, years))
        except Exception as exc:
            errors[symbol] = str(exc)
            continue

        if signal.get("status") == "PENDING" and execution.get("ready_for_entry"):
            if not _event_exists(events, signal_id, ENTRY_EVENT):
                event = append_forward_event({
                    "event_type": ENTRY_EVENT,
                    "candidate_id": manifest["candidate_id"],
                    "candidate_manifest_hash": manifest.get("manifest_hash") or frozen_hash(manifest),
                    "signal_id": signal_id,
                    "status": "OPEN",
                    "symbol": symbol,
                    "direction": signal.get("direction", "LONG"),
                    "execution": "NEXT_OPEN",
                    "signal_date": signal.get("signal_date"),
                    "entry_date": execution["entry_date"],
                    "entry_price": execution["entry_price"],
                    "price_source": "Open",
                }, path)
                events.append(event)
                state = reconstruct_forward_state(events)
                signal = state[signal_id]

        if signal.get("status") == "OPEN" and execution.get("ready_for_exit"):
            if not _event_exists(events, signal_id, CLOSE_EVENT):
                entry_price = float(signal["entry_price"])
                exit_price = float(execution["exit_price"])
                gross_return = exit_price / entry_price - 1.0
                direction = str(signal.get("direction", "LONG")).upper()
                signed_gross = gross_return if direction == "LONG" else -gross_return
                strategy_return = signed_gross - total_cost
                event = append_forward_event({
                    "event_type": CLOSE_EVENT,
                    "candidate_id": manifest["candidate_id"],
                    "candidate_manifest_hash": manifest.get("manifest_hash") or frozen_hash(manifest),
                    "signal_id": signal_id,
                    "status": "CLOSED",
                    "symbol": symbol,
                    "direction": direction,
                    "execution": "NEXT_OPEN",
                    "signal_date": signal.get("signal_date"),
                    "entry_date": signal.get("entry_date"),
                    "entry_price": entry_price,
                    "exit_date": execution["exit_date"],
                    "exit_price": exit_price,
                    "gross_return": gross_return,
                    "strategy_return": strategy_return,
                    "round_trip_cost": total_cost,
                    "hit": strategy_return > 0,
                    "price_source": "Open",
                }, path)
                events.append(event)
                state = reconstruct_forward_state(events)

    return events, state, errors


def forward_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    state = reconstruct_forward_state(events)
    statuses = pd.Series([item.get("status", "PENDING") for item in state.values()], dtype=object)
    closed_returns = pd.Series(
        [item.get("strategy_return") for item in state.values() if item.get("status") == "CLOSED"],
        dtype=float,
    ).dropna()
    event_types = pd.Series([event.get("event_type") for event in events], dtype=object)
    return {
        "events": int(len(events)),
        "signals": int(len(state)),
        "pending": int((statuses == "PENDING").sum()) if len(statuses) else 0,
        "open": int((statuses == "OPEN").sum()) if len(statuses) else 0,
        "closed": int((statuses == "CLOSED").sum()) if len(statuses) else 0,
        "event_counts": event_types.value_counts().to_dict() if len(event_types) else {},
        "closed_mean_return": float(closed_returns.mean()) if len(closed_returns) else None,
        "closed_median_return": float(closed_returns.median()) if len(closed_returns) else None,
        "closed_hit_rate": float((closed_returns > 0).mean()) if len(closed_returns) else None,
        "candidate_ids": sorted({str(event.get("candidate_id")) for event in events if event.get("candidate_id")}),
    }
