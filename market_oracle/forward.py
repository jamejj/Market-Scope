from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import download_history
from .signals import SignalInputs, signal_verdict


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
CANDIDATE_MANIFEST_PATH = CONFIG_DIR / "marketscope_20d_long_candidate_v1.json"
UNSEEN_UNIVERSE_PATH = CONFIG_DIR / "unseen_usa_etf_v1.json"
FORWARD_LEDGER_PATH = DATA_DIR / "forward_ledger_candidate_v1.jsonl"

SIGNAL_EVENT = "SIGNAL_OBSERVED"
SNAPSHOT_AUDIT_EVENT = "SNAPSHOT_AUDIT"
ACCEPT_EVENT = "POSITION_ACCEPTED"
SKIP_EVENT = "POSITION_SKIPPED"
ENTRY_EVENT = "ENTRY_FILLED"
CLOSE_EVENT = "POSITION_CLOSED"
LONG_LABELS = {"KANDYDAT WZROSTOWY", "SILNY KANDYDAT WZROSTOWY"}
HASH_FIELDS_TO_SKIP = {"manifest_hash", "universe_hash", "event_id", "event_hash"}
PIPELINE_FILES = (
    "market_oracle/model.py",
    "market_oracle/features.py",
    "market_oracle/signals.py",
    "market_oracle/cutoff.py",
    "market_oracle/engine.py",
)
CONTRACT_FILES = PIPELINE_FILES + (
    "configs/marketscope_20d_long_candidate_v1.json",
    "configs/unseen_usa_etf_v1.json",
)
WARSAW = ZoneInfo("Europe/Warsaw")
UTC = ZoneInfo("UTC")
GENESIS_HASH = "GENESIS"


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


def pipeline_fingerprint(files: tuple[str, ...] = PIPELINE_FILES) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    for relative in files:
        path = ROOT / relative
        file_hash = file_sha256(path)
        file_hashes[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(file_hash.encode("utf-8"))
    return {
        "files": file_hashes,
        "pipeline_hash": digest.hexdigest(),
        "git_commit": current_commit(),
    }


def git_dirty_paths(paths: tuple[str, ...] = CONTRACT_FILES) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ["git_status_unavailable"]
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def load_candidate_manifest(path: Path = CANDIDATE_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_unseen_universe(path: Path = UNSEEN_UNIVERSE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_hash(payload: dict[str, Any], field: str) -> bool:
    expected = payload.get(field)
    return isinstance(expected, str) and expected == frozen_hash(payload)


def verify_pipeline_contract(manifest: dict[str, Any]) -> bool:
    frozen = manifest.get("frozen_pipeline") or {}
    expected = frozen.get("pipeline_hash")
    return isinstance(expected, str) and pipeline_fingerprint().get("pipeline_hash") == expected


def assert_forward_contract_ready(
    manifest: dict[str, Any],
    *,
    require_clean_tree: bool = True,
    enforce_pipeline: bool = True,
) -> None:
    if manifest.get("manifest_hash") and not verify_frozen_hash(manifest, "manifest_hash"):
        raise ValueError("Candidate manifest hash mismatch.")
    if enforce_pipeline and not verify_pipeline_contract(manifest):
        current = pipeline_fingerprint().get("pipeline_hash")
        expected = (manifest.get("frozen_pipeline") or {}).get("pipeline_hash")
        raise ValueError(f"Candidate pipeline hash mismatch: current={current} expected={expected}")
    if require_clean_tree:
        dirty = git_dirty_paths()
        if dirty:
            raise ValueError(f"Candidate pipeline files are dirty: {dirty}")


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


@contextmanager
def ledger_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def event_hash_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key not in {"event_hash", "event_id"}}


def _read_forward_events_unlocked(path: Path = FORWARD_LEDGER_PATH, *, verify: bool = True) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    events: list[dict[str, Any]] = []
    previous_hash = GENESIS_HASH
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Nieprawidłowy JSONL w {path}:{line_number}") from exc
        if verify:
            if event.get("previous_event_hash") != previous_hash:
                raise ValueError(f"Forward ledger hash-chain mismatch in {path}:{line_number}")
            expected_hash = sha256_payload(event_hash_payload(event))
            if event.get("event_hash") != expected_hash:
                raise ValueError(f"Forward ledger event hash mismatch in {path}:{line_number}")
            previous_hash = expected_hash
        events.append(event)
    return events


def load_forward_events(path: Path = FORWARD_LEDGER_PATH, *, verify: bool = True) -> list[dict[str, Any]]:
    with ledger_lock(path):
        return _read_forward_events_unlocked(path, verify=verify)


def _append_forward_event_unlocked(
    event: dict[str, Any],
    path: Path,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = json_safe(event)
    payload.setdefault("event_time_utc", utc_now_iso())
    payload.setdefault("code_commit", current_commit())
    previous_hash = events[-1]["event_hash"] if events else GENESIS_HASH
    payload["previous_event_hash"] = previous_hash
    payload["event_hash"] = sha256_payload(event_hash_payload(payload))
    payload.setdefault("event_id", payload["event_hash"][:24])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")
    events.append(payload)
    return payload


def append_forward_event(event: dict[str, Any], path: Path = FORWARD_LEDGER_PATH) -> dict[str, Any]:
    with ledger_lock(path):
        events = _read_forward_events_unlocked(path, verify=True)
        return _append_forward_event_unlocked(event, path, events)


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


def _aware_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    return timestamp


def frozen_at_timestamp(manifest: dict[str, Any]) -> pd.Timestamp:
    frozen = _aware_timestamp(manifest.get("frozen_at"))
    if frozen is None:
        raise ValueError("Candidate manifest does not contain a valid frozen_at timestamp.")
    return frozen


def snapshot_hash(snapshot: dict[str, Any]) -> str:
    return sha256_payload(snapshot)


def _snapshot_updated_at(snapshot: dict[str, Any]) -> pd.Timestamp:
    updated = _aware_timestamp(snapshot.get("updated_at"))
    if updated is None:
        raise ValueError("Snapshot has no valid updated_at timestamp.")
    return updated


def _row_signal_date(row: dict[str, Any], snapshot: dict[str, Any]) -> str | None:
    return _date_text(row.get("Data") or snapshot.get("updated_at"))


def snapshot_after_freeze(snapshot: dict[str, Any], manifest: dict[str, Any]) -> bool:
    return _snapshot_updated_at(snapshot) >= frozen_at_timestamp(manifest)


def signal_date_after_freeze(signal_date: str, manifest: dict[str, Any]) -> bool:
    frozen = frozen_at_timestamp(manifest)
    return pd.Timestamp(signal_date).date() >= frozen.tz_convert(WARSAW).date()


def snapshot_has_closed_daily_bars(
    snapshot: dict[str, Any],
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> bool:
    """Conservative guard for Candidate v1's after-close signal timing.

    Candidate v1 is USA/ETF only. If a signal date equals the snapshot day, the
    snapshot must have been written after the US close plus a small Warsaw-time
    buffer. Older signal dates are already closed bars.
    """
    if not rows:
        return True
    updated_local = _snapshot_updated_at(snapshot).tz_convert(WARSAW)
    buffer_minutes = int(manifest.get("execution_contract", {}).get("usa_etf_close_buffer_minutes", 20))
    close_minutes = 22 * 60 + buffer_minutes
    updated_minutes = updated_local.hour * 60 + updated_local.minute
    for row in rows:
        signal_date = _row_signal_date(row, snapshot)
        if signal_date is None:
            return False
        day = pd.Timestamp(signal_date).date()
        if day > updated_local.date():
            return False
        if day == updated_local.date() and updated_minutes < close_minutes:
            return False
    return True


def _symbol_is_candidate_scope(symbol: str, asset_class: str, manifest: dict[str, Any]) -> bool:
    symbol = symbol.strip().upper()
    asset_class = asset_class.strip()
    accepted_classes = set(manifest.get("scope", {}).get("accepted_snapshot_asset_classes") or ["USA / ETF"])
    if symbol.endswith("-USD") or symbol.endswith(".WA") or symbol.startswith("^") or "." in symbol:
        return False
    if asset_class in accepted_classes:
        return True
    return asset_class.upper() in {"USA", "ETF", "USA/ETF", "USA / ETF"}


def row_decision_reason(row: dict[str, Any], manifest: dict[str, Any]) -> tuple[str | None, str]:
    explicit = row.get("DecisionReason")
    if explicit:
        return str(explicit), "SNAPSHOT_EXPLICIT"
    contract = manifest.get("decision_contract", {})
    verdict = signal_verdict(
        SignalInputs(
            probability=_float(row, "P(wzrost)", 0.5),
            expected_return=_float(row, "Oczekiwany ruch", 0.0),
            quality=_text(row, "Jakość modelu", "NISKA — BRAK PRZEWAGI"),
            auc=_float(row, "AUC walidacji", 0.5),
            brier=_float(row, "Brier", 0.25),
            source=_text(row, "Tryb analizy", "ML"),
        ),
        threshold=float(contract.get("threshold", 0.55)),
        min_expected_return=float(contract.get("min_expected_return", 0.0)),
    )
    return verdict.reason, "DERIVED_FROM_SIGNAL_INPUTS"


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
    reason, _ = row_decision_reason(row, manifest)
    required_reason = contract.get("required_decision_reason")
    if required_reason and reason != required_reason:
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
    signal_date = _row_signal_date(row, snapshot)
    if signal_date is None:
        return None
    signal_id = forward_signal_id(row, manifest, signal_date)
    reason, reason_source = row_decision_reason(row, manifest)
    return {
        "event_type": SIGNAL_EVENT,
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_hash": manifest.get("manifest_hash") or frozen_hash(manifest),
        "signal_id": signal_id,
        "status": "OBSERVED",
        "symbol": _text(row, "Symbol").strip().upper(),
        "asset_class": _text(row, "Klasa"),
        "horizon": int(row.get("Horyzont")),
        "direction": manifest.get("scope", {}).get("direction", "LONG"),
        "execution": manifest.get("scope", {}).get("execution", "NEXT_OPEN"),
        "signal_date": signal_date,
        "snapshot_hash": snapshot_hash(snapshot),
        "source_snapshot_updated_at": snapshot.get("updated_at"),
        "source_snapshot_schema_version": snapshot.get("schema_version"),
        "source_snapshot_pipeline_hash": (snapshot.get("candidate_pipeline") or {}).get("pipeline_hash"),
        "analysis_mode": _text(row, "Tryb analizy"),
        "decision_label": _text(row, "Ocena"),
        "decision_reason": reason,
        "decision_reason_source": reason_source,
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
    allow_historical: bool = False,
    enforce_pipeline: bool = True,
    require_clean_tree: bool = True,
    require_closed_bar: bool = True,
) -> int:
    """Append new Candidate v1 signals from a completed monitor snapshot.

    Existing events are never edited. A signal is de-duplicated by candidate,
    symbol, signal date, horizon and direction so a repeated dashboard refresh
    does not create a second pending trade.
    """
    if snapshot.get("status") != "complete":
        return 0
    manifest = load_candidate_manifest(manifest_path)
    assert_forward_contract_ready(
        manifest,
        require_clean_tree=require_clean_tree,
        enforce_pipeline=enforce_pipeline,
    )
    if not allow_historical and not snapshot_after_freeze(snapshot, manifest):
        raise ValueError("Snapshot is older than Candidate v1 frozen_at; refusing historical backfill.")
    current_pipeline = pipeline_fingerprint()
    snapshot_pipeline = snapshot.get("candidate_pipeline") or {}
    if enforce_pipeline and snapshot_pipeline.get("pipeline_hash") != current_pipeline["pipeline_hash"]:
        raise ValueError("Snapshot pipeline hash missing or different from the frozen Candidate v1 pipeline.")

    raw_rows = [row for row in snapshot.get("records") or [] if isinstance(row, dict)]
    candidate_rows = [row for row in raw_rows if is_candidate_row(row, manifest)]
    if not allow_historical:
        candidate_rows = [
            row for row in candidate_rows
            if (signal_date := _row_signal_date(row, snapshot)) and signal_date_after_freeze(signal_date, manifest)
        ]
    if require_closed_bar and not snapshot_has_closed_daily_bars(snapshot, candidate_rows, manifest):
        raise ValueError("Snapshot contains same-day Candidate v1 signals before the daily bar is safely closed.")

    snapshot_id = snapshot_hash(snapshot)
    with ledger_lock(path):
        events = _read_forward_events_unlocked(path, verify=True)
        observed_ids = {
            event.get("signal_id")
            for event in events
            if event.get("event_type") == SIGNAL_EVENT
        }
        audit_exists = any(
            event.get("event_type") == SNAPSHOT_AUDIT_EVENT and event.get("snapshot_hash") == snapshot_id
            for event in events
        )
        added_signals = 0
        accepted = 0
        skipped = 0
        state = reconstruct_forward_state(events)

        if not audit_exists:
            _append_forward_event_unlocked({
                "event_type": SNAPSHOT_AUDIT_EVENT,
                "candidate_id": manifest["candidate_id"],
                "candidate_manifest_hash": manifest.get("manifest_hash") or frozen_hash(manifest),
                "snapshot_hash": snapshot_id,
                "status": "AUDITED",
                "snapshot_updated_at": snapshot.get("updated_at"),
                "snapshot_schema_version": snapshot.get("schema_version"),
                "snapshot_pipeline_hash": snapshot_pipeline.get("pipeline_hash"),
                "current_pipeline_hash": current_pipeline["pipeline_hash"],
                "records_total": len(raw_rows),
                "candidate_rows": len(candidate_rows),
                "errors": snapshot.get("errors") or {},
            }, path, events)

        for row in sort_candidate_rows(candidate_rows, snapshot):
            event = _observed_signal_event(row, snapshot, manifest)
            if event is None or event["signal_id"] in observed_ids:
                continue
            signal = _append_forward_event_unlocked(event, path, events)
            observed_ids.add(signal["signal_id"])
            added_signals += 1
            ok, reason, slot = portfolio_gate(signal, state, manifest)
            if ok:
                decision = {
                    "event_type": ACCEPT_EVENT,
                    "candidate_id": manifest["candidate_id"],
                    "candidate_manifest_hash": manifest.get("manifest_hash") or frozen_hash(manifest),
                    "signal_id": signal["signal_id"],
                    "status": "ACCEPTED",
                    "symbol": signal["symbol"],
                    "direction": signal["direction"],
                    "signal_date": signal["signal_date"],
                    "slot": int(slot),
                    "portfolio_slots": int(manifest.get("portfolio_contract", {}).get("portfolio_slots", 5)),
                    "max_positions": int(manifest.get("portfolio_contract", {}).get("max_positions", 5)),
                    "portfolio_decision": reason,
                }
                accepted += 1
            else:
                decision = {
                    "event_type": SKIP_EVENT,
                    "candidate_id": manifest["candidate_id"],
                    "candidate_manifest_hash": manifest.get("manifest_hash") or frozen_hash(manifest),
                    "signal_id": signal["signal_id"],
                    "status": "SKIPPED",
                    "symbol": signal["symbol"],
                    "direction": signal["direction"],
                    "signal_date": signal["signal_date"],
                    "skip_reason": reason,
                    "portfolio_decision": "SKIPPED",
                }
                skipped += 1
            _append_forward_event_unlocked(decision, path, events)
            state = reconstruct_forward_state(events)

        if not audit_exists and events and events[-1].get("event_type") == SNAPSHOT_AUDIT_EVENT:
            # No candidate signals were appended; the audit event itself is the daily proof of "no accepted setup".
            pass
        return added_signals


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
        elif event_type == ACCEPT_EVENT:
            current.update({
                "status": "ACCEPTED",
                "slot": event.get("slot"),
                "accepted_event_id": event.get("event_id"),
                "portfolio_decision": "ACCEPTED",
            })
        elif event_type == SKIP_EVENT:
            current.update({
                "status": "SKIPPED",
                "skip_reason": event.get("skip_reason"),
                "skip_event_id": event.get("event_id"),
                "portfolio_decision": "SKIPPED",
            })
        elif event_type == ENTRY_EVENT:
            current.update({
                "status": "OPEN",
                "slot": event.get("slot", current.get("slot")),
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


def _portfolio_decision_exists(events: list[dict[str, Any]], signal_id: str) -> bool:
    return any(
        event.get("signal_id") == signal_id and event.get("event_type") in {ACCEPT_EVENT, SKIP_EVENT}
        for event in events
    )


def _active_forward_positions(state: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item for item in state.values()
        if item.get("status") in {"ACCEPTED", "OPEN"}
    ]


def _next_free_slot(state: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> int | None:
    slots = int(manifest.get("portfolio_contract", {}).get("portfolio_slots", 5))
    occupied = {
        int(item["slot"]) for item in _active_forward_positions(state)
        if item.get("slot") is not None
    }
    for slot in range(1, slots + 1):
        if slot not in occupied:
            return slot
    return None


def portfolio_gate(signal: dict[str, Any], state: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> tuple[bool, str, int | None]:
    contract = manifest.get("portfolio_contract", {})
    active = _active_forward_positions(state)
    symbol = str(signal.get("symbol", "")).upper()
    if contract.get("one_position_per_symbol", True):
        if any(str(item.get("symbol", "")).upper() == symbol for item in active):
            return False, "POSITION_SKIPPED_SYMBOL_OPEN", None
    slot = _next_free_slot(state, manifest)
    if slot is None:
        return False, "POSITION_SKIPPED_NO_FREE_SLOT", None
    max_positions = int(contract.get("max_positions", contract.get("portfolio_slots", 5)) or 0)
    if max_positions > 0 and len(active) >= max_positions:
        return False, "POSITION_SKIPPED_MAX_POSITIONS", None
    return True, "POSITION_ACCEPTED", slot


def sort_candidate_rows(rows: list[dict[str, Any]], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]):
        return (
            _row_signal_date(row, snapshot) or "",
            _text(row, "Symbol").strip().upper(),
            int(row.get("Horyzont") or 0),
            -_float(row, "P(wzrost)", 0.5),
            -_float(row, "Oczekiwany ruch", 0.0),
        )

    return sorted(rows, key=key)


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
    enforce_pipeline: bool = True,
    require_clean_tree: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    """Append entry/close events when next-open execution data is available."""
    manifest = load_candidate_manifest(manifest_path)
    assert_forward_contract_ready(
        manifest,
        require_clean_tree=require_clean_tree,
        enforce_pipeline=enforce_pipeline,
    )
    errors: dict[str, str] = {}
    total_cost = float(manifest.get("execution_contract", {}).get("round_trip_cost", 0.0015))

    with ledger_lock(path):
        events = _read_forward_events_unlocked(path, verify=True)
        state = reconstruct_forward_state(events)
        for signal_id in sorted(state):
            signal = state[signal_id]
            if signal.get("candidate_id") != manifest.get("candidate_id"):
                continue
            if signal.get("status") in {"CLOSED", "SKIPPED", "OBSERVED"}:
                continue
            symbol = str(signal.get("symbol", "")).upper()
            if not symbol:
                continue
            try:
                execution = _next_open_execution(signal, _history_for(symbol, histories, years))
            except Exception as exc:
                errors[symbol] = str(exc)
                continue

            if signal.get("status") == "ACCEPTED" and execution.get("ready_for_entry"):
                if not _event_exists(events, signal_id, ENTRY_EVENT):
                    _append_forward_event_unlocked({
                        "event_type": ENTRY_EVENT,
                        "candidate_id": manifest["candidate_id"],
                        "candidate_manifest_hash": manifest.get("manifest_hash") or frozen_hash(manifest),
                        "signal_id": signal_id,
                        "status": "OPEN",
                        "symbol": symbol,
                        "direction": signal.get("direction", "LONG"),
                        "execution": "NEXT_OPEN",
                        "signal_date": signal.get("signal_date"),
                        "slot": signal.get("slot"),
                        "entry_date": execution["entry_date"],
                        "entry_price": execution["entry_price"],
                        "price_source": "Open",
                    }, path, events)
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
                    _append_forward_event_unlocked({
                        "event_type": CLOSE_EVENT,
                        "candidate_id": manifest["candidate_id"],
                        "candidate_manifest_hash": manifest.get("manifest_hash") or frozen_hash(manifest),
                        "signal_id": signal_id,
                        "status": "CLOSED",
                        "symbol": symbol,
                        "direction": direction,
                        "execution": "NEXT_OPEN",
                        "signal_date": signal.get("signal_date"),
                        "slot": signal.get("slot"),
                        "entry_date": signal.get("entry_date"),
                        "entry_price": entry_price,
                        "exit_date": execution["exit_date"],
                        "exit_price": exit_price,
                        "gross_return": gross_return,
                        "strategy_return": strategy_return,
                        "round_trip_cost": total_cost,
                        "hit": strategy_return > 0,
                        "price_source": "Open",
                    }, path, events)
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
        "observed_only": int((statuses == "OBSERVED").sum()) if len(statuses) else 0,
        "accepted_pending_entry": int((statuses == "ACCEPTED").sum()) if len(statuses) else 0,
        "skipped": int((statuses == "SKIPPED").sum()) if len(statuses) else 0,
        "open": int((statuses == "OPEN").sum()) if len(statuses) else 0,
        "closed": int((statuses == "CLOSED").sum()) if len(statuses) else 0,
        "event_counts": event_types.value_counts().to_dict() if len(event_types) else {},
        "closed_mean_return": float(closed_returns.mean()) if len(closed_returns) else None,
        "closed_median_return": float(closed_returns.median()) if len(closed_returns) else None,
        "closed_hit_rate": float((closed_returns > 0).mean()) if len(closed_returns) else None,
        "candidate_ids": sorted({str(event.get("candidate_id")) for event in events if event.get("candidate_id")}),
    }
