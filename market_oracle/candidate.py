from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .engine import scan_market
from .forward import (
    CANDIDATE_MANIFEST_PATH,
    CANDIDATE_SNAPSHOT_PATH,
    FORWARD_LEDGER_PATH,
    FORWARD_UNIVERSE_PATH,
    assert_forward_contract_ready,
    frozen_hash,
    load_candidate_manifest,
    load_forward_events,
    load_forward_universe,
    pipeline_fingerprint,
    record_snapshot_forward_signals,
    refresh_forward_ledger,
    summarize_forward_run_events,
    verify_frozen_hash,
)
from .integrity import (
    SnapshotIntegrityError,
    validate_candidate_snapshot_integrity,
    validate_canonical_candidate_universe,
)
from .signals import SignalInputs, signal_verdict


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def save_candidate_snapshot(snapshot: dict, path: Path = CANDIDATE_SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_json_safe(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_candidate_snapshot(path: Path = CANDIDATE_SNAPSHOT_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def add_explicit_decision_reason(row: dict, manifest: dict) -> dict:
    contract = manifest.get("decision_contract", {})
    verdict = signal_verdict(
        SignalInputs(
            probability=float(row.get("P(wzrost)") or 0.5),
            expected_return=float(row.get("Oczekiwany ruch") or 0.0),
            quality=str(row.get("Jakość modelu") or "NISKA — BRAK PRZEWAGI"),
            auc=None if row.get("AUC walidacji") is None else float(row.get("AUC walidacji")),
            brier=None if row.get("Brier") is None else float(row.get("Brier")),
            source=str(row.get("Tryb analizy") or "ML"),
        ),
        threshold=float(contract.get("threshold", 0.55)),
        min_expected_return=float(contract.get("min_expected_return", 0.0)),
    )
    enriched = dict(row)
    enriched["DecisionReason"] = verdict.reason
    enriched["DecisionLabel"] = verdict.label
    return enriched


def build_candidate_snapshot(
    *,
    manifest_path: Path = CANDIDATE_MANIFEST_PATH,
    universe_path: Path = FORWARD_UNIVERSE_PATH,
    years: int = 8,
    scan_fn: Callable[[list[str], int, int], tuple[pd.DataFrame, dict[str, str]]] | None = None,
    updated_at: str | None = None,
) -> dict:
    manifest = load_candidate_manifest(manifest_path)
    universe = load_forward_universe(universe_path)
    canonical_universe = load_forward_universe(FORWARD_UNIVERSE_PATH)
    if not verify_frozen_hash(canonical_universe, "universe_hash"):
        raise SnapshotIntegrityError("canonical frozen universe hash mismatch")
    if not verify_frozen_hash(universe, "universe_hash"):
        raise SnapshotIntegrityError(f"Forward universe hash mismatch: {universe_path}")
    validate_canonical_candidate_universe(
        universe,
        canonical_universe,
        expected_candidate_id=str(manifest.get("candidate_id") or ""),
    )
    symbols = [str(symbol).upper() for symbol in universe.get("symbols") or []]
    raw_records: list[dict] = []
    errors: dict[str, str] = {}
    completed: list[str] = []
    scan = scan_fn or (lambda symbols_arg, horizon_arg, years_arg: scan_market(symbols_arg, horizon=horizon_arg, years=years_arg))
    horizon = int((universe.get("config", {}).get("horizons") or [20])[0])

    for symbol in symbols:
        frame, failure = scan([symbol], horizon, years)
        if failure:
            errors.update({str(key).upper(): str(value) for key, value in failure.items()})
            continue
        if frame.empty or symbol not in set(frame.get("Symbol", pd.Series(dtype=object)).astype(str).str.upper()):
            errors[symbol] = "NO_CANDIDATE_ROW"
            continue
        raw_records.extend(frame.to_dict("records"))
        completed.append(symbol)

    failed = sorted(set(symbols) - set(completed))
    full_coverage = len(failed) == 0
    timestamp = updated_at or datetime.now(timezone.utc).isoformat()
    snapshot = {
        "status": "complete" if full_coverage else "partial",
        "started_at": timestamp,
        "updated_at": timestamp,
        "schema_version": 1,
        "scan_mode": "candidate_v1_full_ml",
        "scan_phase": "complete" if full_coverage else "partial",
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_hash": manifest.get("manifest_hash") or frozen_hash(manifest),
        "candidate_pipeline": pipeline_fingerprint(),
        "forward_universe": {
            "universe_id": universe["universe_id"],
            "universe_hash": universe["universe_hash"],
            "requested_symbols": symbols,
            "completed_symbols": completed,
            "failed_symbols": failed,
            "full_coverage": full_coverage,
        },
        "horizon": horizon,
        "horizons": [horizon],
        "years": int(years),
        "completed": len(completed),
        "total": len(symbols),
        "records": raw_records,
        "errors": errors,
    }
    validate_candidate_snapshot_integrity(snapshot, expected_symbols=symbols)
    snapshot["records"] = [
        add_explicit_decision_reason(_json_safe(row), manifest)
        for row in raw_records
    ]
    return snapshot


def run_candidate_forward_cycle(
    *,
    manifest_path: Path = CANDIDATE_MANIFEST_PATH,
    universe_path: Path = FORWARD_UNIVERSE_PATH,
    snapshot_path: Path = CANDIDATE_SNAPSHOT_PATH,
    ledger_path: Path = FORWARD_LEDGER_PATH,
    years: int = 8,
    record: bool = True,
    refresh_first: bool = True,
    require_closed_bar: bool = True,
    enforce_pipeline: bool = True,
    require_clean_tree: bool = True,
    scan_fn: Callable[[list[str], int, int], tuple[pd.DataFrame, dict[str, str]]] | None = None,
    refresh_histories: dict[str, pd.DataFrame] | None = None,
    updated_at: str | None = None,
) -> tuple[dict, dict]:
    manifest = load_candidate_manifest(manifest_path)
    assert_forward_contract_ready(manifest, require_clean_tree=require_clean_tree, enforce_pipeline=enforce_pipeline)
    snapshot = build_candidate_snapshot(
        manifest_path=manifest_path,
        universe_path=universe_path,
        years=years,
        scan_fn=scan_fn,
        updated_at=updated_at,
    )
    refresh_errors: dict[str, str] = {}
    try:
        before_events = load_forward_events(ledger_path)
    except Exception:
        before_events = []
    before_count = len(before_events)
    events_after_refresh = before_events
    if refresh_first:
        events_after_refresh, _, refresh_errors = refresh_forward_ledger(
            path=ledger_path,
            manifest_path=manifest_path,
            years=years,
            histories=refresh_histories,
            enforce_pipeline=enforce_pipeline,
            require_clean_tree=require_clean_tree,
        )
    refresh_events = events_after_refresh[before_count:]
    snapshot["pre_scan_refresh_errors"] = refresh_errors
    save_candidate_snapshot(snapshot, snapshot_path)
    added = 0
    before_record_count = len(events_after_refresh)
    after_record_events = events_after_refresh
    if record:
        added = record_snapshot_forward_signals(
            snapshot,
            path=ledger_path,
            manifest_path=manifest_path,
            universe_path=universe_path,
            require_closed_bar=require_closed_bar,
            enforce_pipeline=enforce_pipeline,
            require_clean_tree=require_clean_tree,
        )
        after_record_events = load_forward_events(ledger_path)
    record_events = after_record_events[before_record_count:]
    run_summary = summarize_forward_run_events([*refresh_events, *record_events])
    return snapshot, {"added_signals": added, "refresh_errors": refresh_errors, **run_summary}
