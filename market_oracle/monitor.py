from __future__ import annotations

import json
import time
import fcntl
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .catalog import CATEGORIES, CRYPTO, ETF_CATEGORIES
from .engine import scan_market, scan_market_fast, scan_market_multi
from .journal import (
    JournalSnapshotEligibility,
    journal_error_code,
    journal_snapshot_eligibility,
    record_snapshot_signals,
)
from .product_verdict import product_forecast_verdict, radar_ml_input_error_code
from .radar_contract import RadarCoverageState, radar_coverage_state


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_PATH = DATA_DIR / "signals.json"
LOCK_PATH = DATA_DIR / "signals.lock"
SCAN_SCHEMA_VERSION = 8
DEEP_SCAN_LIMIT = 36
EXPECTED_HORIZONS = {1, 5, 20}
EXPECTED_RECORD_FIELDS = {
    "Symbol", "Klasa", "Horyzont", "Setup", "Zwrot 1d", "Zwrot 5d", "Zwrot 20d",
    "Radar momentum", "Radar score", "Risk/reward", "Edge score", "Akcja radaru",
    "Setup score", "Setup grade", "Momentum score", "Trend score", "Risk control",
    "Liquidity score", "Model edge", "Teza radaru", "Tryb analizy", "Deep score",
}


def default_universe() -> list[str]:
    groups = [
        list(CATEGORIES["GPW — największe spółki"].values())[:14],
        list(CATEGORIES["GPW — średnie i mniejsze"].values())[:12],
        list(CATEGORIES["USA — technologia i półprzewodniki"].values())[:16],
        list(CATEGORIES["USA — banki i finanse"].values())[:6],
        list(CATEGORIES["USA — zdrowie i biotechnologia"].values())[:6],
        list(CATEGORIES["USA — przemysł i energia"].values())[:5],
        list(CATEGORIES["USA — handel, media i usługi"].values())[:6],
        list(CATEGORIES["USA — mniejsze i spekulacyjne"].values())[:8],
        list(CATEGORIES["Świat — spółki notowane w USA"].values())[:4],
        list(ETF_CATEGORIES["Szeroki rynek USA"].values())[:7],
        list(ETF_CATEGORIES["Sektory i technologia"].values())[:6],
        list(ETF_CATEGORIES["Świat i regiony"].values())[:4],
        list(ETF_CATEGORIES["Surowce"].values())[:4],
        list(ETF_CATEGORIES["Tematyczne i wzrostowe"].values())[:4],
        list(CRYPTO.values()),
    ]
    return list(dict.fromkeys(symbol for group in groups for symbol in group))


def _json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    records = []
    for row in frame.to_dict("records"):
        record = dict(row)
        if record.get("Tryb analizy") == "ML":
            verdict = product_forecast_verdict(
                {
                    "probability_up": record.get("P(wzrost)"),
                    "expected_return": record.get("Oczekiwany ruch"),
                    "quality": record.get("Jakość modelu"),
                    "auc": record.get("AUC walidacji"),
                    "brier": record.get("Brier"),
                },
                source="RADAR",
            )
            record["Decision"] = verdict.decision
            record["DecisionReason"] = verdict.reason
        records.append({key: _json_value(value) for key, value in record.items()})
    return records


def _merge_rankable_records(
    rows_by_key: dict[tuple[str, int], dict],
    records: list[dict],
    default_horizon: int,
    errors: dict[str, str],
) -> None:
    for record in records:
        symbol = str(record.get("Symbol") or "UNKNOWN")
        error_code = radar_ml_input_error_code(record)
        if error_code is not None:
            errors[symbol] = error_code
            continue
        rows_by_key[(symbol, int(record.get("Horyzont") or default_horizon))] = record


def _coverage_section(expected: set[tuple[str, int]], successful: set[tuple[str, int]]) -> dict:
    def pairs(values: set[tuple[str, int]]) -> list[list]:
        return [[symbol, horizon] for symbol, horizon in sorted(values)]

    accepted = expected & successful
    return {
        "expected_pairs": pairs(expected),
        "successful_pairs": pairs(accepted),
        "missing_pairs": pairs(expected - accepted),
    }


def _coverage_metadata(
    symbols: list[str], horizons: tuple[int, ...], shortlist: list[str],
    fast_success: set[tuple[str, int]], ml_success: set[tuple[str, int]],
) -> dict:
    return {
        "requested_symbols": list(symbols),
        "requested_horizons": list(horizons),
        "fast": _coverage_section({(symbol, horizon) for symbol in symbols for horizon in horizons}, fast_success),
        "ml": _coverage_section({(symbol, horizon) for symbol in shortlist for horizon in horizons}, ml_success),
    }


def select_deep_shortlist(frame: pd.DataFrame, limit: int = DEEP_SCAN_LIMIT) -> list[str]:
    """Pick symbols for expensive ML after the cheap whole-market pass."""
    if frame.empty or "Symbol" not in frame:
        return []
    work = frame.copy()
    for column in ["Deep score", "Setup score", "Radar score", "Edge score", "Risk control"]:
        if column in work:
            work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
        else:
            work[column] = 0.0
    if "Klasa" not in work:
        work["Klasa"] = "Rynek"
    work["priority"] = (
        work["Deep score"] * 0.52
        + work["Setup score"] * 0.30
        + work["Radar score"].clip(lower=-5, upper=15) * 2.2
        + work["Edge score"].clip(lower=-5, upper=10) * 3.0
        + work["Risk control"].clip(lower=0, upper=100) * 0.08
    )
    ranked = work.sort_values("priority", ascending=False).drop_duplicates("Symbol")
    selected: list[str] = []
    per_class_seed = max(1, min(4, limit // max(1, ranked["Klasa"].nunique())))
    for _, group in ranked.groupby("Klasa", sort=False):
        for symbol in group.head(per_class_seed)["Symbol"]:
            if symbol not in selected:
                selected.append(symbol)
            if len(selected) >= limit:
                return selected
    for symbol in ranked["Symbol"]:
        if symbol not in selected:
            selected.append(symbol)
        if len(selected) >= limit:
            break
    return selected


def save_snapshot(payload: dict, path: Path = SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def radar_data_date(value) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        if len(text) == 10:
            return date.fromisoformat(text).isoformat()
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def radar_snapshot_provenance(snapshot: dict | None) -> dict:
    """Describe Radar computation time and daily market-data dates separately."""
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    raw_records = snapshot.get("records")
    records = raw_records if isinstance(raw_records, list) else []
    valid_dates: set[str] = set()
    missing = 0
    malformed = 0
    for record in records:
        if not isinstance(record, dict) or "Data" not in record:
            missing += 1
            continue
        value = record.get("Data")
        if value is None or value is pd.NA or value is pd.NaT or value == "":
            missing += 1
            continue
        parsed = radar_data_date(value)
        if parsed is None:
            malformed += 1
            continue
        valid_dates.add(parsed)

    ordered_dates = sorted(valid_dates)
    return {
        "computed_at": snapshot.get("updated_at"),
        "data_as_of_min": ordered_dates[0] if ordered_dates else None,
        "data_as_of_max": ordered_dates[-1] if ordered_dates else None,
        "distinct_data_dates": len(ordered_dates),
        "mixed_data_dates": len(ordered_dates) > 1,
        "missing_data_dates": missing,
        "malformed_data_dates": malformed,
        "invalid_data_dates": missing + malformed,
        "records_total": len(records),
        "source": "yfinance",
        "data_kind": "ADJUSTED_DAILY_OHLCV",
        "interval": "1d",
        "is_realtime": False,
    }


def _is_canonical_snapshot_path(path: Path) -> bool:
    return path.resolve() == SNAPSHOT_PATH.resolve()


def run_signal_scan(
    symbols: list[str] | None = None,
    horizon: int = 20,
    horizons: tuple[int, ...] | None = (1, 5, 20),
    years: int = 8,
    fast_years: int = 2,
    deep_limit: int = DEEP_SCAN_LIMIT,
    path: Path = SNAPSHOT_PATH,
) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_PATH if path == SNAPSHOT_PATH else path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return load_snapshot(path) or {"status": "running", "completed": 0, "total": 0, "records": [], "errors": {}}

    universe = symbols or default_universe()
    scan_horizons = horizons or (horizon,)
    fast_expected = {(symbol, item) for symbol in universe for item in scan_horizons}
    fast_success: set[tuple[str, int]] = set()
    ml_success: set[tuple[str, int]] = set()
    shortlist: list[str] = []
    started = datetime.now(timezone.utc)
    rows_by_key: dict[tuple[str, int], dict] = {}
    errors: dict[str, str] = {}
    try:
        from .forward import pipeline_fingerprint

        candidate_pipeline = pipeline_fingerprint()
    except Exception as exc:
        candidate_pipeline = {"error": str(exc)}
    canonical_snapshot = _is_canonical_snapshot_path(path)
    payload = {
        "status": "running", "started_at": started.isoformat(), "updated_at": None,
        "schema_version": SCAN_SCHEMA_VERSION,
        "candidate_pipeline": candidate_pipeline,
        "scan_mode": "two_stage", "scan_phase": "fast_radar",
        "deep_limit": deep_limit, "shortlist": [], "universe_total": len(universe),
        "fast_completed": 0, "ml_completed": 0, "ml_total": 0,
        "horizon": horizon, "horizons": list(scan_horizons), "years": years, "completed": 0, "total": len(universe),
        "records": [], "errors": errors,
        "journal_status": "NOT_ATTEMPTED" if canonical_snapshot else "NOT_APPLICABLE",
    }

    def update_coverage() -> None:
        payload["coverage"] = _coverage_metadata(universe, scan_horizons, shortlist, fast_success, ml_success)
        missing = payload["coverage"]["fast"]["missing_pairs"] or payload["coverage"]["ml"]["missing_pairs"]
        payload["coverage_status"] = "partial" if missing or errors else "complete"

    update_coverage()
    save_snapshot(payload, path)

    try:
        for completed, symbol in enumerate(universe, start=1):
            frame, failure = scan_market_fast([symbol], horizons=scan_horizons, years=fast_years)
            errors.update(failure)
            if not frame.empty:
                _merge_rankable_records(rows_by_key, _records(frame), horizon, errors)
                fast_success.update({
                    key for key, row in rows_by_key.items()
                    if key in fast_expected and row.get("Tryb analizy") == "FAST"
                })
            rows = sorted(rows_by_key.values(), key=lambda row: (row.get("Horyzont") or horizon, -(row.get("Deep score") or row.get("Score") or float("-inf"))))
            payload.update({"completed": completed, "fast_completed": completed, "records": rows, "errors": errors})
            update_coverage()
            save_snapshot(payload, path)

        fast_frame = pd.DataFrame(rows_by_key.values())
        shortlist = select_deep_shortlist(fast_frame, deep_limit)
        payload.update({
            "scan_phase": "deep_ml", "shortlist": shortlist, "ml_total": len(shortlist),
            "completed": len(universe), "total": len(universe) + len(shortlist),
            "records": sorted(rows_by_key.values(), key=lambda row: (row.get("Horyzont") or horizon, -(row.get("Deep score") or row.get("Score") or float("-inf")))),
        })
        update_coverage()
        save_snapshot(payload, path)

        ml_expected = {(symbol, item) for symbol in shortlist for item in scan_horizons}
        for ml_completed, symbol in enumerate(shortlist, start=1):
            if horizons:
                frame, failure = scan_market_multi([symbol], horizons=scan_horizons, years=years)
            else:
                frame, failure = scan_market([symbol], horizon=horizon, years=years)
            errors.update(failure)
            if not frame.empty:
                _merge_rankable_records(rows_by_key, _records(frame), horizon, errors)
                ml_success.update({
                    key for key, row in rows_by_key.items()
                    if key in ml_expected and row.get("Tryb analizy") == "ML"
                })
            rows = sorted(rows_by_key.values(), key=lambda row: (row.get("Horyzont") or horizon, -(row.get("Deep score") or row.get("Score") or float("-inf"))))
            payload.update({
                "completed": len(universe) + ml_completed, "ml_completed": ml_completed,
                "records": rows, "errors": errors,
            })
            update_coverage()
            save_snapshot(payload, path)

        rows = sorted(rows_by_key.values(), key=lambda row: (row.get("Horyzont") or horizon, -(row.get("Deep score") or row.get("Score") or float("-inf"))))
        payload.update({
            "status": "complete", "updated_at": datetime.now(timezone.utc).isoformat(),
            "scan_phase": "complete", "completed": len(universe) + len(shortlist),
            "total": len(universe) + len(shortlist), "records": rows, "errors": errors,
        })
        update_coverage()
        save_snapshot(payload, path)
        if canonical_snapshot:
            eligibility = journal_snapshot_eligibility(payload)
            if eligibility is not JournalSnapshotEligibility.ELIGIBLE:
                payload["journal_status"] = (
                    "SKIPPED_PARTIAL_COVERAGE"
                    if eligibility is JournalSnapshotEligibility.SKIPPED_PARTIAL_COVERAGE
                    else "SKIPPED_INVALID_COVERAGE"
                )
                payload.pop("journal_added", None)
                payload.pop("journal_error", None)
                payload.pop("journal_attempted_at", None)
            else:
                attempted_at = datetime.now(timezone.utc).isoformat()
                try:
                    added = record_snapshot_signals(payload)
                    payload.update({
                        "journal_status": "OK",
                        "journal_added": added,
                        "journal_attempted_at": attempted_at,
                    })
                    payload.pop("journal_error", None)
                except Exception as exc:
                    payload.update({
                        "journal_status": "FAILED",
                        "journal_error": journal_error_code(exc),
                        "journal_attempted_at": attempted_at,
                    })
                    payload.pop("journal_added", None)
            payload["forward_ledger_status"] = "not_recorded_generic_two_stage_scan"
            payload["forward_ledger_note"] = "Candidate v1 proof ledger uses run_candidate_forward.py with a frozen full-ML universe."
            save_snapshot(payload, path)
        return payload
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def snapshot_is_stale(snapshot: dict | None, max_age_hours: float = 6) -> bool:
    if not snapshot or snapshot.get("status") != "complete" or not snapshot.get("updated_at"):
        return True
    schema_version = snapshot.get("schema_version")
    if type(schema_version) is not int or schema_version < SCAN_SCHEMA_VERSION:
        return True
    if radar_coverage_state(snapshot) is RadarCoverageState.INVALID:
        return True
    horizons = set(snapshot.get("horizons") or [snapshot.get("horizon")])
    if not EXPECTED_HORIZONS.issubset(horizons):
        return True
    records = snapshot.get("records") or []
    if any(
        not isinstance(record, dict) or not EXPECTED_RECORD_FIELDS.issubset(record)
        for record in records
    ):
        return True
    from .journal import valid_machine_decision_contract

    if any(
        record.get("Tryb analizy") == "ML"
        and (
            not valid_machine_decision_contract(record)
            or radar_ml_input_error_code(record) is not None
        )
        for record in records
        if isinstance(record, dict)
    ):
        return True
    try:
        updated = datetime.fromisoformat(snapshot["updated_at"])
    except (TypeError, ValueError):
        return True
    return datetime.now(timezone.utc) - updated > timedelta(hours=max_age_hours)


def monitor_loop(interval_hours: float = 6, poll_seconds: int = 60) -> None:
    while True:
        try:
            snapshot = load_snapshot()
            if snapshot_is_stale(snapshot, interval_hours):
                run_signal_scan()
        except Exception as exc:
            snapshot = load_snapshot() or {}
            snapshot.update({"status": "error", "error": str(exc)})
            save_snapshot(snapshot)
        time.sleep(poll_seconds)
