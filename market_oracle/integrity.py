from __future__ import annotations

from datetime import date, datetime
import math
from numbers import Real
from typing import Any, Iterable


INTEGRITY_EXIT_CODE = 65
INTEGRITY_FAILURE_KIND = "INTEGRITY_FAILED"
CANDIDATE_SNAPSHOT_SCHEMA_VERSION = 2


class SnapshotIntegrityError(ValueError):
    """Operational guard failure before the frozen Candidate v1 protocol runs."""

    def __init__(self, errors: Iterable[str] | str):
        items = [errors] if isinstance(errors, str) else [str(item) for item in errors]
        self.errors = items
        self.failure_kind = INTEGRITY_FAILURE_KIND
        self.exit_code = INTEGRITY_EXIT_CODE
        super().__init__("Candidate snapshot integrity failed: " + "; ".join(items))


def normalize_target_session_date(
    value: date | str | None,
    *,
    required: bool,
    context: str = "target_session_date",
) -> str | None:
    """Return an exact ISO session date or fail closed."""

    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise SnapshotIntegrityError(f"{context} is missing")
        return None
    if isinstance(value, datetime) or isinstance(value, bool):
        raise SnapshotIntegrityError(f"{context} is malformed")
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise SnapshotIntegrityError(f"{context} is malformed")
    text = value.strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise SnapshotIntegrityError(f"{context} is malformed") from exc
    if text != parsed.isoformat():
        raise SnapshotIntegrityError(f"{context} is malformed")
    return text


def validate_candidate_snapshot_session(
    snapshot: dict[str, Any],
    *,
    target_session_date: date | str | None = None,
    require_target: bool,
) -> str | None:
    """Bind a Candidate snapshot and every raw row to one market session."""

    explicit_target = normalize_target_session_date(
        target_session_date,
        required=require_target,
        context="target_session_date",
    )
    snapshot_target = normalize_target_session_date(
        snapshot.get("target_session_date"),
        required=require_target or explicit_target is not None,
        context="snapshot target_session_date",
    )
    if require_target and snapshot.get("schema_version") != CANDIDATE_SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotIntegrityError(
            f"canonical proof input requires schema_version {CANDIDATE_SNAPSHOT_SCHEMA_VERSION}"
        )
    if explicit_target is not None and snapshot_target != explicit_target:
        raise SnapshotIntegrityError(
            f"snapshot target {snapshot_target!r} does not match runner target {explicit_target!r}"
        )
    effective_target = explicit_target or snapshot_target
    if effective_target is None:
        return None

    records = snapshot.get("records")
    if not isinstance(records, list):
        raise SnapshotIntegrityError("snapshot records are missing for target-session validation")
    errors: list[str] = []
    for index, row in enumerate(records):
        symbol = _symbol(row, index) if isinstance(row, dict) else f"record[{index}]"
        if not isinstance(row, dict) or "Data" not in row or row.get("Data") is None:
            errors.append(f"{symbol}: Data is missing")
            continue
        try:
            row_date = normalize_target_session_date(
                row.get("Data"),
                required=True,
                context=f"{symbol}: Data",
            )
        except SnapshotIntegrityError as exc:
            errors.extend(exc.errors)
            continue
        if row_date != effective_target:
            errors.append(
                f"{symbol}: Data {row_date} does not match target session {effective_target}"
            )
    if errors:
        raise SnapshotIntegrityError(errors)
    return effective_target


def validate_canonical_candidate_universe(
    universe: dict[str, Any],
    canonical_universe: dict[str, Any],
    *,
    expected_candidate_id: str,
) -> None:
    """Bind an operational universe file to the versioned Candidate v1 contract."""

    errors: list[str] = []
    canonical_symbols = [
        str(symbol).strip().upper()
        for symbol in canonical_universe.get("symbols") or []
    ]
    supplied_symbols = [
        str(symbol).strip().upper()
        for symbol in universe.get("symbols") or []
    ]
    if len(canonical_symbols) != 5:
        errors.append("canonical frozen universe does not contain exactly five symbols")
    if universe.get("universe_hash") != canonical_universe.get("universe_hash"):
        errors.append("universe_hash does not match the canonical frozen universe")
    if universe.get("universe_id") != canonical_universe.get("universe_id"):
        errors.append("universe_id does not match the canonical frozen universe")
    if universe.get("candidate_id") != expected_candidate_id:
        errors.append("candidate_id does not match Candidate v1")
    if canonical_universe.get("candidate_id") != expected_candidate_id:
        errors.append("canonical frozen universe candidate_id does not match Candidate v1")
    if supplied_symbols != canonical_symbols:
        errors.append("ordered symbols do not match the canonical frozen universe")
    if errors:
        raise SnapshotIntegrityError(errors)


def validate_canonical_candidate_manifest(
    manifest: dict[str, Any],
    canonical_manifest: dict[str, Any],
) -> None:
    """Bind a supplied Candidate manifest to the versioned Candidate v1 contract."""

    errors: list[str] = []
    if manifest.get("manifest_hash") != canonical_manifest.get("manifest_hash"):
        errors.append("manifest_hash does not match the canonical frozen manifest")
    if manifest.get("candidate_id") != canonical_manifest.get("candidate_id"):
        errors.append("candidate_id does not match the canonical frozen manifest")
    if errors:
        raise SnapshotIntegrityError(errors)


def _symbol(row: dict[str, Any], index: int) -> str:
    value = row.get("Symbol")
    if value is None or not str(value).strip():
        return f"record[{index}]"
    return str(value).strip().upper()


def _raw_finite_number(
    row: dict[str, Any],
    *,
    key: str,
    field_name: str,
    symbol: str,
    errors: list[str],
) -> float | None:
    if key not in row or row.get(key) is None:
        errors.append(f"{symbol}: {field_name} is missing")
        return None
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        errors.append(f"{symbol}: {field_name} is not a raw numeric value")
        return None
    number = float(value)
    if not math.isfinite(number):
        errors.append(f"{symbol}: {field_name} is non-finite")
        return None
    return number


def validate_candidate_snapshot_integrity(
    snapshot: dict[str, Any],
    *,
    expected_symbols: Iterable[str] | None = None,
    require_full_universe: bool = True,
) -> None:
    """Validate raw Candidate inputs without interpreting their market direction.

    This is an operational allow/block interlock outside the frozen decision
    protocol. Finite expected returns of any sign, including exactly zero, are
    valid. Probabilities are valid only when finite and inside the closed [0, 1]
    interval.
    """

    errors: list[str] = []
    raw_records = snapshot.get("records")
    if not isinstance(raw_records, list):
        raw_records = []
        errors.append("snapshot records are missing")
    records = [row for row in raw_records if isinstance(row, dict)]
    if len(records) != len(raw_records):
        errors.append("snapshot contains a non-object record")

    record_symbols = [_symbol(row, index) for index, row in enumerate(records)]
    duplicates = sorted({symbol for symbol in record_symbols if record_symbols.count(symbol) > 1})
    if duplicates:
        errors.append(f"duplicate snapshot symbols: {duplicates}")

    for index, row in enumerate(records):
        symbol = _symbol(row, index)
        probability = _raw_finite_number(
            row,
            key="P(wzrost)",
            field_name="probability_up",
            symbol=symbol,
            errors=errors,
        )
        if probability is not None and not 0.0 <= probability <= 1.0:
            errors.append(f"{symbol}: probability_up is outside [0, 1]")
        _raw_finite_number(
            row,
            key="Oczekiwany ruch",
            field_name="expected_return",
            symbol=symbol,
            errors=errors,
        )

    if require_full_universe:
        expected = [str(symbol).strip().upper() for symbol in (expected_symbols or [])]
        expected_set = set(expected)
        actual_set = set(record_symbols)
        if snapshot.get("status") != "complete":
            errors.append(f"snapshot status is {snapshot.get('status')!r}, not complete")
        if not expected:
            errors.append("expected frozen universe is missing")
        elif len(records) != len(expected) or actual_set != expected_set:
            errors.append(
                "snapshot records do not equal the exact frozen universe: "
                f"missing={sorted(expected_set - actual_set)} extra={sorted(actual_set - expected_set)}"
            )

        meta = snapshot.get("forward_universe") or {}
        requested = [str(symbol).strip().upper() for symbol in meta.get("requested_symbols") or []]
        completed = [str(symbol).strip().upper() for symbol in meta.get("completed_symbols") or []]
        failed = [str(symbol).strip().upper() for symbol in meta.get("failed_symbols") or []]
        if requested != expected:
            errors.append("requested symbols do not equal the ordered frozen universe")
        if completed != expected:
            errors.append("completed symbols do not equal the ordered frozen universe")
        if failed:
            errors.append(f"failed Candidate v1 frozen-universe symbols: {failed}")
        if not bool(meta.get("full_coverage")):
            errors.append("full frozen universe coverage is false (full_coverage=false)")

    if errors:
        raise SnapshotIntegrityError(errors)
