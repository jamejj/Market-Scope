from __future__ import annotations

from enum import Enum
from typing import Any


RADAR_COVERAGE_SCHEMA_VERSION = 8


class RadarCoverageState(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"


def radar_coverage_state(snapshot: Any) -> RadarCoverageState:
    """Validate and classify the persisted Radar coverage contract."""
    if not isinstance(snapshot, dict):
        return RadarCoverageState.INVALID
    schema_version = snapshot.get("schema_version")
    if type(schema_version) is not int or schema_version < RADAR_COVERAGE_SCHEMA_VERSION:
        return RadarCoverageState.INVALID

    coverage = snapshot.get("coverage")
    symbols = coverage.get("requested_symbols") if isinstance(coverage, dict) else None
    horizons = coverage.get("requested_horizons") if isinstance(coverage, dict) else None
    shortlist = snapshot.get("shortlist")
    if (
        not isinstance(symbols, list) or not symbols
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
        or not isinstance(horizons, list) or not horizons
        or any(type(horizon) is not int or horizon <= 0 for horizon in horizons)
        or len(set(horizons)) != len(horizons)
        or not isinstance(shortlist, list)
        or any(not isinstance(symbol, str) or symbol not in symbols for symbol in shortlist)
        or coverage.get("requested_horizons") != snapshot.get("horizons")
        or snapshot.get("universe_total") != len(symbols)
        or not isinstance(snapshot.get("errors"), dict)
    ):
        return RadarCoverageState.INVALID

    def pair_set(value: Any) -> set[tuple[str, int]] | None:
        if not isinstance(value, list):
            return None
        pairs: set[tuple[str, int]] = set()
        for pair in value:
            if (
                not isinstance(pair, list) or len(pair) != 2
                or not isinstance(pair[0], str) or not pair[0]
                or type(pair[1]) is not int or pair[1] <= 0
            ):
                return None
            pairs.add((pair[0], pair[1]))
        return pairs if len(pairs) == len(value) else None

    missing_any = False
    success_by_mode: dict[str, set[tuple[str, int]]] = {}
    for mode, expected in (
        ("fast", {(symbol, horizon) for symbol in symbols for horizon in horizons}),
        ("ml", {(symbol, horizon) for symbol in shortlist for horizon in horizons}),
    ):
        section = coverage.get(mode)
        if not isinstance(section, dict):
            return RadarCoverageState.INVALID
        declared = pair_set(section.get("expected_pairs"))
        successful = pair_set(section.get("successful_pairs"))
        missing = pair_set(section.get("missing_pairs"))
        if declared != expected or successful is None or missing is None:
            return RadarCoverageState.INVALID
        if not successful <= expected or missing != expected - successful:
            return RadarCoverageState.INVALID
        success_by_mode[mode] = successful
        missing_any |= bool(missing)

    records = snapshot.get("records")
    if not isinstance(records, list):
        return RadarCoverageState.INVALID
    observed: dict[str, set[tuple[str, int]]] = {"fast": set(), "ml": set()}
    for record in records:
        if not isinstance(record, dict):
            return RadarCoverageState.INVALID
        mode = str(record.get("Tryb analizy") or "").lower()
        symbol, horizon = record.get("Symbol"), record.get("Horyzont")
        if mode not in observed or not isinstance(symbol, str) or type(horizon) is not int:
            return RadarCoverageState.INVALID
        observed[mode].add((symbol, horizon))
    if not success_by_mode["ml"] <= observed["ml"]:
        return RadarCoverageState.INVALID
    if not success_by_mode["fast"] - success_by_mode["ml"] <= observed["fast"]:
        return RadarCoverageState.INVALID
    if not observed["ml"] <= success_by_mode["ml"] or not observed["fast"] <= success_by_mode["fast"]:
        return RadarCoverageState.INVALID

    state = (
        RadarCoverageState.PARTIAL
        if missing_any or snapshot["errors"]
        else RadarCoverageState.COMPLETE
    )
    if snapshot.get("coverage_status") != state.value.lower():
        return RadarCoverageState.INVALID
    return state
