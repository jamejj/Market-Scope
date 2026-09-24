"""Regression tests for Radar workflow progress versus actual coverage."""

import ast
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import market_oracle.monitor as monitor
from market_oracle.journal import JournalSnapshotEligibility, journal_snapshot_eligibility
from market_oracle.presentation import _scan_stage_text, build_start_guidance
from market_oracle.radar_contract import RadarCoverageState, radar_coverage_state


def _row(symbol, horizon, mode="FAST"):
    return {**{field: 0 for field in monitor.EXPECTED_RECORD_FIELDS}, **{
        "Symbol": symbol,
        "Horyzont": horizon,
        "Klasa": "USA",
        "Tryb analizy": mode,
        "Score": 2.0,
        "Deep score": 50.0,
        "P(wzrost)": 0.61 if mode == "ML" else None,
        "Oczekiwany ruch": 0.03 if mode == "ML" else None,
    }}


def _app_functions(*names):
    source = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    namespace = {"pd": pd, "_unique_symbols": lambda frame: int(frame["Symbol"].nunique()) if not frame.empty else 0}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


def test_failed_fast_and_ml_attempts_do_not_claim_successful_coverage(tmp_path, monkeypatch):
    def fast(symbols, horizons, years):
        symbol = symbols[0]
        if symbol == "B":
            return pd.DataFrame(), {symbol: "provider failure"}
        return pd.DataFrame([_row(symbol, 1), _row(symbol, 5)]), {}

    monkeypatch.setattr(monitor, "scan_market_fast", fast)
    monkeypatch.setattr(monitor, "scan_market_multi", lambda symbols, horizons, years: (pd.DataFrame(), {"A": "model failure"}))
    monkeypatch.setattr(monitor, "select_deep_shortlist", lambda frame, limit: ["A"])

    result = monitor.run_signal_scan(["A", "B"], horizons=(1, 5), path=tmp_path / "signals.json")

    assert result["status"] == "complete"  # workflow finished
    assert result["coverage_status"] == "partial"
    assert result["completed"] == result["total"] == 3  # attempts, not coverage
    assert result["coverage"]["requested_symbols"] == ["A", "B"]
    assert result["coverage"]["fast"]["expected_pairs"] == [["A", 1], ["A", 5], ["B", 1], ["B", 5]]
    assert result["coverage"]["fast"]["successful_pairs"] == [["A", 1], ["A", 5]]
    assert result["coverage"]["fast"]["missing_pairs"] == [["B", 1], ["B", 5]]
    assert result["coverage"]["ml"]["expected_pairs"] == [["A", 1], ["A", 5]]
    assert result["coverage"]["ml"]["successful_pairs"] == []
    assert result["coverage"]["ml"]["missing_pairs"] == [["A", 1], ["A", 5]]
    assert {(row["Symbol"], row["Horyzont"], row["Tryb analizy"]) for row in result["records"]} == {
        ("A", 1, "FAST"), ("A", 5, "FAST")
    }


def test_missing_one_fast_horizon_is_partial_even_without_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "scan_market_fast", lambda symbols, horizons, years: (pd.DataFrame([_row("A", 1)]), {}))
    monkeypatch.setattr(monitor, "select_deep_shortlist", lambda frame, limit: [])

    result = monitor.run_signal_scan(["A"], horizons=(1, 5), path=tmp_path / "signals.json")

    assert result["status"] == "complete"
    assert result["errors"] == {}
    assert result["coverage_status"] == "partial"
    assert result["coverage"]["fast"]["missing_pairs"] == [["A", 5]]


def test_all_expected_fast_and_ml_pairs_are_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "scan_market_fast", lambda symbols, horizons, years: (pd.DataFrame([_row("A", 1), _row("A", 5)]), {}))
    monkeypatch.setattr(monitor, "scan_market_multi", lambda symbols, horizons, years: (pd.DataFrame([_row("A", 1, "ML"), _row("A", 5, "ML")]), {}))
    monkeypatch.setattr(monitor, "select_deep_shortlist", lambda frame, limit: ["A"])

    result = monitor.run_signal_scan(["A"], horizons=(1, 5), path=tmp_path / "signals.json")

    assert result["coverage_status"] == "complete"
    assert result["coverage"]["fast"]["successful_pairs"] == [["A", 1], ["A", 5]]
    assert result["coverage"]["ml"]["successful_pairs"] == [["A", 1], ["A", 5]]
    assert {row["Tryb analizy"] for row in result["records"]} == {"ML"}


def test_canonical_coverage_contract_distinguishes_full_partial_and_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(
        monitor,
        "scan_market_fast",
        lambda symbols, horizons, years: (pd.DataFrame([_row("A", 1)]), {}),
    )
    monkeypatch.setattr(monitor, "select_deep_shortlist", lambda frame, limit: [])
    partial = monitor.run_signal_scan(["A"], horizons=(1, 5), path=tmp_path / "partial.json")

    monkeypatch.setattr(
        monitor,
        "scan_market_fast",
        lambda symbols, horizons, years: (pd.DataFrame([_row("A", 1), _row("A", 5)]), {}),
    )
    complete = monitor.run_signal_scan(["A"], horizons=(1, 5), path=tmp_path / "complete.json")

    assert radar_coverage_state(complete) is RadarCoverageState.COMPLETE
    assert radar_coverage_state(partial) is RadarCoverageState.PARTIAL
    assert radar_coverage_state({**partial, "schema_version": 6}) is RadarCoverageState.INVALID
    assert radar_coverage_state({**partial, "coverage_status": "complete"}) is RadarCoverageState.INVALID
    assert radar_coverage_state({**complete, "coverage": None}) is RadarCoverageState.INVALID


@pytest.mark.parametrize("schema_version", [True, "8", {"version": 8}])
def test_malformed_schema_version_is_invalid_and_stale(schema_version):
    snapshot = {
        "status": "complete",
        "schema_version": schema_version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    assert radar_coverage_state(snapshot) is RadarCoverageState.INVALID
    assert monitor.snapshot_is_stale(snapshot) is True


def test_fresh_partial_snapshot_is_not_stale_but_missing_or_malformed_coverage_is(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "scan_market_fast", lambda symbols, horizons, years: (pd.DataFrame(), {"A": "failure"}))
    monkeypatch.setattr(monitor, "select_deep_shortlist", lambda frame, limit: [])
    snapshot = monitor.run_signal_scan(["A"], path=tmp_path / "signals.json")

    assert monitor.snapshot_is_stale(snapshot) is False
    assert monitor.snapshot_is_stale({**snapshot, "coverage": None}) is True
    assert monitor.snapshot_is_stale({**snapshot, "coverage_status": "complete"}) is True
    assert monitor.snapshot_is_stale({**snapshot, "schema_version": 7}) is True

    malformed = {**snapshot, "coverage": None}
    assert radar_coverage_state(malformed) is RadarCoverageState.INVALID
    assert journal_snapshot_eligibility(malformed) is JournalSnapshotEligibility.SKIPPED_INVALID_COVERAGE


def test_snapshot_claiming_accepted_ml_pair_without_its_record_is_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "scan_market_fast", lambda symbols, horizons, years: (pd.DataFrame([_row("A", h) for h in (1, 5, 20)]), {}))
    monkeypatch.setattr(monitor, "scan_market_multi", lambda symbols, horizons, years: (pd.DataFrame([_row("A", h, "ML") for h in (1, 5, 20)]), {}))
    monkeypatch.setattr(monitor, "select_deep_shortlist", lambda frame, limit: ["A"])
    snapshot = monitor.run_signal_scan(["A"], path=tmp_path / "signals.json")

    assert monitor.snapshot_is_stale(snapshot) is False
    assert monitor.snapshot_is_stale({**snapshot, "records": []}) is True
    extra_fast = {**_row("B", 1), "Data": "2026-09-18"}
    assert monitor.snapshot_is_stale({**snapshot, "records": [*snapshot["records"], extra_fast]}) is True


def test_three_presentation_paths_do_not_infer_fast_success_from_workflow_or_one_row():
    snapshot = {"status": "complete", "coverage_status": "partial", "universe_total": 10, "fast_completed": 0,
                "coverage": {"fast": {"successful_pairs": [], "expected_pairs": [["A", 1]]}, "ml": {"successful_pairs": [], "expected_pairs": []}},
                "completed": 10, "total": 10, "records": []}
    frame = pd.DataFrame([_row("A", 1)])
    app = _app_functions("classify_signal_errors", "signal_scan_contract", "scan_stage_summary")

    assert "FAST 0/10" in _scan_stage_text(snapshot, 10)[0]
    assert "pokrycie częściowe" in _scan_stage_text(snapshot, 10)[0]
    assert app["signal_scan_contract"](frame, snapshot)["fast_completed"] == 0
    stage = app["scan_stage_summary"](snapshot, 10)
    assert stage["fast_completed"] == 0
    assert stage["top_status"] != "Gotowy"


def test_start_guidance_labels_finished_partial_scan_and_keeps_valid_rows():
    snapshot = {"status": "complete", "coverage_status": "partial", "updated_at": datetime.now(timezone.utc).isoformat(),
                "universe_total": 2, "fast_completed": 2, "ml_total": 0, "records": [_row("A", 1)]}

    guidance = build_start_guidance(snapshot=snapshot, cockpit={}, automation={}, proof_state={"label": "OK"}, radar_stale=False)

    assert "częściow" in str(guidance).lower()
    assert any(card["id"] == "radar_overview" for card in guidance["cards"])


def test_radar_completion_banner_never_calls_partial_coverage_success():
    app = _app_functions("radar_scan_completion_view")
    partial = {"status": "complete", "coverage_status": "partial", "updated_at": "2026-09-18T12:00:00+00:00"}

    view = app["radar_scan_completion_view"](partial, stale=False)

    assert view["tone"] == "warning"
    assert "częściow" in view["text"].lower()


def test_radar_csv_export_carries_partial_coverage_without_mutating_raw_rows():
    app = _app_functions("radar_export_frame")
    raw = pd.DataFrame([_row("A", 1)])
    original = raw.copy(deep=True)
    snapshot = {"coverage_status": "partial", "coverage": {
        "fast": {"expected_pairs": [["A", 1], ["B", 1]], "successful_pairs": [["A", 1]]},
        "ml": {"expected_pairs": [], "successful_pairs": []},
    }}

    exported = app["radar_export_frame"](raw, snapshot)

    pd.testing.assert_frame_equal(raw, original)
    assert "partial" in exported.to_csv(index=False)
    assert exported.loc[0, "FAST coverage"] == "1/2"
    assert exported.loc[0, "Coverage status"] == "partial"
