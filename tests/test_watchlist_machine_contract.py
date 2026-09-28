from __future__ import annotations

import ast
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from market_oracle import watchlist as watchlist_module
from market_oracle.presentation import build_analysis_report
from market_oracle.product_verdict import MachineDecisionState, NON_DIRECTIONAL_REASONS


def canonical_snapshot(
    decision: object = 1,
    reason: object = "LONG_CONFIRMED",
    label: object = "LONG",
    **overrides,
) -> dict:
    return {
        "available": True,
        "symbol": "SPY",
        "horizon": 20,
        "created_at": "2026-09-01T12:00:00+00:00",
        "verdict_decision": decision,
        "verdict": reason,
        "verdict_label": label,
        "probability_up": 0.61,
        "expected_return": 0.03,
        "quality": "WYSOKA",
        **overrides,
    }


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (canonical_snapshot(), MachineDecisionState.LONG),
        (canonical_snapshot(-1, "SHORT_CONFIRMED", "SHORT"), MachineDecisionState.SHORT),
        *[
            (canonical_snapshot(0, reason, "legacy display copy"), MachineDecisionState.NEUTRAL)
            for reason in sorted(NON_DIRECTIONAL_REASONS)
        ],
    ],
)
def test_watchlist_machine_contract_accepts_only_canonical_pairs(snapshot, expected):
    assert watchlist_module.watchlist_machine_decision_state(snapshot) is expected


@pytest.mark.parametrize(
    "decision",
    [True, False, "1", "-1", "0", 1.0, -1.0, 0.0, 1.2, -0.4, float("nan"), float("inf"), float("-inf"), pd.NA, None],
)
def test_watchlist_machine_contract_rejects_coerced_missing_and_nonfinite_decisions(decision):
    snapshot = canonical_snapshot(decision=decision, reason="LONG_CONFIRMED", label="LONG")

    assert watchlist_module.watchlist_machine_decision_state(snapshot) is MachineDecisionState.INVALID


@pytest.mark.parametrize(
    ("decision", "reason"),
    [
        (1, "SHORT_CONFIRMED"),
        (1, "PROBABILITY_INSIDE_BAND"),
        (-1, "LONG_CONFIRMED"),
        (-1, "LOW_QUALITY"),
        (0, "LONG_CONFIRMED"),
        (0, "SHORT_CONFIRMED"),
        (0, "UNKNOWN"),
        (1, None),
        (-1, None),
        (0, None),
    ],
)
def test_watchlist_machine_contract_rejects_decision_reason_conflicts(decision, reason):
    assert (
        watchlist_module.watchlist_machine_decision_state(canonical_snapshot(decision, reason, "LONG"))
        is MachineDecisionState.INVALID
    )


@pytest.mark.parametrize(
    "snapshot",
    [
        {},
        {"verdict_label": "LONG"},
        {"label": "SHORT"},
        {"verdict": "LONG_CONFIRMED"},
        {"reason": "SHORT_CONFIRMED"},
        {"verdict_decision": None, "verdict_label": "LONG", "verdict": "LONG_CONFIRMED"},
    ],
)
def test_watchlist_copy_never_creates_machine_direction(snapshot):
    assert watchlist_module.watchlist_machine_decision_state(snapshot) is MachineDecisionState.INVALID


def test_changing_only_raw_verdict_label_does_not_change_machine_state():
    long_with_short_copy = canonical_snapshot(label="SHORT")
    neutral_with_long_copy = canonical_snapshot(0, "LOW_QUALITY", "LONG")

    assert watchlist_module.watchlist_machine_decision_state(long_with_short_copy) is MachineDecisionState.LONG
    assert watchlist_module.watchlist_machine_decision_state(neutral_with_long_copy) is MachineDecisionState.NEUTRAL
    assert watchlist_module.watchlist_machine_decision_label(long_with_short_copy) == "Potwierdzony kierunek wzrostowy"
    assert watchlist_module.watchlist_machine_decision_label(neutral_with_long_copy) == "Brak potwierdzenia kierunku"


def test_neutral_and_invalid_have_distinct_machine_state_and_display():
    neutral = canonical_snapshot(0, "LOW_QUALITY", "BRAK SYGNAŁU")
    invalid = {"verdict_label": "OBSERWUJ"}

    assert watchlist_module.watchlist_machine_decision_state(neutral) is MachineDecisionState.NEUTRAL
    assert watchlist_module.watchlist_machine_decision_state(invalid) is MachineDecisionState.INVALID
    assert watchlist_module.watchlist_machine_decision_label(neutral) == "Brak potwierdzenia kierunku"
    assert watchlist_module.watchlist_machine_decision_label(invalid) == "Klasyfikacja niedostępna"


@pytest.mark.parametrize("invalid_side", ["saved", "current"])
def test_invalid_machine_evidence_makes_directional_comparison_unavailable(invalid_side):
    saved = canonical_snapshot()
    current = canonical_snapshot()
    if invalid_side == "saved":
        saved = {**saved, "verdict_decision": "1"}
    else:
        current = {**current, "verdict_decision": 1.2}

    comparison = watchlist_module.compare_watch_item_to_current(saved, current, now="2026-09-02")

    assert comparison["comparison_status"] == "DECISION_UNAVAILABLE"
    assert comparison["label"] == "Nie można wiarygodnie porównać kierunku"
    assert comparison["verdict_transition"] is None
    assert not {
        "GAINED_CONFIRMATION",
        "STILL_CONFIRMED",
        "WEAKENED",
        "REVERSED",
    }.intersection({comparison["comparison_status"]})
    assert any("klasyfikac" in reason.lower() for reason in comparison["reasons"])


def test_canonical_comparison_preserves_existing_directional_transitions():
    long_saved = canonical_snapshot()
    neutral_saved = canonical_snapshot(0, "LOW_QUALITY", "BRAK SYGNAŁU")
    short_current = canonical_snapshot(-1, "SHORT_CONFIRMED", "SHORT")
    neutral_current = canonical_snapshot(0, "PROBABILITY_INSIDE_BAND", "OBSERWUJ")

    assert watchlist_module.compare_watch_item_to_current(long_saved, long_saved)["comparison_status"] == "STILL_CONFIRMED"
    assert watchlist_module.compare_watch_item_to_current(neutral_saved, long_saved)["comparison_status"] == "GAINED_CONFIRMATION"
    assert watchlist_module.compare_watch_item_to_current(long_saved, neutral_current)["comparison_status"] == "WEAKENED"
    assert watchlist_module.compare_watch_item_to_current(long_saved, short_current)["comparison_status"] == "REVERSED"


def test_legacy_item_without_machine_evidence_stays_loadable_but_unclassified(tmp_path):
    path = tmp_path / "watchlist.json"
    legacy_item = {
        "id": "legacy-spy",
        "symbol": "SPY",
        "horizon": 20,
        "created_at": "2026-09-01T12:00:00+00:00",
        "status": "ACTIVE",
        "thesis": "Historyczny wpis pozostaje widoczny.",
    }
    path.write_text(json.dumps([legacy_item]), encoding="utf-8")

    loaded = watchlist_module.load_watchlist(path)

    assert loaded == [legacy_item]
    assert watchlist_module.watchlist_machine_decision_state(loaded[0]) is MachineDecisionState.INVALID
    assert watchlist_module.watchlist_machine_decision_label(loaded[0]) == "Klasyfikacja niedostępna"


@pytest.mark.parametrize(
    ("probability", "expected_return", "expected_state"),
    [
        (0.61, 0.03, MachineDecisionState.LONG),
        (0.39, -0.03, MachineDecisionState.SHORT),
        (0.50, 0.00, MachineDecisionState.NEUTRAL),
    ],
)
def test_full_analysis_and_current_snapshot_producers_emit_canonical_states(
    probability, expected_return, expected_state
):
    result = {
        "symbol": "SPY",
        "last_date": pd.Timestamp("2026-09-01"),
        "last_price": 500.0,
        "forecasts": {
            20: {
                "probability_up": probability,
                "expected_return": expected_return,
                "quality": "WYSOKA",
                "auc": 0.70,
                "brier": 0.20,
                "lower_return": -0.05,
                "upper_return": 0.08,
            }
        },
    }
    report = build_analysis_report(result, selected_horizon=20)
    saved = watchlist_module.watch_item_from_analysis(result, report)
    current = watchlist_module.watch_item_current_snapshot(result, {"symbol": "SPY", "horizon": 20})

    assert watchlist_module.watchlist_machine_decision_state(saved) is expected_state
    assert watchlist_module.watchlist_machine_decision_state(current) is expected_state
    assert expected_state is not MachineDecisionState.INVALID


def test_invalid_direction_does_not_change_watchlist_lifecycle():
    item = canonical_snapshot(
        decision="1",
        data_as_of="2026-09-01",
        calendar_kind="CRYPTO_24_7",
        symbol="BTC-USD",
        horizon=2,
    )
    current = canonical_snapshot(symbol="BTC-USD", horizon=2)

    comparison = watchlist_module.compare_watch_item_to_current(item, current, now=date(2026, 9, 3))

    assert comparison["comparison_status"] == "DECISION_UNAVAILABLE"
    assert comparison["lifecycle"]["status"] == "HORIZON_ENDED"
    assert comparison["lifecycle"]["elapsed"] == 2


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(source, function) or ""


def test_watchlist_ui_does_not_use_raw_verdict_label_as_machine_status():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    comparison_source = _function_source(app_path, "render_watchlist_comparison")
    watchlist_source = _function_source(app_path, "render_watchlist")

    assert "verdict_label" not in comparison_source
    assert "selected.get('verdict_label')" not in watchlist_source
    assert "watchlist_machine_decision_label" in comparison_source
    assert "watchlist_machine_decision_label" in watchlist_source
