from __future__ import annotations

import ast
import copy
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_oracle import watchlist as watchlist_module
from market_oracle.presentation import (
    build_analysis_report,
    build_start_guidance,
    radar_display_frame,
)
from market_oracle.product_verdict import strict_finite_real
from market_oracle.watchlist import (
    compare_watch_item_to_current,
    load_watchlist,
    watch_item_current_snapshot,
    watch_item_from_analysis,
)


INVALID_EXPECTED_RETURNS = [
    True,
    "0.03",
    "-0.03",
    "garbage",
    None,
    pd.NA,
    float("nan"),
    float("inf"),
    float("-inf"),
]


def analysis_result(expected_return: object = 0.03, *, symbol: str = "SPY") -> dict:
    return {
        "symbol": symbol,
        "last_date": pd.Timestamp("2026-09-29"),
        "last_price": 500.0,
        "benchmark": "^GSPC",
        "technical": {
            "return_1d": 0.01,
            "return_5d": 0.03,
            "return_20d": 0.10,
            "rsi_14": 62.0,
            "above_sma_50": True,
            "above_sma_200": True,
        },
        "risk": {"max_drawdown": -0.18},
        "forecasts": {
            20: {
                "probability_up": 0.61,
                "expected_return": expected_return,
                "lower_return": -0.06,
                "upper_return": 0.08,
                "auc": 0.66,
                "brier": 0.21,
                "quality": "WYSOKA",
            }
        },
    }


def canonical_watch_snapshot(expected_return: object) -> dict:
    return {
        "available": True,
        "symbol": "SPY",
        "horizon": 20,
        "created_at": "2026-09-29T12:00:00+00:00",
        "data_as_of": "2026-09-29",
        "verdict_decision": 1,
        "verdict": "LONG_CONFIRMED",
        "verdict_label": "LONG",
        "probability_up": 0.61,
        "expected_return": expected_return,
        "quality": "WYSOKA",
    }


def radar_row(symbol: str, expected_return: object, *, action: str | None = None) -> dict:
    row = {
        "Symbol": symbol,
        "Tryb analizy": "FAST",
        "Horyzont": 20,
        "P(wzrost)": 0.50,
        "Oczekiwany ruch": expected_return,
        "Jakość modelu": "FAST — BEZ ML",
        "Zwrot 1d": 0.0,
        "Zwrot 5d": 0.0,
        "Zwrot 20d": 0.0,
        "Zmienność roczna": 0.0,
        "RSI 14": 50.0,
        "Dolna granica 90%": -0.05,
        "Górna granica 90%": 0.0,
        "AUC walidacji": 0.5,
        "Brier": 0.27,
        "Max drawdown": -0.10,
        "Teza radaru": "kontrolowany test",
        "Data": "2026-09-29",
        "Cena": 100.0,
    }
    if action is not None:
        row["Akcja radaru"] = action
    return row


class MarkdownCapture:
    def __init__(self) -> None:
        self.values: list[str] = []

    def markdown(self, value: str, **_: object) -> None:
        self.values.append(value)


def load_app_functions(*names: str, st_capture: MarkdownCapture | None = None) -> dict[str, object]:
    source_path = Path(__file__).resolve().parents[1] / "app.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    wanted = set(names)
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in functions} == wanted
    namespace = {
        "html": html,
        "math": math,
        "pd": pd,
        "strict_finite_real": strict_finite_real,
        "pct": lambda value: f"{value:.1%}",
        "display_radar_direction": lambda row: "FAST discovery",
        "radar_data_date": lambda value: str(value)[:10] if value else None,
        "watchlist_machine_decision_label": watchlist_module.watchlist_machine_decision_label,
        "st": st_capture,
    }
    future_annotations = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations")],
        level=0,
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future_annotations, *functions], type_ignores=[])
    )
    exec(compile(module, str(source_path), "exec"), namespace)
    return {name: namespace[name] for name in names}


@pytest.mark.parametrize("expected_return", INVALID_EXPECTED_RETURNS)
def test_full_analysis_never_formats_invalid_expected_return_as_number(expected_return):
    report = build_analysis_report(analysis_result(expected_return), selected_horizon=20)
    visible = " ".join(
        [
            *report["evidence"],
            *report["counterpoints"],
            *(" ".join(card) for card in report["cards"]),
            *(str(card) for card in report["horizon_cards"]),
        ]
    )

    assert report["cards"][3][1] == "—"
    assert report["horizon_cards"][0]["expected"] == "—"
    assert "oczekiwany ruch —" in report["evidence"][1]
    assert "'expected': '—'" in str(report["horizon_cards"])
    assert "Oczekiwany ruch nie jest dodatni" not in visible
    assert "Oczekiwany ruch nie jest ujemny" not in visible


@pytest.mark.parametrize(
    ("expected_return", "formatted"),
    [(0, "+0.0%"), (1, "+100.0%"), (-1, "-100.0%"), (np.float64(0.025), "+2.5%")],
)
def test_full_analysis_preserves_valid_real_expected_return(expected_return, formatted):
    report = build_analysis_report(analysis_result(expected_return), selected_horizon=20)

    assert report["cards"][3][1] == formatted
    assert report["horizon_cards"][0]["expected"] == formatted


def test_primary_horizon_tie_break_ignores_coerced_invalid_expected_return():
    result = analysis_result()
    base = result["forecasts"][20]
    result["forecasts"] = {
        5: {**base, "expected_return": True},
        20: {**base, "expected_return": "0.99"},
    }

    report = build_analysis_report(result)

    assert report["primary_horizon"] == 20
    assert report["verdict"]["reason"] == "INCOMPLETE_FORECAST"


@pytest.mark.parametrize("expected_return", INVALID_EXPECTED_RETURNS)
def test_watchlist_new_and_current_snapshots_store_invalid_expected_return_as_none(expected_return):
    result = analysis_result(expected_return)
    report = build_analysis_report(result, selected_horizon=20)

    saved = watch_item_from_analysis(result, report)
    current = watch_item_current_snapshot(result, {"symbol": "SPY", "horizon": 20})

    assert saved["expected_return"] is None
    assert current["expected_return"] is None


@pytest.mark.parametrize("invalid_side", ["saved", "current"])
def test_watchlist_expected_return_delta_is_unavailable_if_either_side_is_invalid(invalid_side):
    saved = canonical_watch_snapshot(0.01)
    current = canonical_watch_snapshot(0.03)
    if invalid_side == "saved":
        saved["expected_return"] = "0.01"
    else:
        current["expected_return"] = True

    comparison = compare_watch_item_to_current(saved, current, now="2026-09-30")

    assert comparison["delta_expected_return"] is None
    assert all("Oczekiwany ruch zmienił się" not in reason for reason in comparison["reasons"])


def test_legacy_watchlist_keeps_malformed_optional_expected_return_but_ui_marks_it_unavailable(tmp_path):
    path = tmp_path / "watchlist.json"
    path.write_text(
        '[{"id":"legacy","symbol":"SPY","horizon":20,'
        '"created_at":"2026-09-29T12:00:00+00:00","status":"ACTIVE",'
        '"expected_return":"0.03"}]',
        encoding="utf-8",
    )
    loaded = load_watchlist(path)
    functions = load_app_functions(
        "short_datetime",
        "watch_age_days",
        "watch_status_label",
        "watchlist_dataframe",
    )

    table = functions["watchlist_dataframe"](loaded)

    assert loaded[0]["expected_return"] == "0.03"
    assert pd.isna(table.loc[0, "Oczekiwany ruch"])
    assert "—" in table.style.format({"Oczekiwany ruch": "{:+.1%}"}, na_rep="—").to_html()


def test_watchlist_comparison_card_never_renders_bool_expected_return_as_percentage():
    capture = MarkdownCapture()
    functions = load_app_functions(
        "clean_text",
        "value_pct",
        "signed_pct",
        "signed_pp",
        "render_watchlist_comparison",
        st_capture=capture,
    )
    saved = canonical_watch_snapshot(True)
    current = canonical_watch_snapshot("0.03")
    comparison = compare_watch_item_to_current(saved, current, now="2026-09-30")

    functions["render_watchlist_comparison"](saved, current, comparison)
    rendered = " ".join(capture.values)

    assert "+100.0%" not in rendered
    assert "+3.0%" not in rendered
    assert rendered.count("ruch —") == 2


@pytest.mark.parametrize(
    "expected_return",
    [True, "0.03", "-0.03", "garbage", None, pd.NA, float("nan"), float("inf"), float("-inf")],
)
def test_radar_sanitizes_invalid_expected_return_before_scores_actions_and_display(expected_return):
    functions = load_app_functions(
        "signed_pct",
        "with_signal_display_columns",
        "radar_export_frame",
        "_ensure_radar_columns",
    )
    raw = pd.DataFrame([radar_row("FAST_BAD", expected_return)])
    original = raw.copy(deep=True)

    scored = functions["_ensure_radar_columns"](raw)
    displayed = functions["with_signal_display_columns"](scored)
    exported = functions["radar_export_frame"](raw, {"coverage_status": "complete", "coverage": {}})
    presentation = radar_display_frame(raw)

    pd.testing.assert_frame_equal(raw, original)
    assert pd.isna(scored.loc[0, "Oczekiwany ruch"])
    assert scored.loc[0, "Akcja radaru"] == "NEUTRALNIE"
    assert displayed.loc[0, "Ruch / impet"] == "ruch FAST: —"
    assert pd.isna(exported.loc[0, "Oczekiwany ruch"])
    assert pd.isna(presentation.loc[0, "Oczekiwany ruch"])


@pytest.mark.parametrize("expected_return", [0, 1, -1, np.float64(0.03)])
def test_radar_preserves_valid_real_expected_return(expected_return):
    functions = load_app_functions("_ensure_radar_columns")
    raw = pd.DataFrame([radar_row("FAST_VALID", expected_return)])

    scored = functions["_ensure_radar_columns"](raw)

    assert scored.loc[0, "Oczekiwany ruch"] == float(expected_return)


def test_radar_invalid_expected_return_does_not_mutate_source_snapshot_or_export_value():
    functions = load_app_functions("radar_export_frame", "_ensure_radar_columns")
    snapshot = {
        "coverage_status": "complete",
        "coverage": {},
        "records": [radar_row("FAST_BAD", "0.03")],
    }
    original = copy.deepcopy(snapshot)
    raw = pd.DataFrame(snapshot["records"])

    scored = functions["_ensure_radar_columns"](raw)
    exported = functions["radar_export_frame"](scored, snapshot)

    assert snapshot == original
    assert snapshot["records"][0]["Oczekiwany ruch"] == "0.03"
    assert pd.isna(exported.loc[0, "Oczekiwany ruch"])


def test_start_risk_leader_ignores_invalid_expected_return_tie_break_and_copy():
    invalid = radar_row("INVALID", "0.99", action="RYZYKO / UNIKAJ")
    valid = radar_row("VALID", 0.02, action="RYZYKO / UNIKAJ")
    snapshot = {
        "status": "complete",
        "updated_at": "2026-09-30T08:00:00+00:00",
        "records": [invalid, valid],
    }

    guidance = build_start_guidance(
        snapshot=snapshot,
        cockpit={},
        automation={},
        proof_state={"label": "OK", "klass": "", "detail": "healthy"},
    )
    card = next(card for card in guidance["cards"] if card["id"] == "risk_alert")

    assert card["symbol"] == "VALID"
    assert "ruch/impet +2.0%" in card["body"]
    assert "+99.0%" not in card["body"]


@pytest.mark.parametrize(
    "expected_return",
    [True, "0.03", "-0.03", "garbage", None, pd.NA, float("nan"), float("inf"), float("-inf")],
)
def test_start_risk_card_displays_invalid_expected_return_as_unavailable(expected_return):
    row = radar_row("INVALID", expected_return, action="RYZYKO / UNIKAJ")
    guidance = build_start_guidance(
        snapshot={
            "status": "complete",
            "updated_at": "2026-09-30T08:00:00+00:00",
            "records": [row],
        },
        cockpit={},
        automation={},
        proof_state={"label": "OK", "klass": "", "detail": "healthy"},
    )
    card = next(card for card in guidance["cards"] if card["id"] == "risk_alert")

    assert "ruch/impet —" in card["body"]
    assert "+100.0%" not in card["body"]
    assert "+3.0%" not in card["body"]
