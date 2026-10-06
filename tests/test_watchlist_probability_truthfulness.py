from __future__ import annotations

import ast
from copy import deepcopy
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_oracle import watchlist
from market_oracle.product_verdict import finite_probability, strict_finite_real


INVALID = [True, False, "0.7", "1", 1.1, -0.1, np.nan, np.inf, -np.inf, None, pd.NA]


def observation(probability=0.6):
    return {
        "id": "watch-spy", "symbol": "SPY", "horizon": 20,
        "created_at": "2026-10-01T12:00:00+00:00", "status": "ACTIVE",
        "calendar_kind": "NYSE", "data_as_of": "2026-10-01",
        "verdict_decision": 1, "verdict": "LONG_CONFIRMED",
        "probability_up": probability, "expected_return": 0.03,
        "quality": "WYSOKA", "available": True,
    }


class UI:
    def __init__(self):
        self.markdown_values = []
        self.session_state = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def markdown(self, value, **kwargs):
        self.markdown_values.append(value)

    def tabs(self, labels):
        return [self for _ in labels]

    def columns(self, count):
        return [self for _ in range(count)]

    def selectbox(self, label, options, **kwargs):
        return next(iter(options))

    def button(self, *args, **kwargs):
        return False

    def header(self, *args, **kwargs):
        pass

    write = caption = info = dataframe = header


def app_functions(ui, items):
    path = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "clean_text", "short_datetime", "value_pct", "signed_pct", "signed_pp",
        "watch_age_days", "watch_status_label", "watchlist_dataframe",
        "render_watchlist_comparison", "render_watchlist",
    }
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "html": html, "math": math, "pd": pd, "st": ui,
        "pct": lambda value: f"{value:.1%}",
        "finite_probability": finite_probability,
        "strict_finite_real": strict_finite_real,
        "safe_load_watchlist": lambda: (items, None),
        "watchlist_summary": watchlist.watchlist_summary,
        "watchlist_machine_decision_label": watchlist.watchlist_machine_decision_label,
        "watchlist_analysis_matches_selection": lambda *args: False,
        "radar_data_date": lambda value: str(value) if value else None,
        "years": 5,
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    # Date labels are unrelated to probability; avoid depending on wall-clock time.
    namespace["watch_status_label"] = lambda item: "Aktywna"
    return namespace


@pytest.mark.parametrize("probability", INVALID)
@pytest.mark.parametrize("side", ["historical", "current"])
def test_invalid_probability_cannot_create_delta_or_change_reason(probability, side):
    item = observation(probability if side == "historical" else 0.6)
    current = observation(probability if side == "current" else 0.6)
    result = watchlist.compare_watch_item_to_current(item, current, now="2026-10-06")
    assert result["delta_probability"] is None
    assert not any("P(wzrost) zmieniło się" in reason for reason in result["reasons"])
    assert result["comparison_status"] == "STILL_CONFIRMED"


@pytest.mark.parametrize("side", ["historical", "current"])
def test_missing_probability_cannot_create_delta(side):
    item, current = observation(), observation()
    (item if side == "historical" else current).pop("probability_up")
    result = watchlist.compare_watch_item_to_current(item, current, now="2026-10-06")
    assert result["delta_probability"] is None
    assert not any("P(wzrost) zmieniło się" in reason for reason in result["reasons"])


@pytest.mark.parametrize("probability", INVALID)
def test_invalid_history_is_unavailable_in_dataframe_and_rendered_cards(probability):
    item = observation(probability)
    ui = UI()
    functions = app_functions(ui, [item])
    frame = functions["watchlist_dataframe"]([item])
    assert pd.isna(frame.loc[0, "P(wzrost)"])
    rendered = frame.style.format({"P(wzrost)": "{:.1%}"}, na_rep="—").to_html()
    assert "—" in rendered

    functions["render_watchlist"]()
    assert "<small>P(wzrost)</small><strong>—</strong>" in "".join(ui.markdown_values)


@pytest.mark.parametrize("probability", INVALID)
@pytest.mark.parametrize("side", ["historical", "current"])
def test_comparison_cards_render_invalid_probability_as_unavailable(probability, side):
    item = observation(probability if side == "historical" else 0.6)
    current = observation(probability if side == "current" else 0.6)
    ui = UI()
    functions = app_functions(ui, [item])
    comparison = watchlist.compare_watch_item_to_current(item, current, now="2026-10-06")
    functions["render_watchlist_comparison"](item, current, comparison)
    rendered = ui.markdown_values[-1]
    card = rendered.split("<small>Wtedy</small>" if side == "historical" else "<small>Teraz</small>")[1].split("</div>")[0]
    assert "P(wzrost): —" in card


@pytest.mark.parametrize("probability", [0, 1, 0.7, np.float64(0.7)])
def test_valid_probability_survives_dataframe_cards_and_delta(probability):
    item, current = observation(probability), observation(0.6)
    ui = UI()
    functions = app_functions(ui, [item])
    assert functions["watchlist_dataframe"]([item]).loc[0, "P(wzrost)"] == probability
    functions["render_watchlist"]()
    assert f"<small>P(wzrost)</small><strong>{probability:.1%}</strong>" in "".join(ui.markdown_values)
    comparison = watchlist.compare_watch_item_to_current(item, current, now="2026-10-06")
    assert comparison["delta_probability"] == pytest.approx(0.6 - probability)


def test_consumers_do_not_mutate_historical_records():
    item, current = observation("0.7"), observation(0.6)
    before_item, before_current = deepcopy(item), deepcopy(current)
    ui = UI()
    functions = app_functions(ui, [item])
    functions["watchlist_dataframe"]([item])
    functions["render_watchlist"]()
    comparison = watchlist.compare_watch_item_to_current(item, current, now="2026-10-06")
    functions["render_watchlist_comparison"](item, current, comparison)
    assert item == before_item
    assert current == before_current
