from __future__ import annotations

import math
from enum import Enum
from numbers import Real
from typing import Any

from .signals import DEFAULT_SIGNAL_THRESHOLD, SignalInputs, SignalVerdict, signal_verdict


class MachineDecisionState(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    INVALID = "INVALID"


NON_DIRECTIONAL_REASONS = frozenset({
    "EXPECTED_RETURN_CONFLICT",
    "EXPECTED_RETURN_TOO_SMALL",
    "INCOMPLETE_FORECAST",
    "LOW_QUALITY",
    "PROBABILITY_INSIDE_BAND",
})


def _machine_decision_state(decision: int, reason: Any) -> MachineDecisionState:
    if not isinstance(reason, str):
        return MachineDecisionState.INVALID
    if decision == 1 and reason == "LONG_CONFIRMED":
        return MachineDecisionState.LONG
    if decision == -1 and reason == "SHORT_CONFIRMED":
        return MachineDecisionState.SHORT
    if decision == 0 and reason in NON_DIRECTIONAL_REASONS:
        return MachineDecisionState.NEUTRAL
    return MachineDecisionState.INVALID


def persisted_machine_decision_state(row: Any) -> MachineDecisionState:
    """Classify a persisted ML or legacy mode-less record without coercion."""
    if not hasattr(row, "get"):
        return MachineDecisionState.INVALID
    if "Tryb analizy" in row and row.get("Tryb analizy") != "ML":
        return MachineDecisionState.INVALID
    decision = row.get("Decision")
    if type(decision) is not int or decision not in {-1, 0, 1}:
        return MachineDecisionState.INVALID
    return _machine_decision_state(decision, row.get("DecisionReason"))


def dataframe_machine_decision_state(row: Any) -> MachineDecisionState:
    """Classify an ML row after pandas may losslessly coerce ints to floats."""
    if not hasattr(row, "get") or row.get("Tryb analizy") != "ML":
        return MachineDecisionState.INVALID
    decision = row.get("Decision")
    if isinstance(decision, bool) or not isinstance(decision, Real):
        return MachineDecisionState.INVALID
    number = float(decision)
    if not math.isfinite(number) or number not in {-1.0, 0.0, 1.0}:
        return MachineDecisionState.INVALID
    return _machine_decision_state(int(number), row.get("DecisionReason"))


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def has_complete_expected_return(forecast: dict) -> bool:
    """Return whether the product can safely evaluate the forecast direction."""
    return finite_float(forecast.get("expected_return")) is not None


def finite_probability(value: Any) -> float | None:
    """Return a valid probability without changing generic numeric semantics."""
    probability = finite_float(value)
    if probability is None or not 0.0 <= probability <= 1.0:
        return None
    return probability


def forecast_integrity_issue(forecast: dict) -> str | None:
    """Identify the product input that prevents a directional verdict."""
    if not has_complete_expected_return(forecast):
        return "EXPECTED_RETURN"
    if finite_probability(forecast.get("probability_up")) is None:
        return "PROBABILITY"
    return None


def has_complete_product_forecast(forecast: dict) -> bool:
    return forecast_integrity_issue(forecast) is None


def product_forecast_verdict(forecast: dict, *, source: str) -> SignalVerdict:
    """Apply the shared gate, failing closed when product inputs are incomplete.

    Real finite boundary values remain valid inputs. Missing, non-finite and
    out-of-range values never impersonate neutral values and therefore cannot
    produce a directional product verdict.
    """
    expected_return = finite_float(forecast.get("expected_return"))
    probability = finite_probability(forecast.get("probability_up"))
    if expected_return is None or probability is None:
        return SignalVerdict(0, "INCOMPLETE_FORECAST", "OBSERWUJ")

    return signal_verdict(
        SignalInputs(
            probability=probability,
            expected_return=expected_return,
            quality=str(forecast.get("quality") or "NISKA — BRAK PRZEWAGI"),
            auc=finite_float(forecast.get("auc")),
            brier=finite_float(forecast.get("brier")),
            source=source,
        ),
        threshold=DEFAULT_SIGNAL_THRESHOLD,
    )
