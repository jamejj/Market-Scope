from __future__ import annotations

import math
from typing import Any

from .signals import DEFAULT_SIGNAL_THRESHOLD, SignalInputs, SignalVerdict, signal_verdict


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def has_complete_expected_return(forecast: dict) -> bool:
    """Return whether the product can safely evaluate the forecast direction."""
    return finite_float(forecast.get("expected_return")) is not None


def product_forecast_verdict(forecast: dict, *, source: str) -> SignalVerdict:
    """Apply the shared gate, failing closed when expected return is incomplete.

    A real finite zero remains a valid input. Missing and non-finite values never
    impersonate zero and therefore cannot produce a directional product verdict.
    """
    expected_return = finite_float(forecast.get("expected_return"))
    if expected_return is None:
        return SignalVerdict(0, "INCOMPLETE_FORECAST", "OBSERWUJ")

    probability = finite_float(forecast.get("probability_up"))
    return signal_verdict(
        SignalInputs(
            probability=0.5 if probability is None else probability,
            expected_return=expected_return,
            quality=str(forecast.get("quality") or "NISKA — BRAK PRZEWAGI"),
            auc=finite_float(forecast.get("auc")),
            brier=finite_float(forecast.get("brier")),
            source=source,
        ),
        threshold=DEFAULT_SIGNAL_THRESHOLD,
    )
