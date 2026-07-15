from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalInputs:
    """Auditable input bundle for the final directional gate.

    The app deliberately keeps model estimation separate from the final decision.
    Production, backtest and journal-facing code should pass this same object so
    differences in results come from data and timing, not hidden defaults.
    """

    probability: float
    expected_return: float
    quality: str = "WYSOKA"
    auc: float | None = None
    brier: float | None = None
    source: str = "ML"


@dataclass(frozen=True)
class SignalVerdict:
    """Final gate output with an audit reason."""

    decision: int
    reason: str
    label: str


def signal_inputs_from_forecast(forecast: dict, source: str = "ML") -> SignalInputs:
    """Create the shared decision payload from a production forecast dict."""
    return SignalInputs(
        probability=float(forecast.get("probability_up", 0.5)),
        expected_return=float(forecast.get("expected_return", 0.0)),
        quality=str(forecast.get("quality", "NISKA — BRAK PRZEWAGI")),
        auc=None if forecast.get("auc") is None else float(forecast["auc"]),
        brier=None if forecast.get("brier") is None else float(forecast["brier"]),
        source=source,
    )


def signal_verdict(inputs: SignalInputs, threshold: float = 0.56, min_expected_return: float = 0.0) -> SignalVerdict:
    """Shared directional decision used by production labels and validation.

    Returns 1 for long, -1 for short and 0 for no trade. Weak model quality is
    deliberately not promoted to a directional signal.
    """
    probability = max(0.0, min(1.0, float(inputs.probability)))
    expected_return = float(inputs.expected_return)
    quality = str(inputs.quality)
    if quality.startswith("NISKA"):
        return SignalVerdict(0, "LOW_QUALITY", "BRAK SYGNAŁU")
    if probability >= threshold:
        if expected_return < 0:
            return SignalVerdict(0, "EXPECTED_RETURN_CONFLICT", "OBSERWUJ")
        if expected_return < min_expected_return:
            return SignalVerdict(0, "EXPECTED_RETURN_TOO_SMALL", "OBSERWUJ")
        return SignalVerdict(1, "LONG_CONFIRMED", "LONG")
    if probability <= 1 - threshold:
        if expected_return > 0:
            return SignalVerdict(0, "EXPECTED_RETURN_CONFLICT", "OBSERWUJ")
        if abs(expected_return) < min_expected_return:
            return SignalVerdict(0, "EXPECTED_RETURN_TOO_SMALL", "OBSERWUJ")
        return SignalVerdict(-1, "SHORT_CONFIRMED", "SHORT")
    return SignalVerdict(0, "PROBABILITY_INSIDE_BAND", "OBSERWUJ")


def signal_decision(inputs: SignalInputs, threshold: float = 0.56, min_expected_return: float = 0.0) -> int:
    return signal_verdict(inputs, threshold, min_expected_return).decision
