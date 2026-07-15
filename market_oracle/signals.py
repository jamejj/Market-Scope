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


def signal_decision(inputs: SignalInputs, threshold: float = 0.56) -> int:
    """Shared directional decision used by production labels and validation.

    Returns 1 for long, -1 for short and 0 for no trade. Weak model quality is
    deliberately not promoted to a directional signal.
    """
    probability = max(0.0, min(1.0, float(inputs.probability)))
    expected_return = float(inputs.expected_return)
    quality = str(inputs.quality)
    if quality.startswith("NISKA"):
        return 0
    if probability >= threshold and expected_return >= 0:
        return 1
    if probability <= 1 - threshold and expected_return <= 0:
        return -1
    return 0
