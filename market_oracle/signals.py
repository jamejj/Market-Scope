from __future__ import annotations


def signal_decision(probability: float, expected_return: float = 0.0, quality: str = "WYSOKA", threshold: float = 0.56) -> int:
    """Shared directional decision used by production labels and validation.

    Returns 1 for long, -1 for short and 0 for no trade. Weak model quality is
    deliberately not promoted to a directional signal.
    """
    if quality.startswith("NISKA"):
        return 0
    if probability >= threshold and expected_return >= 0:
        return 1
    if probability <= 1 - threshold and expected_return <= 0:
        return -1
    return 0
