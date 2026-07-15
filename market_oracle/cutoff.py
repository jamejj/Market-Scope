from __future__ import annotations


def available_label_end(signal_position: int, horizon: int) -> int:
    """Exclusive end index for labels known after close at signal_position.

    A row at index t has target Close[t+horizon] / Close[t]. After close at
    signal_position, the newest fully known label is therefore
    signal_position - horizon. The returned value is suitable for iloc slicing.
    """
    return max(0, int(signal_position) - int(horizon) + 1)


def first_signal_position(min_train: int, horizon: int) -> int:
    """First signal index that leaves min_train labeled observations available."""
    return int(min_train) + int(horizon) - 1
