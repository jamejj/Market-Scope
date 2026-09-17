"""Provider-neutral provenance for market-data observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from numbers import Real


class DataType(str, Enum):
    TRADE = "TRADE"
    QUOTE = "QUOTE"
    BAR = "BAR"


class PriceBasis(str, Enum):
    RAW = "RAW"
    ADJUSTED = "ADJUSTED"
    UNKNOWN = "UNKNOWN"


class DeliveryMode(str, Enum):
    REALTIME = "REALTIME"
    DELAYED = "DELAYED"
    END_OF_DAY = "END_OF_DAY"
    UNKNOWN = "UNKNOWN"


class RealtimeCapability(str, Enum):
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


class BarState(str, Enum):
    CLOSED = "CLOSED"
    IN_PROGRESS = "IN_PROGRESS"
    UNKNOWN = "UNKNOWN"


class FeedStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


def _utc_datetime(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware datetime")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class MarketDataProvenance:
    instrument_id: str
    source_symbol: str
    source: str
    currency: str
    data_type: DataType
    price_basis: PriceBasis
    delivery_mode: DeliveryMode
    realtime_capability: RealtimeCapability
    retrieved_at: datetime
    source_timestamp: datetime | None = None
    interval: str | None = None
    bar_state: BarState | None = None
    session_date: date | None = None
    declared_delay_seconds: float | None = None
    venue_id: str | None = None
    calendar_id: str | None = None
    exchange_timezone: str | None = None

    def __post_init__(self) -> None:
        for field in ("instrument_id", "source_symbol", "source", "currency"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be non-empty text")
        for field in ("venue_id", "calendar_id", "exchange_timezone"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field} must be non-empty text when present")
        for field, enum_type in (
            ("data_type", DataType),
            ("price_basis", PriceBasis),
            ("delivery_mode", DeliveryMode),
            ("realtime_capability", RealtimeCapability),
        ):
            if type(getattr(self, field)) is not enum_type:
                raise ValueError(f"{field} must be {enum_type.__name__}")
        if self.delivery_mode is DeliveryMode.REALTIME and self.realtime_capability is RealtimeCapability.NO:
            raise ValueError("REALTIME delivery conflicts with NO realtime capability")
        if self.data_type is DataType.BAR:
            if not isinstance(self.interval, str) or not self.interval.strip():
                raise ValueError("BAR requires interval")
            if not isinstance(self.bar_state, BarState):
                raise ValueError("BAR requires bar_state")
        elif self.interval is not None or self.bar_state is not None:
            raise ValueError("Only BAR may have interval or bar_state")
        if self.delivery_mode is DeliveryMode.END_OF_DAY and self.data_type is not DataType.BAR:
            raise ValueError("END_OF_DAY requires BAR")
        object.__setattr__(self, "retrieved_at", _utc_datetime(self.retrieved_at, "retrieved_at"))
        if self.source_timestamp is not None:
            object.__setattr__(
                self, "source_timestamp", _utc_datetime(self.source_timestamp, "source_timestamp")
            )
        if self.session_date is not None and type(self.session_date) is not date:
            raise ValueError("session_date must be a date")
        if self.declared_delay_seconds is not None:
            delay = self.declared_delay_seconds
            if isinstance(delay, bool) or not isinstance(delay, Real):
                raise ValueError("declared_delay_seconds must be finite and non-negative")
            try:
                finite_delay = math.isfinite(float(delay))
            except OverflowError as exc:
                raise ValueError("declared_delay_seconds must be finite and non-negative") from exc
            if not finite_delay or delay < 0:
                raise ValueError("declared_delay_seconds must be finite and non-negative")


@dataclass(frozen=True)
class FeedState:
    feed_status: FeedStatus
    checked_at: datetime
    last_observation_provenance: MarketDataProvenance | None = None

    def __post_init__(self) -> None:
        if type(self.feed_status) is not FeedStatus:
            raise ValueError("feed_status must be FeedStatus")
        if self.last_observation_provenance is not None and not isinstance(
            self.last_observation_provenance, MarketDataProvenance
        ):
            raise ValueError("last_observation_provenance must be MarketDataProvenance")
        object.__setattr__(self, "checked_at", _utc_datetime(self.checked_at, "checked_at"))
        if (
            self.last_observation_provenance is not None
            and self.last_observation_provenance.retrieved_at > self.checked_at
        ):
            raise ValueError("last observation was retrieved after feed check")


def observed_age(provenance: MarketDataProvenance, *, now: datetime | None = None) -> timedelta | None:
    """Age of a source event at read time, never a feed-mode inference."""
    checked_at = _utc_datetime(datetime.now(timezone.utc) if now is None else now, "now")
    if provenance.source_timestamp is None:
        return None
    age = checked_at - provenance.source_timestamp
    return age if age >= timedelta(0) else None
