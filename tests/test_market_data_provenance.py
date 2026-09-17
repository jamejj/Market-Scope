from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta, timezone

import pytest

from market_oracle.market_data_provenance import (
    BarState,
    DataType,
    DeliveryMode,
    FeedState,
    FeedStatus,
    MarketDataProvenance,
    PriceBasis,
    RealtimeCapability,
    observed_age,
)


def test_realtime_quote_keeps_declared_delivery_separate_from_age():
    received = datetime(2026, 9, 17, 14, 2, 4, tzinfo=timezone.utc)
    observation = MarketDataProvenance(
        instrument_id="AAPL",
        source_symbol="AAPL",
        source="example-feed",
        currency="USD",
        data_type=DataType.QUOTE,
        price_basis=PriceBasis.RAW,
        delivery_mode=DeliveryMode.REALTIME,
        realtime_capability=RealtimeCapability.YES,
        source_timestamp=datetime(2026, 9, 17, 14, 2, 3, tzinfo=timezone.utc),
        retrieved_at=received,
    )

    assert observation.delivery_mode is DeliveryMode.REALTIME
    assert observation.source_timestamp == datetime(2026, 9, 17, 14, 2, 3, tzinfo=timezone.utc)
    assert observation.retrieved_at == received


def _quote(**changes):
    fields = {
        "instrument_id": "AAPL",
        "source_symbol": "AAPL",
        "source": "example-feed",
        "currency": "USD",
        "data_type": DataType.QUOTE,
        "price_basis": PriceBasis.RAW,
        "delivery_mode": DeliveryMode.DELAYED,
        "realtime_capability": RealtimeCapability.YES,
        "source_timestamp": datetime(2026, 9, 17, 13, 47, tzinfo=timezone.utc),
        "retrieved_at": datetime(2026, 9, 17, 14, 2, tzinfo=timezone.utc),
        "declared_delay_seconds": 900,
    }
    fields.update(changes)
    return MarketDataProvenance(**fields)


def test_delayed_quote_preserves_declared_delay_without_inference():
    observation = _quote()

    assert observation.delivery_mode is DeliveryMode.DELAYED
    assert observation.declared_delay_seconds == 900
    assert observation.realtime_capability is RealtimeCapability.YES
    with pytest.raises(FrozenInstanceError):
        observation.delivery_mode = DeliveryMode.REALTIME


def test_realtime_delivery_rejects_explicit_no_realtime_capability():
    with pytest.raises(ValueError):
        _quote(
            delivery_mode=DeliveryMode.REALTIME,
            realtime_capability=RealtimeCapability.NO,
            declared_delay_seconds=None,
        )


def test_realtime_delivery_allows_unknown_realtime_capability():
    observation = _quote(
        delivery_mode=DeliveryMode.REALTIME,
        realtime_capability=RealtimeCapability.UNKNOWN,
        declared_delay_seconds=None,
    )

    assert observation.delivery_mode is DeliveryMode.REALTIME


def test_realtime_daily_bar_can_be_in_progress():
    observation = _quote(
        data_type=DataType.BAR,
        interval="1d",
        delivery_mode=DeliveryMode.REALTIME,
        bar_state=BarState.IN_PROGRESS,
        session_date=date(2026, 9, 17),
        declared_delay_seconds=None,
    )

    assert observation.interval == "1d"
    assert observation.bar_state is BarState.IN_PROGRESS
    assert observation.delivery_mode is DeliveryMode.REALTIME


def test_end_of_day_bar_requires_explicit_closed_state_to_claim_closed():
    observation = _quote(
        data_type=DataType.BAR,
        interval="1d",
        delivery_mode=DeliveryMode.END_OF_DAY,
        bar_state=BarState.UNKNOWN,
        session_date=date(2026, 9, 16),
        price_basis=PriceBasis.ADJUSTED,
        declared_delay_seconds=None,
    )

    assert observation.bar_state is BarState.UNKNOWN
    assert replace(observation, bar_state=BarState.CLOSED).bar_state is BarState.CLOSED


@pytest.mark.parametrize("changes", [
    {"interval": "1d"},
    {"bar_state": BarState.UNKNOWN},
    {"data_type": DataType.BAR},
    {"data_type": DataType.BAR, "interval": "1d"},
    {"delivery_mode": DeliveryMode.END_OF_DAY},
    {"data_type": DataType.TRADE, "delivery_mode": DeliveryMode.END_OF_DAY},
])
def test_rejects_incompatible_data_type_fields(changes):
    with pytest.raises(ValueError):
        _quote(**changes)


def test_aware_timestamps_are_normalized_to_utc():
    observation = _quote(
        source_timestamp=datetime(2026, 9, 17, 15, 47, tzinfo=timezone(timedelta(hours=2))),
        retrieved_at=datetime(2026, 9, 17, 16, 2, tzinfo=timezone(timedelta(hours=2))),
    )

    assert observation.source_timestamp == datetime(2026, 9, 17, 13, 47, tzinfo=timezone.utc)
    assert observation.source_timestamp.tzinfo is timezone.utc
    assert observation.retrieved_at == datetime(2026, 9, 17, 14, 2, tzinfo=timezone.utc)
    assert observation.retrieved_at.tzinfo is timezone.utc


@pytest.mark.parametrize("changes", [
    {"retrieved_at": None},
    {"retrieved_at": datetime(2026, 9, 17, 14, 2)},
    {"retrieved_at": "2026-09-17T14:02:00Z"},
    {"source_timestamp": datetime(2026, 9, 17, 13, 47)},
    {"source_timestamp": "2026-09-17T13:47:00Z"},
])
def test_rejects_missing_naive_or_untyped_timestamps(changes):
    with pytest.raises(ValueError):
        _quote(**changes)


def test_session_date_is_a_date_not_a_synthetic_timestamp():
    observation = _quote(session_date=date(2026, 9, 17))

    assert type(observation.session_date) is date
    with pytest.raises(ValueError):
        _quote(session_date=datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc))


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf"), -float("inf"), True, "900"])
def test_rejects_invalid_declared_delay(delay):
    with pytest.raises(ValueError):
        _quote(declared_delay_seconds=delay)


def test_unknown_price_basis_and_feed_mode_remain_explicit():
    observation = _quote(
        price_basis=PriceBasis.UNKNOWN,
        delivery_mode=DeliveryMode.UNKNOWN,
        realtime_capability=RealtimeCapability.UNKNOWN,
        declared_delay_seconds=None,
        source_timestamp=None,
    )

    assert observation.price_basis is PriceBasis.UNKNOWN
    assert observation.delivery_mode is DeliveryMode.UNKNOWN
    assert observation.source_timestamp is None


def test_unavailable_feed_keeps_the_last_observation_timestamp_unchanged():
    last_observation = _quote()
    feed = FeedState(
        feed_status=FeedStatus.UNAVAILABLE,
        checked_at=datetime(2026, 9, 17, 16, 7, tzinfo=timezone(timedelta(hours=2))),
        last_observation_provenance=last_observation,
    )

    assert feed.checked_at == datetime(2026, 9, 17, 14, 7, tzinfo=timezone.utc)
    assert feed.checked_at.tzinfo is timezone.utc
    assert feed.last_observation_provenance is last_observation
    assert feed.last_observation_provenance.retrieved_at == datetime(2026, 9, 17, 14, 2, tzinfo=timezone.utc)
    with pytest.raises(FrozenInstanceError):
        feed.feed_status = FeedStatus.AVAILABLE


def test_feed_state_rejects_observation_retrieved_after_check():
    with pytest.raises(ValueError):
        FeedState(
            feed_status=FeedStatus.UNAVAILABLE,
            checked_at=datetime(2026, 9, 17, 14, 0, tzinfo=timezone.utc),
            last_observation_provenance=_quote(),
        )


def test_feed_state_allows_observation_retrieved_at_check_time():
    feed = FeedState(
        feed_status=FeedStatus.AVAILABLE,
        checked_at=datetime(2026, 9, 17, 14, 2, tzinfo=timezone.utc),
        last_observation_provenance=_quote(),
    )

    assert feed.last_observation_provenance.retrieved_at == feed.checked_at


@pytest.mark.parametrize("checked_at", [None, datetime(2026, 9, 17, 14, 7), "2026-09-17T14:07:00Z"])
def test_feed_state_rejects_missing_naive_or_untyped_check_time(checked_at):
    with pytest.raises(ValueError):
        FeedState(feed_status=FeedStatus.UNKNOWN, checked_at=checked_at)


def test_feed_state_allows_no_last_observation_without_fabricating_one():
    feed = FeedState(
        feed_status=FeedStatus.UNKNOWN,
        checked_at=datetime(2026, 9, 17, 14, 7, tzinfo=timezone.utc),
    )

    assert feed.last_observation_provenance is None


def test_observed_age_is_computed_at_read_time_without_changing_declared_mode():
    observation = _quote(delivery_mode=DeliveryMode.REALTIME, declared_delay_seconds=None)
    now = datetime(2026, 9, 17, 14, 2, 30, tzinfo=timezone.utc)

    assert observed_age(observation, now=now) == timedelta(minutes=15, seconds=30)
    assert observation.delivery_mode is DeliveryMode.REALTIME
    assert "observed_age" not in observation.__dict__


def test_observed_age_is_unknown_without_source_time_or_with_clock_skew():
    no_source_time = _quote(source_timestamp=None)
    future_source_time = _quote(source_timestamp=datetime(2026, 9, 17, 14, 10, tzinfo=timezone.utc))
    now = datetime(2026, 9, 17, 14, 5, tzinfo=timezone.utc)

    assert observed_age(no_source_time, now=now) is None
    assert observed_age(future_source_time, now=now) is None


def test_observed_age_rejects_naive_now():
    with pytest.raises(ValueError):
        observed_age(_quote(), now=datetime(2026, 9, 17, 14, 5))


@pytest.mark.parametrize("now", [0, ""])
def test_observed_age_rejects_explicit_falsy_non_datetime_now(now):
    with pytest.raises(ValueError):
        observed_age(_quote(), now=now)


@pytest.mark.parametrize("changes", [
    {"instrument_id": ""},
    {"source_symbol": "  "},
    {"source": ""},
    {"currency": ""},
    {"data_type": "QUOTE"},
    {"price_basis": "RAW"},
    {"delivery_mode": "DELAYED"},
    {"realtime_capability": "YES"},
])
def test_observation_rejects_missing_identity_or_unvalidated_enum_values(changes):
    with pytest.raises(ValueError):
        _quote(**changes)


@pytest.mark.parametrize("changes", [
    {"feed_status": "UNAVAILABLE"},
    {"last_observation_provenance": "not-an-observation"},
])
def test_feed_state_rejects_unvalidated_status_or_last_observation(changes):
    fields = {
        "feed_status": FeedStatus.UNAVAILABLE,
        "checked_at": datetime(2026, 9, 17, 14, 7, tzinfo=timezone.utc),
        "last_observation_provenance": _quote(),
    }
    fields.update(changes)
    with pytest.raises(ValueError):
        FeedState(**fields)


@pytest.mark.parametrize("changes", [
    {"venue_id": 42},
    {"calendar_id": []},
    {"exchange_timezone": object()},
    {"venue_id": "  "},
])
def test_observation_rejects_invalid_optional_identity_metadata(changes):
    with pytest.raises(ValueError):
        _quote(**changes)


def test_huge_declared_delay_fails_with_contract_error():
    with pytest.raises(ValueError):
        _quote(declared_delay_seconds=10 ** 1000)
