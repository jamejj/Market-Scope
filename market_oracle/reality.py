from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


DATE_COLUMNS = ("Date", "EntryDate", "ExitDate")


@dataclass(frozen=True)
class RealityConfig:
    """Conservative execution audit for already-generated validation records."""

    horizons: tuple[int, ...] | None = None
    symbols: tuple[str, ...] | None = None
    markets: tuple[str, ...] | None = None
    one_position_per_symbol: bool = True
    allow_same_day_reentry: bool = False
    max_positions: int | None = None
    benchmark_symbol: str | None = "SPY"
    annualization_days: int = 365
    bootstrap_samples: int = 2000
    random_state: int = 42


def normalize_records(records: pd.DataFrame) -> pd.DataFrame:
    """Return records with typed dates/numbers expected by the Reality Lab."""
    frame = records.copy()
    for column in DATE_COLUMNS:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column]).dt.tz_localize(None).dt.normalize()
    numeric_columns = (
        "Horizon", "Fold", "Position", "EntryPrice", "ExitPrice", "Return",
        "RoundTripCost", "BuyHoldReturn", "ActualUp", "ExecutionUp",
        "Probability", "ExpectedReturn", "ValidationAUC", "ValidationBrier",
    )
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def filter_records(records: pd.DataFrame, config: RealityConfig) -> pd.DataFrame:
    frame = normalize_records(records)
    if config.horizons is not None:
        frame = frame[frame["Horizon"].isin(config.horizons)]
    if config.symbols is not None:
        frame = frame[frame["Symbol"].isin(config.symbols)]
    if config.markets is not None:
        frame = frame[frame["Market"].isin(config.markets)]
    return frame.copy()


def select_non_overlapping_trades(records: pd.DataFrame, config: RealityConfig = RealityConfig()) -> pd.DataFrame:
    """Select first valid signal and ignore overlap until the position closes.

    This is intentionally stricter than the raw validation count: repeated
    signals during one market move do not become multiple independent trades.
    """
    frame = filter_records(records, config)
    active = frame[frame["Position"].fillna(0) != 0].copy()
    if active.empty:
        return active.assign(RealityTradeId=pd.Series(dtype=int), RejectedByReality=pd.Series(dtype=object))

    active = active.sort_values(
        ["EntryDate", "Date", "Symbol", "Horizon", "Probability", "ExpectedReturn"],
        ascending=[True, True, True, True, False, False],
    ).reset_index(drop=True)
    selected_rows: list[pd.Series] = []
    open_by_symbol: dict[str, pd.Timestamp] = {}
    open_portfolio: list[tuple[str, pd.Timestamp]] = []

    def closed_before(exit_date: pd.Timestamp, entry_date: pd.Timestamp) -> bool:
        return exit_date <= entry_date if config.allow_same_day_reentry else exit_date < entry_date

    for _, row in active.iterrows():
        symbol = str(row["Symbol"])
        entry = pd.Timestamp(row["EntryDate"])
        exit_date = pd.Timestamp(row["ExitDate"])
        open_portfolio = [
            (open_symbol, open_exit)
            for open_symbol, open_exit in open_portfolio
            if not closed_before(open_exit, entry)
        ]
        if config.one_position_per_symbol and symbol in open_by_symbol:
            last_exit = open_by_symbol[symbol]
            if not closed_before(last_exit, entry):
                continue
        if config.max_positions is not None and len(open_portfolio) >= config.max_positions:
            continue
        selected_rows.append(row)
        open_by_symbol[symbol] = exit_date
        open_portfolio.append((symbol, exit_date))

    if not selected_rows:
        return active.iloc[0:0].assign(RealityTradeId=pd.Series(dtype=int), RejectedByReality=pd.Series(dtype=object))
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected["RealityTradeId"] = np.arange(1, len(selected) + 1)
    selected["RejectedByReality"] = False
    return selected


def _open_series(history: pd.DataFrame) -> pd.Series:
    series = history["Open"].astype(float).copy()
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    return series[~series.index.duplicated(keep="last")].sort_index()


def _nearest_open_slice(history: pd.DataFrame, entry: pd.Timestamp, exit_date: pd.Timestamp) -> pd.Series:
    opens = _open_series(history)
    window = opens.loc[(opens.index >= entry) & (opens.index <= exit_date)]
    return window.dropna()


def trade_daily_returns(trade: pd.Series, history: pd.DataFrame) -> pd.DataFrame:
    """Open-to-open daily path for one selected trade, with cost charged at entry."""
    entry = pd.Timestamp(trade["EntryDate"]).normalize()
    exit_date = pd.Timestamp(trade["ExitDate"]).normalize()
    position = int(trade["Position"])
    cost = abs(position) * float(trade.get("RoundTripCost", 0.0) or 0.0)
    opens = _nearest_open_slice(history, entry, exit_date)
    rows: list[dict] = [{
        "Date": entry,
        "RealityTradeId": int(trade["RealityTradeId"]),
        "Symbol": str(trade["Symbol"]),
        "Horizon": int(trade["Horizon"]),
        "Fold": int(trade["Fold"]),
        "PositionReturn": -cost,
        "UnderlyingLongReturn": -cost,
        "Entry": 1,
        "Active": 1,
    }]
    if len(opens) >= 2:
        returns = opens.pct_change().dropna()
        for date, value in returns.items():
            rows.append({
                "Date": pd.Timestamp(date).normalize(),
                "RealityTradeId": int(trade["RealityTradeId"]),
                "Symbol": str(trade["Symbol"]),
                "Horizon": int(trade["Horizon"]),
                "Fold": int(trade["Fold"]),
                "PositionReturn": position * float(value),
                "UnderlyingLongReturn": float(value),
                "Entry": 0,
                "Active": 1,
            })
    else:
        # Fallback keeps the audit usable even when a cached history is missing
        # one exact open date; the report should still flag missing histories.
        rows.append({
            "Date": exit_date,
            "RealityTradeId": int(trade["RealityTradeId"]),
            "Symbol": str(trade["Symbol"]),
            "Horizon": int(trade["Horizon"]),
            "Fold": int(trade["Fold"]),
            "PositionReturn": float(trade["Return"]),
            "UnderlyingLongReturn": float(trade.get("BuyHoldReturn", trade["Return"])) - cost,
            "Entry": 0,
            "Active": 1,
        })
    return pd.DataFrame(rows)


def build_daily_curve(
    trades: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    *,
    max_positions: int | None = None,
    benchmark_history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build an actual daily equity curve from selected trades and cached prices."""
    if trades.empty:
        return pd.DataFrame(columns=[
            "Date", "StrategyReturn", "UnderlyingSameTradesReturn", "BenchmarkReturn",
            "ActivePositions", "Entries", "Equity", "UnderlyingSameTradesEquity", "BenchmarkEquity",
        ]), []

    missing_histories: list[str] = []
    pieces: list[pd.DataFrame] = []
    for _, trade in trades.iterrows():
        symbol = str(trade["Symbol"])
        history = histories.get(symbol)
        if history is None or history.empty:
            missing_histories.append(symbol)
            continue
        pieces.append(trade_daily_returns(trade, history))
    if not pieces:
        return pd.DataFrame(), sorted(set(missing_histories))

    daily_positions = pd.concat(pieces, ignore_index=True)
    daily_positions["Date"] = pd.to_datetime(daily_positions["Date"]).dt.normalize()
    grouped = daily_positions.groupby("Date").agg(
        StrategyGross=("PositionReturn", "sum"),
        UnderlyingGross=("UnderlyingLongReturn", "sum"),
        ActivePositions=("RealityTradeId", "nunique"),
        Entries=("Entry", "sum"),
    ).sort_index()
    slots = max_positions or grouped["ActivePositions"].replace(0, np.nan)
    grouped["StrategyReturn"] = grouped["StrategyGross"] / slots
    grouped["UnderlyingSameTradesReturn"] = grouped["UnderlyingGross"] / slots
    grouped[["StrategyReturn", "UnderlyingSameTradesReturn"]] = grouped[
        ["StrategyReturn", "UnderlyingSameTradesReturn"]
    ].fillna(0.0)

    all_days = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    grouped = grouped.reindex(all_days, fill_value=0.0)
    grouped.index.name = "Date"
    grouped["ActivePositions"] = grouped["ActivePositions"].astype(int)
    grouped["Entries"] = grouped["Entries"].astype(int)

    if benchmark_history is not None and not benchmark_history.empty:
        benchmark_open = _open_series(benchmark_history)
        benchmark_return = benchmark_open.pct_change().reindex(grouped.index).fillna(0.0)
        exposure = grouped["ActivePositions"].clip(lower=0)
        if max_positions:
            exposure = (exposure / max_positions).clip(upper=1.0)
        else:
            exposure = (exposure > 0).astype(float)
        grouped["BenchmarkReturn"] = benchmark_return * exposure
    else:
        grouped["BenchmarkReturn"] = grouped["UnderlyingSameTradesReturn"]

    grouped["Equity"] = (1 + grouped["StrategyReturn"]).cumprod()
    grouped["UnderlyingSameTradesEquity"] = (1 + grouped["UnderlyingSameTradesReturn"]).cumprod()
    grouped["BenchmarkEquity"] = (1 + grouped["BenchmarkReturn"]).cumprod()
    grouped = grouped.reset_index()
    return grouped, sorted(set(missing_histories))


def _profit_factor(returns: pd.Series) -> float | None:
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    if losses <= 0:
        return None
    return gains / losses


def _json_scalar(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _json_records(frame: pd.DataFrame) -> list[dict]:
    return [
        {str(key): _json_scalar(value) for key, value in row.items()}
        for row in frame.astype(object).where(pd.notna(frame), None).to_dict("records")
    ]


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1).min())


def _ci(values: Iterable[float], samples: int, random_state: int) -> tuple[float, float] | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 20:
        return None
    rng = np.random.default_rng(random_state)
    means = [float(rng.choice(array, size=len(array), replace=True).mean()) for _ in range(samples)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _fold_block_ci(trades: pd.DataFrame, samples: int, random_state: int) -> tuple[float, float] | None:
    if trades.empty or trades["Fold"].nunique() < 2:
        return None
    groups = [group["Return"].astype(float).to_numpy() for _, group in trades.groupby("Fold")]
    if sum(len(group) for group in groups) < 20:
        return None
    rng = np.random.default_rng(random_state)
    means = []
    for _ in range(samples):
        picked = [groups[int(i)] for i in rng.integers(0, len(groups), len(groups))]
        combined = np.concatenate(picked)
        means.append(float(combined.mean()))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def trade_summary(trades: pd.DataFrame, config: RealityConfig = RealityConfig()) -> dict:
    returns = trades["Return"].astype(float) if not trades.empty else pd.Series(dtype=float)
    return {
        "selected_trades": int(len(trades)),
        "mean_return": float(returns.mean()) if len(returns) else 0.0,
        "median_return": float(returns.median()) if len(returns) else 0.0,
        "hit_rate": float((returns > 0).mean()) if len(returns) else None,
        "profit_factor": _profit_factor(returns),
        "actual_up_rate": float(trades["ActualUp"].mean()) if "ActualUp" in trades and len(trades) else None,
        "execution_up_rate": float(trades["ExecutionUp"].mean()) if "ExecutionUp" in trades and len(trades) else None,
        "avg_probability": float(trades["Probability"].mean()) if "Probability" in trades and len(trades) else None,
        "avg_expected_return": float(trades["ExpectedReturn"].mean()) if "ExpectedReturn" in trades and len(trades) else None,
        "avg_validation_auc": float(trades["ValidationAUC"].mean()) if "ValidationAUC" in trades and len(trades) else None,
        "avg_validation_brier": float(trades["ValidationBrier"].mean()) if "ValidationBrier" in trades and len(trades) else None,
        "trade_expectancy_ci_95": _ci(returns, config.bootstrap_samples, config.random_state),
        "fold_block_expectancy_ci_95": _fold_block_ci(trades, config.bootstrap_samples, config.random_state),
    }


def curve_summary(curve: pd.DataFrame, config: RealityConfig = RealityConfig()) -> dict:
    if curve.empty:
        return {
            "total_return": 0.0,
            "benchmark_total_return": 0.0,
            "underlying_same_trades_total_return": 0.0,
            "max_drawdown": 0.0,
            "benchmark_max_drawdown": 0.0,
            "daily_sharpe": 0.0,
            "exposure_days": 0.0,
            "avg_active_positions": 0.0,
        }
    daily = curve["StrategyReturn"].astype(float)
    std = float(daily.std())
    return {
        "total_return": float(curve["Equity"].iloc[-1] - 1),
        "benchmark_total_return": float(curve["BenchmarkEquity"].iloc[-1] - 1),
        "underlying_same_trades_total_return": float(curve["UnderlyingSameTradesEquity"].iloc[-1] - 1),
        "max_drawdown": _max_drawdown(curve["Equity"]),
        "benchmark_max_drawdown": _max_drawdown(curve["BenchmarkEquity"]),
        "daily_sharpe": float(daily.mean() / std * math.sqrt(config.annualization_days)) if std > 0 else 0.0,
        "exposure_days": float((curve["ActivePositions"] > 0).mean()),
        "avg_active_positions": float(curve["ActivePositions"].mean()),
    }


def group_reality_summary(
    records: pd.DataFrame, selected: pd.DataFrame, by: str | list[str], config: RealityConfig = RealityConfig(),
) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    keys = by if isinstance(by, list) else [by]
    all_groups = records.groupby(keys, dropna=False)
    rows = []
    for key, group in all_groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        mask = pd.Series(True, index=selected.index)
        for column, value in zip(keys, key_tuple):
            mask &= selected[column] == value
        chosen = selected[mask]
        rows.append({
            **dict(zip(keys, key_tuple)),
            "observations": int(len(group)),
            "raw_signals": int((group["Position"].fillna(0) != 0).sum()),
            **trade_summary(chosen, config),
        })
    return pd.DataFrame(rows)


def reality_check_report(
    records: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    config: RealityConfig = RealityConfig(),
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    filtered = filter_records(records, config)
    selected = select_non_overlapping_trades(filtered, config)
    benchmark_history = histories.get(config.benchmark_symbol or "") if config.benchmark_symbol else None
    curve, missing_histories = build_daily_curve(
        selected,
        histories,
        max_positions=config.max_positions,
        benchmark_history=benchmark_history,
    )
    report = {
        "config": {
            "horizons": config.horizons,
            "symbols": config.symbols,
            "markets": config.markets,
            "one_position_per_symbol": config.one_position_per_symbol,
            "allow_same_day_reentry": config.allow_same_day_reentry,
            "max_positions": config.max_positions,
            "benchmark_symbol": config.benchmark_symbol,
        },
        "summary": {
            "observations": int(len(filtered)),
            "raw_signals": int((filtered["Position"].fillna(0) != 0).sum()) if not filtered.empty else 0,
            **trade_summary(selected, config),
            **curve_summary(curve, config),
        },
        "by_horizon": _json_records(group_reality_summary(filtered, selected, "Horizon", config)),
        "by_symbol": _json_records(group_reality_summary(filtered, selected, "Symbol", config)),
        "by_fold": _json_records(group_reality_summary(filtered, selected, "Fold", config)),
        "missing_histories": missing_histories,
        "methodology": (
            "Reality Check selects the first active signal per symbol and ignores later signals until "
            "that position closes. Portfolio curve uses cached Open-to-Open prices and charges the "
            "recorded round-trip cost at entry. This is a diagnostic audit, not a live trading proof."
        ),
    }
    return _json_scalar(report), selected, curve
