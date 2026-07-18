from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


DATE_COLUMNS = ("Date", "EntryDate", "ExitDate")
PRICE_CHECK_COLUMNS = ("EntryPrice", "ExitPrice")
CRYPTO_MARKET_LABELS = {"CRYPTO", "KRYPTO", "Krypto"}


@dataclass(frozen=True)
class RealityConfig:
    """Conservative execution audit for already-generated validation records."""

    horizons: tuple[int, ...] | None = None
    symbols: tuple[str, ...] | None = None
    markets: tuple[str, ...] | None = None
    one_position_per_symbol: bool = True
    allow_same_day_reentry: bool = False
    max_positions: int | None = 5
    portfolio_slots: int = 5
    benchmark_symbol: str | None = "SPY"
    annualization_days: int | None = None
    bootstrap_samples: int = 2000
    random_state: int = 42
    strict_history: bool = True
    price_tolerance_bps: float = 2.0
    calendar_block_days: int = 180


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


def _portfolio_slots(config: RealityConfig) -> int:
    slots = config.max_positions if config.max_positions is not None else config.portfolio_slots
    return max(1, int(slots or 1))


def _closed_before(exit_date: pd.Timestamp, entry_date: pd.Timestamp, config: RealityConfig) -> bool:
    return exit_date <= entry_date if config.allow_same_day_reentry else exit_date < entry_date


def audit_trade_selection(records: pd.DataFrame, config: RealityConfig = RealityConfig()) -> pd.DataFrame:
    """Chronologically audit which active signals survive overlap/position constraints."""
    frame = filter_records(records, config)
    active = frame[frame["Position"].fillna(0) != 0].copy()
    if active.empty:
        return active.assign(
            ActiveSignalId=pd.Series(dtype=int),
            RealityTradeId=pd.Series(dtype=object),
            RealitySelection=pd.Series(dtype=object),
        )

    active = active.sort_values(
        ["EntryDate", "Date", "Symbol", "Horizon", "Probability", "ExpectedReturn"],
        ascending=[True, True, True, True, False, False],
    ).reset_index(drop=True)
    active["ActiveSignalId"] = np.arange(1, len(active) + 1)
    active["RealityTradeId"] = None
    active["RealitySelection"] = "PENDING"

    open_by_symbol: dict[str, pd.Timestamp] = {}
    open_portfolio: list[tuple[str, pd.Timestamp]] = []
    trade_id = 0

    for idx, row in active.iterrows():
        symbol = str(row["Symbol"])
        entry = pd.Timestamp(row["EntryDate"])
        exit_date = pd.Timestamp(row["ExitDate"])
        open_portfolio = [
            (open_symbol, open_exit)
            for open_symbol, open_exit in open_portfolio
            if not _closed_before(open_exit, entry, config)
        ]
        if config.one_position_per_symbol and symbol in open_by_symbol:
            last_exit = open_by_symbol[symbol]
            if not _closed_before(last_exit, entry, config):
                active.at[idx, "RealitySelection"] = "SYMBOL_OVERLAP"
                continue
        if config.max_positions is not None and len(open_portfolio) >= config.max_positions:
            active.at[idx, "RealitySelection"] = "GLOBAL_POSITION_CAP"
            continue
        trade_id += 1
        active.at[idx, "RealityTradeId"] = trade_id
        active.at[idx, "RealitySelection"] = "SELECTED"
        open_by_symbol[symbol] = exit_date
        open_portfolio.append((symbol, exit_date))

    return active


def select_non_overlapping_trades(records: pd.DataFrame, config: RealityConfig = RealityConfig()) -> pd.DataFrame:
    """Select first valid signal per symbol and optionally cap total concurrent positions.

    The output is not a claim of fully independent trades: different symbols can
    still overlap globally unless max_positions forces them out.
    """
    audit = audit_trade_selection(records, config)
    selected = audit[audit["RealitySelection"] == "SELECTED"].copy()
    if selected.empty:
        return selected
    selected["RealityTradeId"] = selected["RealityTradeId"].astype(int)
    selected["RejectedByReality"] = False
    return selected.reset_index(drop=True)


def _open_series(history: pd.DataFrame) -> pd.Series:
    series = history["Open"].astype(float).copy()
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    return series[~series.index.duplicated(keep="last")].sort_index()


def _open_window(history: pd.DataFrame, entry: pd.Timestamp, exit_date: pd.Timestamp) -> pd.Series:
    opens = _open_series(history)
    return opens.loc[(opens.index >= entry) & (opens.index <= exit_date)].dropna()


def _price_on(history: pd.DataFrame, date: pd.Timestamp) -> float | None:
    opens = _open_series(history)
    if date not in opens.index:
        return None
    value = float(opens.loc[date])
    return value if math.isfinite(value) else None


def _price_matches(expected: float, actual: float, tolerance_bps: float) -> bool:
    tolerance = max(1e-8, abs(expected) * tolerance_bps / 10_000)
    return abs(expected - actual) <= tolerance


def validate_trade_price_alignment(
    trades: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    *,
    tolerance_bps: float = 2.0,
) -> list[dict]:
    """Verify selected trade execution prices against cached Open prices."""
    issues: list[dict] = []
    for _, trade in trades.iterrows():
        symbol = str(trade["Symbol"])
        history = histories.get(symbol)
        trade_id = int(trade["RealityTradeId"])
        if history is None or history.empty:
            issues.append({"trade_id": trade_id, "symbol": symbol, "issue": "MISSING_HISTORY"})
            continue
        for date_column, price_column in (("EntryDate", "EntryPrice"), ("ExitDate", "ExitPrice")):
            date = pd.Timestamp(trade[date_column]).normalize()
            actual = _price_on(history, date)
            expected = float(trade[price_column])
            if actual is None:
                issues.append({
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "issue": "MISSING_OPEN",
                    "date_column": date_column,
                    "date": date,
                })
                continue
            if not _price_matches(expected, actual, tolerance_bps):
                issues.append({
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "issue": "PRICE_MISMATCH",
                    "date_column": date_column,
                    "date": date,
                    "expected": expected,
                    "actual": actual,
                })
    return issues


def _calendar_returns(history: pd.DataFrame, entry: pd.Timestamp, exit_date: pd.Timestamp) -> pd.Series:
    opens = _open_window(history, entry, exit_date)
    returns = opens.pct_change().dropna()
    calendar = pd.date_range(entry, exit_date, freq="D")
    return returns.reindex(calendar).fillna(0.0).astype(float)


def trade_daily_returns(trade: pd.Series, history: pd.DataFrame) -> pd.DataFrame:
    """Open-to-open daily path for one selected trade, with cost charged once at entry."""
    entry = pd.Timestamp(trade["EntryDate"]).normalize()
    exit_date = pd.Timestamp(trade["ExitDate"]).normalize()
    position = int(trade["Position"])
    cost = abs(position) * float(trade.get("RoundTripCost", 0.0) or 0.0)
    daily = _calendar_returns(history, entry, exit_date)
    rows: list[dict] = []
    for date, value in daily.items():
        is_entry = pd.Timestamp(date).normalize() == entry
        is_exit = pd.Timestamp(date).normalize() == exit_date
        entry_cost = cost if is_entry else 0.0
        rows.append({
            "Date": pd.Timestamp(date).normalize(),
            "RealityTradeId": int(trade["RealityTradeId"]),
            "Symbol": str(trade["Symbol"]),
            "Horizon": int(trade["Horizon"]),
            "Fold": int(trade["Fold"]),
            "PositionReturn": position * float(value) - entry_cost,
            "UnderlyingLongReturn": float(value) - entry_cost,
            "Entry": int(is_entry),
            "Exit": int(is_exit),
            "Active": 1,
        })
    return pd.DataFrame(rows)


def benchmark_daily_returns(trade: pd.Series, benchmark_history: pd.DataFrame) -> pd.DataFrame:
    """Benchmark path using the same slot, dates and cost as the MarketScope trade."""
    entry = pd.Timestamp(trade["EntryDate"]).normalize()
    exit_date = pd.Timestamp(trade["ExitDate"]).normalize()
    cost = abs(int(trade["Position"])) * float(trade.get("RoundTripCost", 0.0) or 0.0)
    daily = _calendar_returns(benchmark_history, entry, exit_date)
    rows = []
    for date, value in daily.items():
        is_entry = pd.Timestamp(date).normalize() == entry
        rows.append({
            "Date": pd.Timestamp(date).normalize(),
            "RealityTradeId": int(trade["RealityTradeId"]),
            "BenchmarkSlotReturn": float(value) - (cost if is_entry else 0.0),
        })
    return pd.DataFrame(rows)


def build_daily_curve(
    trades: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    *,
    portfolio_slots: int = 5,
    benchmark_history: pd.DataFrame | None = None,
    strict_history: bool = True,
    price_tolerance_bps: float = 2.0,
) -> tuple[pd.DataFrame, list[dict]]:
    """Build a fixed-slot portfolio curve from selected trades and cached Open prices."""
    empty_columns = [
        "Date", "StrategyReturn", "UnderlyingSameTradesReturn", "BenchmarkReturn",
        "ActivePositions", "Entries", "Exits", "GrossExposure", "CashWeight",
        "Equity", "UnderlyingSameTradesEquity", "BenchmarkEquity",
    ]
    if trades.empty:
        return pd.DataFrame(columns=empty_columns), []

    slots = max(1, int(portfolio_slots))
    price_issues = validate_trade_price_alignment(
        trades,
        histories,
        tolerance_bps=price_tolerance_bps,
    )
    if strict_history and price_issues:
        preview = "; ".join(str(issue) for issue in price_issues[:3])
        raise ValueError(f"Reality Check price/history audit failed: {preview}")

    pieces: list[pd.DataFrame] = []
    benchmark_pieces: list[pd.DataFrame] = []
    for _, trade in trades.iterrows():
        symbol = str(trade["Symbol"])
        history = histories.get(symbol)
        if history is None or history.empty:
            continue
        pieces.append(trade_daily_returns(trade, history))
        if benchmark_history is not None and not benchmark_history.empty:
            benchmark_pieces.append(benchmark_daily_returns(trade, benchmark_history))
    if not pieces:
        if strict_history:
            raise ValueError("Reality Check has selected trades but no usable cached histories.")
        return pd.DataFrame(columns=empty_columns), price_issues

    daily_positions = pd.concat(pieces, ignore_index=True)
    daily_positions["Date"] = pd.to_datetime(daily_positions["Date"]).dt.normalize()
    grouped = daily_positions.groupby("Date").agg(
        StrategyGross=("PositionReturn", "sum"),
        UnderlyingGross=("UnderlyingLongReturn", "sum"),
        ActivePositions=("RealityTradeId", "nunique"),
        Entries=("Entry", "sum"),
        Exits=("Exit", "sum"),
    ).sort_index()
    grouped["StrategyReturn"] = grouped["StrategyGross"] / slots
    grouped["UnderlyingSameTradesReturn"] = grouped["UnderlyingGross"] / slots

    all_days = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    grouped = grouped.reindex(all_days, fill_value=0.0)
    grouped.index.name = "Date"
    grouped["ActivePositions"] = grouped["ActivePositions"].astype(int)
    grouped["Entries"] = grouped["Entries"].astype(int)
    grouped["Exits"] = grouped["Exits"].astype(int)
    grouped["GrossExposure"] = grouped["ActivePositions"] / slots
    grouped["CashWeight"] = (1.0 - grouped["GrossExposure"]).clip(lower=0.0)

    if benchmark_pieces:
        daily_benchmark = pd.concat(benchmark_pieces, ignore_index=True)
        daily_benchmark["Date"] = pd.to_datetime(daily_benchmark["Date"]).dt.normalize()
        bench = daily_benchmark.groupby("Date").agg(BenchmarkGross=("BenchmarkSlotReturn", "sum")).sort_index()
        bench = bench.reindex(grouped.index, fill_value=0.0)
        grouped["BenchmarkReturn"] = bench["BenchmarkGross"] / slots
    else:
        grouped["BenchmarkReturn"] = grouped["UnderlyingSameTradesReturn"]

    grouped["Equity"] = (1 + grouped["StrategyReturn"]).cumprod()
    grouped["UnderlyingSameTradesEquity"] = (1 + grouped["UnderlyingSameTradesReturn"]).cumprod()
    grouped["BenchmarkEquity"] = (1 + grouped["BenchmarkReturn"]).cumprod()
    grouped = grouped.reset_index()
    return grouped, price_issues


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


def _calendar_block_ci(
    trades: pd.DataFrame,
    samples: int,
    random_state: int,
    block_days: int,
) -> tuple[float, float] | None:
    if trades.empty or "EntryDate" not in trades:
        return None
    start = pd.to_datetime(trades["EntryDate"]).min()
    if pd.isna(start):
        return None
    block_ids = ((pd.to_datetime(trades["EntryDate"]) - start).dt.days // max(1, block_days)).astype(int)
    groups = [group["Return"].astype(float).to_numpy() for _, group in trades.groupby(block_ids)]
    if len(groups) < 2 or sum(len(group) for group in groups) < 20:
        return None
    rng = np.random.default_rng(random_state)
    means = []
    for _ in range(samples):
        picked = [groups[int(i)] for i in rng.integers(0, len(groups), len(groups))]
        combined = np.concatenate(picked)
        means.append(float(combined.mean()))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _without_best_trades(trades: pd.DataFrame, max_drop: int = 3) -> dict[str, dict]:
    if trades.empty:
        return {}
    ordered = trades["Return"].astype(float).sort_values(ascending=False).reset_index(drop=True)
    out: dict[str, dict] = {}
    for drop in range(1, max_drop + 1):
        remaining = ordered.iloc[drop:]
        if remaining.empty:
            continue
        out[f"drop_best_{drop}"] = {
            "trades": int(len(remaining)),
            "mean_return": float(remaining.mean()),
            "median_return": float(remaining.median()),
            "hit_rate": float((remaining > 0).mean()),
            "profit_factor": _profit_factor(remaining),
        }
    return out


def _effective_annualization(config: RealityConfig, records: pd.DataFrame) -> int:
    if config.annualization_days:
        return int(config.annualization_days)
    markets = {str(value).upper() for value in records.get("Market", pd.Series(dtype=object)).dropna().unique()}
    if markets and markets <= {"CRYPTO", "KRYPTO"}:
        return 365
    if markets and "CRYPTO" not in markets and "KRYPTO" not in markets:
        return 252
    return 365


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
        "calendar_block_expectancy_ci_95": _calendar_block_ci(
            trades,
            config.bootstrap_samples,
            config.random_state,
            config.calendar_block_days,
        ),
        "without_best_trades": _without_best_trades(trades),
    }


def curve_summary(curve: pd.DataFrame, config: RealityConfig = RealityConfig(), records: pd.DataFrame | None = None) -> dict:
    annualization = _effective_annualization(config, records if records is not None else pd.DataFrame())
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
            "max_active_positions": 0,
            "avg_gross_exposure": 0.0,
            "max_gross_exposure": 0.0,
            "portfolio_slots": _portfolio_slots(config),
            "annualization_days": annualization,
        }
    daily = curve["StrategyReturn"].astype(float)
    std = float(daily.std())
    return {
        "total_return": float(curve["Equity"].iloc[-1] - 1),
        "benchmark_total_return": float(curve["BenchmarkEquity"].iloc[-1] - 1),
        "underlying_same_trades_total_return": float(curve["UnderlyingSameTradesEquity"].iloc[-1] - 1),
        "max_drawdown": _max_drawdown(curve["Equity"]),
        "benchmark_max_drawdown": _max_drawdown(curve["BenchmarkEquity"]),
        "daily_sharpe": float(daily.mean() / std * math.sqrt(annualization)) if std > 0 else 0.0,
        "exposure_days": float((curve["ActivePositions"] > 0).mean()),
        "avg_active_positions": float(curve["ActivePositions"].mean()),
        "max_active_positions": int(curve["ActivePositions"].max()),
        "avg_gross_exposure": float(curve["GrossExposure"].mean()),
        "max_gross_exposure": float(curve["GrossExposure"].max()),
        "portfolio_slots": _portfolio_slots(config),
        "annualization_days": annualization,
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


def _selection_counts(audit: pd.DataFrame) -> dict[str, int]:
    if audit.empty or "RealitySelection" not in audit:
        return {}
    return {str(key): int(value) for key, value in audit["RealitySelection"].value_counts().to_dict().items()}


def reality_check_report(
    records: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    config: RealityConfig = RealityConfig(),
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    filtered = filter_records(records, config)
    audit = audit_trade_selection(filtered, config)
    selected = audit[audit["RealitySelection"] == "SELECTED"].copy()
    if not selected.empty:
        selected["RealityTradeId"] = selected["RealityTradeId"].astype(int)
        selected["RejectedByReality"] = False
        selected = selected.reset_index(drop=True)

    benchmark_history = histories.get(config.benchmark_symbol or "") if config.benchmark_symbol else None
    curve, price_issues = build_daily_curve(
        selected,
        histories,
        portfolio_slots=_portfolio_slots(config),
        benchmark_history=benchmark_history,
        strict_history=config.strict_history,
        price_tolerance_bps=config.price_tolerance_bps,
    )
    report = {
        "config": {
            "horizons": config.horizons,
            "symbols": config.symbols,
            "markets": config.markets,
            "one_position_per_symbol": config.one_position_per_symbol,
            "allow_same_day_reentry": config.allow_same_day_reentry,
            "max_positions": config.max_positions,
            "portfolio_slots": config.portfolio_slots,
            "benchmark_symbol": config.benchmark_symbol,
            "annualization_days": config.annualization_days,
            "strict_history": config.strict_history,
            "price_tolerance_bps": config.price_tolerance_bps,
            "calendar_block_days": config.calendar_block_days,
        },
        "summary": {
            "observations": int(len(filtered)),
            "raw_signals": int((filtered["Position"].fillna(0) != 0).sum()) if not filtered.empty else 0,
            "selection_counts": _selection_counts(audit),
            **trade_summary(selected, config),
            **curve_summary(curve, config, filtered),
        },
        "by_horizon": _json_records(group_reality_summary(filtered, selected, "Horizon", config)),
        "by_symbol": _json_records(group_reality_summary(filtered, selected, "Symbol", config)),
        "by_fold": _json_records(group_reality_summary(filtered, selected, "Fold", config)),
        "price_issues": _json_scalar(price_issues),
        "methodology": (
            "Reality Check selects active signals chronologically, removes same-symbol overlap, and can cap global "
            "concurrent positions. The equity curve uses fixed capital slots assigned at entry; existing positions "
            "are not freely rebalanced when other positions open or close. Benchmark slots use the same entry/exit "
            "dates and round-trip costs. This is a diagnostic audit, not a live trading proof."
        ),
    }
    return _json_scalar(report), selected, curve
