from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .backtest import _fit_backtest_ensemble, _supervised_execution_frame
from .data import download_history
from .model import _model_probabilities, _weighted_prediction
from .signals import SignalInputs, signal_verdict


@dataclass(frozen=True)
class ValidationConfig:
    """Frozen experiment contract for aggregate validation."""

    horizons: tuple[int, ...] = (1, 5, 20)
    threshold: float = 0.56
    cost_bps: float = 10.0
    slippage_bps: float = 5.0
    initial_train: int = 420
    test_size: int = 90
    max_folds: int | None = 4
    holdout_size: int = 90


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_end: int
    test_start: int
    test_end: int
    fold_type: str = "WALK_FORWARD"


def _fold_ranges(length: int, horizon: int, config: ValidationConfig) -> list[FoldSpec]:
    """Return chronological folds with a purge gap before each test block."""
    minimum_train = max(220, config.initial_train)
    test_size = max(20, config.test_size)
    ranges: list[FoldSpec] = []
    test_start = minimum_train + horizon
    fold_id = 1
    holdout_size = max(0, int(config.holdout_size))
    holdout_start: int | None = None
    regular_end = length - horizon
    if holdout_size >= 20 and length - holdout_size >= minimum_train + horizon + 20:
        holdout_start = length - holdout_size
        regular_end = max(minimum_train + horizon, holdout_start - horizon)

    while test_start < regular_end:
        train_end = test_start - horizon
        test_end = min(regular_end, test_start + test_size)
        if train_end >= 220 and test_end - test_start >= 20:
            ranges.append(FoldSpec(fold_id, train_end, test_start, test_end))
            fold_id += 1
        if config.max_folds is not None and len(ranges) >= config.max_folds:
            break
        test_start += test_size
    if holdout_start is not None:
        train_end = holdout_start - horizon
        if train_end >= 220 and length - holdout_start >= 20:
            ranges.append(FoldSpec(fold_id, train_end, holdout_start, length, "HOLDOUT"))
    return ranges


def _signal_record(
    *,
    symbol: str,
    market: str,
    horizon: int,
    fold: FoldSpec,
    index_position: int,
    X: pd.DataFrame,
    y: pd.Series,
    model_forward: pd.Series,
    execution_forward: pd.Series,
    prices: pd.DataFrame,
    state,
    threshold: float,
    total_cost: float,
) -> dict:
    raw_probability = float(
        _weighted_prediction(_model_probabilities(state.class_models, X.iloc[[index_position]]), state.class_weights)[0]
    )
    probability = (
        float(state.calibrator.predict_proba(np.array([[raw_probability]]))[:, 1][0])
        if state.calibrator is not None else raw_probability
    )
    expected_return = float(
        _weighted_prediction(
            {name: model.predict(X.iloc[[index_position]]) for name, model in state.reg_models.items()},
            state.reg_weights,
        )[0]
    )
    inputs = SignalInputs(
        probability=probability,
        expected_return=expected_return,
        quality=state.quality,
        auc=state.validation_auc,
        brier=state.validation_brier,
        source="AGGREGATE_VALIDATION",
    )
    verdict = signal_verdict(inputs, threshold)
    position = verdict.decision
    gross = position * float(execution_forward.iloc[index_position])
    net = gross - abs(position) * total_cost
    linear_probability = float(state.class_models["linear"].predict_proba(X.iloc[[index_position]])[:, 1][0])
    linear_position = 1 if linear_probability >= threshold else (-1 if linear_probability <= 1 - threshold else 0)
    momentum_position = 1 if float(X.iloc[index_position].get("ret_20", 0.0)) > 0 else -1
    always_long_return = float(execution_forward.iloc[index_position]) - total_cost
    buy_hold_return = float(execution_forward.iloc[index_position])
    momentum_return = momentum_position * float(execution_forward.iloc[index_position]) - total_cost
    linear_return = linear_position * float(execution_forward.iloc[index_position]) - abs(linear_position) * total_cost
    price_row = prices.iloc[index_position]
    return {
        "Date": X.index[index_position],
        "Symbol": symbol,
        "Market": market,
        "Horizon": int(horizon),
        "Fold": int(fold.fold_id),
        "FoldType": fold.fold_type,
        "TrainEndDate": X.index[state.train_end - 1],
        "CoreEndDate": X.index[state.core_end - 1],
        "CalibrationStartDate": X.index[state.calibration_start],
        "CalibrationEndDate": X.index[state.calibration_end - 1],
        "TestStartDate": X.index[fold.test_start],
        "TestEndDate": X.index[fold.test_end - 1],
        "PurgeGap": int(horizon),
        "EntryDate": price_row["EntryDate"],
        "ExitDate": price_row["ExitDate"],
        "EntryPrice": float(price_row["EntryPrice"]),
        "ExitPrice": float(price_row["ExitPrice"]),
        "Probability": probability,
        "ExpectedReturn": expected_return,
        "Quality": state.quality,
        "ValidationAUC": state.validation_auc,
        "ValidationBrier": state.validation_brier,
        "Position": int(position),
        "DecisionReason": verdict.reason,
        "DecisionLabel": verdict.label,
        "ModelReturn": float(model_forward.iloc[index_position]),
        "ExecutionReturn": float(execution_forward.iloc[index_position]),
        "GrossReturn": gross,
        "Return": net,
        "RoundTripCost": total_cost,
        "AlwaysLongReturn": always_long_return,
        "BuyHoldReturn": buy_hold_return,
        "MomentumPosition": int(momentum_position),
        "MomentumReturn": momentum_return,
        "LinearProbability": linear_probability,
        "LinearPosition": int(linear_position),
        "LinearReturn": linear_return,
        "ActualUp": int(y.iloc[index_position]),
        "ExecutionUp": int(execution_forward.iloc[index_position] > 0),
        "Target": "close_to_close",
        "Execution": "next_open",
    }


def validate_history(
    symbol: str,
    data: pd.DataFrame,
    *,
    market: str = "Unknown",
    context: pd.DataFrame | None = None,
    config: ValidationConfig = ValidationConfig(),
) -> pd.DataFrame:
    """Validate one instrument across horizons and chronological folds.

    Every potential decision point is retained, including rejected signals. This
    makes the output suitable for aggregate edge analysis and rejection auditing.
    """
    records: list[dict] = []
    total_cost = (config.cost_bps + config.slippage_bps) / 10_000
    for horizon in config.horizons:
        X, y, model_forward, execution_forward, prices = _supervised_execution_frame(data, horizon, context)
        for fold in _fold_ranges(len(X), horizon, config):
            try:
                state = _fit_backtest_ensemble(X, y, model_forward, fold.train_end, horizon)
            except ValueError:
                continue
            for i in range(fold.test_start, fold.test_end):
                records.append(
                    _signal_record(
                        symbol=symbol,
                        market=market,
                        horizon=horizon,
                        fold=fold,
                        index_position=i,
                        X=X,
                        y=y,
                        model_forward=model_forward,
                        execution_forward=execution_forward,
                        prices=prices,
                        state=state,
                        threshold=config.threshold,
                        total_cost=total_cost,
                    )
                )
    return pd.DataFrame(records)


def aggregate_validate_histories(
    histories: dict[str, pd.DataFrame],
    *,
    markets: dict[str, str] | None = None,
    contexts: dict[str, pd.DataFrame | None] | None = None,
    config: ValidationConfig = ValidationConfig(),
) -> pd.DataFrame:
    frames = []
    for symbol, data in histories.items():
        frames.append(
            validate_history(
                symbol,
                data,
                market=(markets or {}).get(symbol, "Unknown"),
                context=(contexts or {}).get(symbol),
                config=config,
            )
        )
    if not frames:
        return pd.DataFrame()
    non_empty = [frame for frame in frames if not frame.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def aggregate_validate_universe(
    symbols: dict[str, str] | list[str] | tuple[str, ...],
    *,
    years: int = 8,
    config: ValidationConfig = ValidationConfig(),
    loader: Callable[[str, int], pd.DataFrame] = download_history,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Download and validate a universe. Intended for explicit, long-running jobs."""
    if isinstance(symbols, dict):
        symbol_to_market = symbols
    else:
        symbol_to_market = {symbol: "Unknown" for symbol in symbols}

    histories: dict[str, pd.DataFrame] = {}
    contexts: dict[str, pd.DataFrame | None] = {}
    errors: dict[str, str] = {}
    try:
        from .engine import _benchmark_for
    except Exception:
        _benchmark_for = lambda symbol: None  # type: ignore[assignment]

    for symbol in symbol_to_market:
        try:
            histories[symbol] = loader(symbol, years)
            benchmark = _benchmark_for(symbol)
            if benchmark:
                try:
                    contexts[symbol] = loader(benchmark, years)
                except Exception:
                    contexts[symbol] = None
            else:
                contexts[symbol] = None
        except Exception as exc:
            errors[symbol] = str(exc)
    frame = aggregate_validate_histories(histories, markets=symbol_to_market, contexts=contexts, config=config)
    return frame, errors


def _portfolio_daily(frame: pd.DataFrame, return_col: str = "Return") -> pd.Series:
    if frame.empty or return_col not in frame:
        return pd.Series(dtype=float)
    scaled = frame[["Date", "Horizon", return_col]].copy()
    scaled["Strategy"] = scaled[return_col].astype(float) / scaled["Horizon"].replace(0, np.nan).astype(float)
    return scaled.groupby("Date")["Strategy"].mean().sort_index().fillna(0.0)


def _risk_stats(daily: pd.Series) -> dict[str, float]:
    if daily.empty:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0}
    std = float(daily.std())
    downside = float(daily[daily < 0].std())
    equity = (1 + daily).cumprod()
    return {
        "sharpe": float(daily.mean() / std * np.sqrt(252)) if np.isfinite(std) and std > 0 else 0.0,
        "sortino": float(daily.mean() / downside * np.sqrt(252)) if np.isfinite(downside) and downside > 0 else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()) if not equity.empty else 0.0,
    }


def _non_overlapping_trades(frame: pd.DataFrame) -> int:
    active = frame[frame["Position"] != 0].sort_values(["Symbol", "EntryDate", "ExitDate"])
    count = 0
    last_exit_by_symbol: dict[str, pd.Timestamp] = {}
    for _, row in active.iterrows():
        symbol = str(row["Symbol"])
        entry = pd.Timestamp(row["EntryDate"])
        exit_date = pd.Timestamp(row["ExitDate"])
        if symbol not in last_exit_by_symbol or entry > last_exit_by_symbol[symbol]:
            count += 1
            last_exit_by_symbol[symbol] = exit_date
    return count


def _concentration_stats(frame: pd.DataFrame) -> dict[str, float | None]:
    active = frame[frame["Position"] != 0]
    if active.empty:
        return {
            "top_symbol_profit_share": None,
            "top_1pct_trade_profit_share": None,
            "return_without_best_1pct": None,
            "return_without_best_5pct": None,
            "return_without_best_10pct": None,
        }
    positive_by_symbol = active.groupby("Symbol")["Return"].sum().clip(lower=0)
    positive_total = float(positive_by_symbol.sum())
    returns = active["Return"].astype(float).sort_values(ascending=False).reset_index(drop=True)
    positive_returns = returns[returns > 0]
    positive_trade_total = float(positive_returns.sum())

    def without_best(percent: float) -> float | None:
        if returns.empty:
            return None
        drop_n = max(1, int(np.ceil(len(returns) * percent)))
        kept = returns.iloc[drop_n:]
        return float(kept.mean()) if len(kept) else 0.0

    top_1pct_n = max(1, int(np.ceil(len(positive_returns) * 0.01))) if len(positive_returns) else 0
    return {
        "top_symbol_profit_share": float(positive_by_symbol.max() / positive_total) if positive_total > 0 else None,
        "top_1pct_trade_profit_share": (
            float(positive_returns.iloc[:top_1pct_n].sum() / positive_trade_total)
            if positive_trade_total > 0 and top_1pct_n else None
        ),
        "return_without_best_1pct": without_best(0.01),
        "return_without_best_5pct": without_best(0.05),
        "return_without_best_10pct": without_best(0.10),
    }


def cost_stress_summary(frame: pd.DataFrame, multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)) -> dict[str, dict]:
    if frame.empty:
        return {f"{multiplier:g}x": {"mean_return": 0.0, "profit_factor": None} for multiplier in multipliers}
    out: dict[str, dict] = {}
    for multiplier in multipliers:
        stressed = frame.copy()
        stressed["Return"] = stressed["GrossReturn"] - stressed["Position"].abs() * stressed["RoundTripCost"] * multiplier
        active_returns = stressed.loc[stressed["Position"] != 0, "Return"].astype(float)
        gains = active_returns[active_returns > 0].sum()
        losses = -active_returns[active_returns < 0].sum()
        out[f"{multiplier:g}x"] = {
            "mean_return": float(active_returns.mean()) if len(active_returns) else 0.0,
            "median_return": float(active_returns.median()) if len(active_returns) else 0.0,
            "profit_factor": float(gains / losses) if losses > 0 else (None if gains == 0 else float("inf")),
        }
    return out


def _current_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def experiment_manifest(
    config: ValidationConfig,
    symbols: list[str] | tuple[str, ...] | dict[str, str],
    *,
    commit_hash: str | None = None,
) -> dict:
    universe = sorted(symbols.keys() if isinstance(symbols, dict) else symbols)
    commit = commit_hash or _current_commit()
    payload = {
        "commit": commit,
        "config": asdict(config),
        "universe": universe,
        "engine": "SignalInputs+classifier/regression ensemble",
        "target": "close_to_close",
        "execution": "next_open",
    }
    experiment_id = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return {
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }


def aggregate_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "observations": 0,
            "trades": 0,
            "rejected": 0,
            "non_overlapping_trades": 0,
            "mean_return": 0.0,
            "median_return": 0.0,
            "net_expectancy": 0.0,
            "profit_factor": None,
            "hit_rate": None,
            "exposure": 0.0,
            "avg_concurrent_positions": 0.0,
            "turnover_per_day": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "auc": 0.5,
            "brier": 0.25,
            "rejection_reasons": {},
            "cost_stress": {},
        }

    active = frame["Position"] != 0
    returns = frame.loc[active, "Return"].astype(float)
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    probability = frame["Probability"].clip(1e-4, 1 - 1e-4)
    rejection_reasons = frame.loc[~active, "DecisionReason"].value_counts().to_dict()
    daily = _portfolio_daily(frame)
    risk = _risk_stats(daily)
    concurrent = frame.loc[active].groupby("Date")["Position"].count() if active.any() else pd.Series(dtype=float)
    ci = bootstrap_mean_return(frame, samples=250)
    return {
        "observations": int(len(frame)),
        "trades": int(active.sum()),
        "rejected": int((~active).sum()),
        "non_overlapping_trades": _non_overlapping_trades(frame),
        "mean_return": float(returns.mean()) if len(returns) else 0.0,
        "median_return": float(returns.median()) if len(returns) else 0.0,
        "net_expectancy": float(returns.mean()) if len(returns) else 0.0,
        "expectancy_ci_95": ci,
        "profit_factor": float(gains / losses) if losses > 0 else (None if gains == 0 else float("inf")),
        "hit_rate": float((returns > 0).mean()) if len(returns) else None,
        "exposure": float(active.mean()),
        "avg_concurrent_positions": float(concurrent.mean()) if len(concurrent) else 0.0,
        "turnover_per_day": float(active.sum() / max(1, frame["Date"].nunique())),
        **risk,
        "auc": float(roc_auc_score(frame["ActualUp"], probability)) if frame["ActualUp"].nunique() > 1 else 0.5,
        "brier": float(brier_score_loss(frame["ActualUp"], probability)),
        "avg_validation_auc": float(frame["ValidationAUC"].mean()),
        "avg_validation_brier": float(frame["ValidationBrier"].mean()),
        "rejection_reasons": {str(key): int(value) for key, value in rejection_reasons.items()},
        "benchmark_mean_returns": {
            "always_long": float(frame["AlwaysLongReturn"].mean()),
            "buy_hold_proxy": float(frame["BuyHoldReturn"].mean()),
            "momentum": float(frame["MomentumReturn"].mean()),
            "logistic_regression": float(frame["LinearReturn"].mean()),
        },
        "cost_stress": cost_stress_summary(frame),
        **_concentration_stats(frame),
    }


def group_summary(frame: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    grouped = frame.groupby(by, dropna=False)
    for key, group in grouped:
        summary = aggregate_summary(group)
        if not isinstance(key, tuple):
            key = (key,)
        keys = by if isinstance(by, list) else [by]
        rows.append({**dict(zip(keys, key)), **{k: v for k, v in summary.items() if k != "rejection_reasons"}})
    return pd.DataFrame(rows)


def bootstrap_mean_return(
    frame: pd.DataFrame, *, samples: int = 500, random_state: int = 42,
) -> tuple[float, float] | None:
    """Simple signal-level CI for active returns; conservative block bootstrap comes next."""
    returns = frame.loc[frame["Position"] != 0, "Return"].astype(float).to_numpy()
    if len(returns) < 20:
        return None
    rng = np.random.default_rng(random_state)
    means = [float(rng.choice(returns, size=len(returns), replace=True).mean()) for _ in range(samples)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def validation_report(
    frame: pd.DataFrame,
    config: ValidationConfig,
    symbols: list[str] | tuple[str, ...] | dict[str, str],
    *,
    commit_hash: str | None = None,
) -> dict:
    """Produce an auditable experiment report without hiding weak groups."""
    return {
        "manifest": experiment_manifest(config, symbols, commit_hash=commit_hash),
        "summary": aggregate_summary(frame),
        "by_market": group_summary(frame, "Market").to_dict("records"),
        "by_symbol": group_summary(frame, "Symbol").to_dict("records"),
        "by_horizon": group_summary(frame, "Horizon").to_dict("records"),
        "by_fold": group_summary(frame, ["FoldType", "Fold"]).to_dict("records"),
    }
