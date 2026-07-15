from __future__ import annotations

from dataclasses import dataclass
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


def _fold_ranges(length: int, horizon: int, config: ValidationConfig) -> list[tuple[int, int, int, int]]:
    """Return (fold_id, train_end, test_start, test_end) with a purge gap."""
    minimum_train = max(220, config.initial_train)
    test_size = max(20, config.test_size)
    ranges: list[tuple[int, int, int, int]] = []
    test_start = minimum_train + horizon
    fold_id = 1
    while test_start < length - horizon:
        train_end = test_start - horizon
        test_end = min(length, test_start + test_size)
        if train_end >= 220 and test_end - test_start >= 20:
            ranges.append((fold_id, train_end, test_start, test_end))
            fold_id += 1
        if config.max_folds is not None and len(ranges) >= config.max_folds:
            break
        test_start += test_size
    return ranges


def _signal_record(
    *,
    symbol: str,
    market: str,
    horizon: int,
    fold_id: int,
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
    price_row = prices.iloc[index_position]
    return {
        "Date": X.index[index_position],
        "Symbol": symbol,
        "Market": market,
        "Horizon": int(horizon),
        "Fold": int(fold_id),
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
        for fold_id, train_end, test_start, test_end in _fold_ranges(len(X), horizon, config):
            try:
                state = _fit_backtest_ensemble(X, y, model_forward, train_end, horizon)
            except ValueError:
                continue
            for i in range(test_start, test_end):
                records.append(
                    _signal_record(
                        symbol=symbol,
                        market=market,
                        horizon=horizon,
                        fold_id=fold_id,
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


def aggregate_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "observations": 0,
            "trades": 0,
            "rejected": 0,
            "mean_return": 0.0,
            "median_return": 0.0,
            "profit_factor": None,
            "hit_rate": None,
            "auc": 0.5,
            "brier": 0.25,
            "rejection_reasons": {},
        }

    active = frame["Position"] != 0
    returns = frame.loc[active, "Return"].astype(float)
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    probability = frame["Probability"].clip(1e-4, 1 - 1e-4)
    rejection_reasons = frame.loc[~active, "DecisionReason"].value_counts().to_dict()
    return {
        "observations": int(len(frame)),
        "trades": int(active.sum()),
        "rejected": int((~active).sum()),
        "mean_return": float(returns.mean()) if len(returns) else 0.0,
        "median_return": float(returns.median()) if len(returns) else 0.0,
        "profit_factor": float(gains / losses) if losses > 0 else (None if gains == 0 else float("inf")),
        "hit_rate": float((returns > 0).mean()) if len(returns) else None,
        "auc": float(roc_auc_score(frame["ActualUp"], probability)) if frame["ActualUp"].nunique() > 1 else 0.5,
        "brier": float(brier_score_loss(frame["ActualUp"], probability)),
        "avg_validation_auc": float(frame["ValidationAUC"].mean()),
        "avg_validation_brier": float(frame["ValidationBrier"].mean()),
        "rejection_reasons": {str(key): int(value) for key, value in rejection_reasons.items()},
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
