from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .features import build_features
from .model import FittedForecastState, fit_forecast_state
from .risk import periods_per_year
from .signals import DEFAULT_SIGNAL_THRESHOLD, signal_verdict


def _execution_returns(data: pd.DataFrame, horizon: int) -> pd.Series:
    """Signal after close t, enter next open, exit open after horizon bars."""
    open_price = data["Open"].astype(float)
    return open_price.shift(-(horizon + 1)) / open_price.shift(-1) - 1


def _execution_prices(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    open_price = data["Open"].astype(float)
    return pd.DataFrame({
        "EntryPrice": open_price.shift(-1),
        "ExitPrice": open_price.shift(-(horizon + 1)),
        "EntryDate": pd.Series(data.index, index=data.index).shift(-1),
        "ExitDate": pd.Series(data.index, index=data.index).shift(-(horizon + 1)),
    }, index=data.index)


def _supervised_execution_frame(
    data: pd.DataFrame, horizon: int, context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    features = build_features(data, context)
    # Keep the prediction target aligned with production: close[t+h] > close[t].
    # Execution P&L is measured separately from next open to future open.
    close_to_close = data["Close"].astype(float).shift(-horizon) / data["Close"].astype(float) - 1
    execution_return = _execution_returns(data, horizon)
    prices = _execution_prices(data, horizon)
    target = (close_to_close > 0).astype(float)
    valid = (
        features.notna().all(axis=1)
        & close_to_close.notna()
        & execution_return.notna()
        & prices["EntryPrice"].notna()
        & prices["ExitPrice"].notna()
    )
    return (
        features.loc[valid],
        target.loc[valid].astype(int),
        close_to_close.loc[valid],
        execution_return.loc[valid],
        prices.loc[valid],
    )


def _fit_backtest_state(
    X: pd.DataFrame, y: pd.Series, returns: pd.Series, train_end: int, horizon: int,
) -> FittedForecastState:
    """Fit the exact shared production forecast state on the available past slice."""
    return fit_forecast_state(X.iloc[:train_end], y.iloc[:train_end], returns.iloc[:train_end], horizon)


def walk_forward_backtest(
    data: pd.DataFrame,
    horizon: int = 5,
    threshold: float = DEFAULT_SIGNAL_THRESHOLD,
    cost_bps: float = 10,
    slippage_bps: float = 5,
    context: pd.DataFrame | None = None,
    refit_every: int = 60,
) -> tuple[pd.DataFrame, dict]:
    """Expanding walk-forward test of the production model family.

    Prediction is made after close on day t, execution is next session open, and
    exit is the open after the requested horizon. Costs are round-trip bps plus
    simplified slippage bps.
    """
    X, y, model_forward, execution_forward, prices = _supervised_execution_frame(data, horizon, context)
    start = max(300, int(len(X) * 0.55))
    records = []
    state: FittedForecastState | None = None
    for i in range(start, len(X)):
        if state is None or (i - start) % max(1, refit_every) == 0:
            train_end = i - horizon - 1
            if train_end < 250:
                continue
            state = _fit_backtest_state(X, y, model_forward, train_end, horizon)
        prediction = state.predict(X.iloc[[i]])
        inputs = prediction.signal_inputs(source="BACKTEST")
        verdict = signal_verdict(inputs, threshold)
        position = verdict.decision
        gross = position * float(execution_forward.iloc[i])
        total_cost = abs(position) * (cost_bps + slippage_bps) / 10_000
        net = gross - total_cost
        price_row = prices.iloc[i]
        records.append({
            "Date": X.index[i],
            "EntryDate": price_row["EntryDate"],
            "ExitDate": price_row["ExitDate"],
            "EntryPrice": float(price_row["EntryPrice"]),
            "ExitPrice": float(price_row["ExitPrice"]),
            "Probability": prediction.probability_up,
            "ExpectedReturn": prediction.expected_return,
            "Quality": state.quality,
            "ValidationAUC": state.auc,
            "ValidationBrier": state.brier,
            "Skill": state.skill,
            "Position": position,
            "DecisionReason": verdict.reason,
            "DecisionLabel": verdict.label,
            "GrossReturn": gross,
            "Return": net,
            "ActualUp": int(y.iloc[i]),
            "ExecutionUp": int(execution_forward.iloc[i] > 0),
        })
    result = pd.DataFrame(records).set_index("Date")
    if result.empty:
        raise ValueError("Za mało danych do backtestu.")
    # Horizon trades overlap; divide exposure to avoid pretending each trade has full independent capital.
    result["Strategy"] = result["Return"] / horizon
    result["Equity"] = (1 + result["Strategy"]).cumprod()
    benchmark = data["Open"].reindex(result.index).pct_change().fillna(0)
    result["BuyHold"] = (1 + benchmark).cumprod()
    daily = result["Strategy"]
    active = result["Position"] != 0
    annualizer = periods_per_year(data.index)
    probability = result["Probability"].clip(1e-4, 1 - 1e-4)
    metrics = {
        "total_return": float(result["Equity"].iloc[-1] - 1),
        "annual_return": float(result["Equity"].iloc[-1] ** (annualizer / len(result)) - 1),
        "annual_volatility": float(daily.std() * np.sqrt(annualizer)),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(annualizer)) if daily.std() else 0.0,
        "max_drawdown": float((result["Equity"] / result["Equity"].cummax() - 1).min()),
        "trades": int(active.sum()),
        "hit_rate": float((result.loc[active, "Return"] > 0).mean()) if active.any() else 0.0,
        "auc": float(roc_auc_score(result["ActualUp"], probability)) if result["ActualUp"].nunique() > 1 else 0.5,
        "brier": float(brier_score_loss(result["ActualUp"], probability)),
        "execution_hit_rate": float((result.loc[active, "GrossReturn"] > 0).mean()) if active.any() else 0.0,
        "target": "close_to_close",
        "execution": "next_open",
        "cost_bps": float(cost_bps),
        "slippage_bps": float(slippage_bps),
    }
    return result, metrics
