from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .features import build_features
from .model import _classification_models, _classification_weights, _model_probabilities, _weighted_prediction
from .risk import periods_per_year
from .signals import signal_decision


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
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    features = build_features(data, context)
    # Keep the prediction target aligned with production: close[t+h] > close[t].
    # Execution P&L is measured separately from next open to future open.
    close_to_close = data["Close"].astype(float).shift(-horizon) / data["Close"].astype(float) - 1
    realized = _execution_returns(data, horizon)
    prices = _execution_prices(data, horizon)
    target = (close_to_close > 0).astype(float)
    valid = (
        features.notna().all(axis=1)
        & close_to_close.notna()
        & realized.notna()
        & prices["EntryPrice"].notna()
        & prices["ExitPrice"].notna()
    )
    return features.loc[valid], target.loc[valid].astype(int), realized.loc[valid], prices.loc[valid]


def _fit_backtest_ensemble(
    X: pd.DataFrame, y: pd.Series, train_end: int, horizon: int,
) -> tuple[dict[str, object], dict[str, float], LogisticRegression | None]:
    """Fit the same classifier family as production using only past data."""
    train_end = int(train_end)
    calibration_size = max(45, min(120, train_end // 5))
    calibration_start = max(180 + horizon, train_end - calibration_size)
    core_end = max(160, calibration_start - horizon)
    if calibration_start >= train_end - 20 or core_end >= calibration_start:
        calibration_start = max(180 + horizon, int(train_end * 0.78))
        core_end = max(160, calibration_start - horizon)

    X_core, y_core = X.iloc[:core_end], y.iloc[:core_end]
    X_cal, y_cal = X.iloc[calibration_start:train_end], y.iloc[calibration_start:train_end]
    if y_core.nunique() < 2:
        raise ValueError("Za mało zróżnicowanych danych w oknie treningowym.")

    weight_models = _classification_models()
    for model in weight_models.values():
        model.fit(X_core, y_core)
    if len(X_cal) >= 35 and y_cal.nunique() > 1:
        calibration_predictions = _model_probabilities(weight_models, X_cal)
        weights = _classification_weights(calibration_predictions, y_cal.to_numpy())
        raw_cal = _weighted_prediction(calibration_predictions, weights)
        calibrator = LogisticRegression(C=0.5, max_iter=1000)
        calibrator.fit(raw_cal.reshape(-1, 1), y_cal)
    else:
        weights = {"linear": 0.45, "boosting": 0.35, "extra_trees": 0.20}
        calibrator = None

    models = _classification_models()
    for model in models.values():
        model.fit(X.iloc[:train_end], y.iloc[:train_end])
    return models, weights, calibrator


def walk_forward_backtest(
    data: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.56,
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
    X, y, forward, prices = _supervised_execution_frame(data, horizon, context)
    start = max(300, int(len(X) * 0.55))
    records = []
    models = None
    weights: dict[str, float] | None = None
    calibrator = None
    for i in range(start, len(X)):
        if models is None or (i - start) % max(1, refit_every) == 0:
            train_end = i - horizon - 1
            if train_end < 200:
                continue
            models, weights, calibrator = _fit_backtest_ensemble(X, y, train_end, horizon)
        raw_probability = float(_weighted_prediction(_model_probabilities(models, X.iloc[[i]]), weights or {})[0])
        probability = (
            float(calibrator.predict_proba(np.array([[raw_probability]]))[:, 1][0])
            if calibrator is not None else raw_probability
        )
        # Backtest still only has the classifier leg; expected return is set by direction so
        # the entry rule stays consistent with production's "probability + direction" gate.
        expected_direction = 1.0 if probability >= 0.5 else -1.0
        position = signal_decision(probability, expected_direction, "WYSOKA", threshold)
        gross = position * float(forward.iloc[i])
        total_cost = abs(position) * (cost_bps + slippage_bps) / 10_000
        net = gross - total_cost
        price_row = prices.iloc[i]
        records.append({
            "Date": X.index[i],
            "EntryDate": price_row["EntryDate"],
            "ExitDate": price_row["ExitDate"],
            "EntryPrice": float(price_row["EntryPrice"]),
            "ExitPrice": float(price_row["ExitPrice"]),
            "Probability": probability,
            "Position": position,
            "GrossReturn": gross,
            "Return": net,
            "ActualUp": int(y.iloc[i]),
            "ExecutionUp": int(forward.iloc[i] > 0),
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
