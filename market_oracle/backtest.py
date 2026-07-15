from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .features import build_features
from .model import (
    _classification_models,
    _classification_weights,
    _model_probabilities,
    _regression_models,
    _regression_weights,
    _weighted_prediction,
)
from .risk import periods_per_year
from .signals import SignalInputs, signal_verdict


@dataclass
class _BacktestEnsemble:
    class_models: dict[str, object]
    class_weights: dict[str, float]
    calibrator: LogisticRegression | None
    reg_models: dict[str, object]
    reg_weights: dict[str, float]
    quality: str
    validation_auc: float
    validation_brier: float
    train_end: int
    core_end: int
    calibration_start: int
    calibration_end: int


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


def _quality_from_validation(y_true: pd.Series, probability: np.ndarray) -> tuple[str, float, float]:
    probability = np.clip(probability, 1e-4, 1 - 1e-4)
    if len(y_true) < 35 or y_true.nunique() < 2:
        return "NISKA — BRAK PRZEWAGI", 0.5, 0.25
    auc = float(roc_auc_score(y_true, probability))
    brier = float(brier_score_loss(y_true, probability))
    if auc >= 0.60 and brier <= 0.24:
        quality = "WYSOKA"
    elif auc >= 0.55 and brier <= 0.26:
        quality = "UMIARKOWANA"
    else:
        quality = "NISKA — BRAK PRZEWAGI"
    return quality, auc, brier


def _fit_backtest_ensemble(
    X: pd.DataFrame, y: pd.Series, returns: pd.Series, train_end: int, horizon: int,
) -> _BacktestEnsemble:
    """Fit the production classifier and expected-return model families on past data."""
    train_end = int(train_end)
    calibration_size = max(45, min(120, train_end // 5))
    calibration_start = max(180 + horizon, train_end - calibration_size)
    core_end = max(160, calibration_start - horizon)
    if calibration_start >= train_end - 20 or core_end >= calibration_start:
        calibration_start = max(180 + horizon, int(train_end * 0.78))
        core_end = max(160, calibration_start - horizon)

    X_core, y_core = X.iloc[:core_end], y.iloc[:core_end]
    X_cal, y_cal = X.iloc[calibration_start:train_end], y.iloc[calibration_start:train_end]
    r_core = returns.iloc[:core_end]
    r_cal = returns.iloc[calibration_start:train_end]
    if y_core.nunique() < 2:
        raise ValueError("Za mało zróżnicowanych danych w oknie treningowym.")

    weight_models = _classification_models()
    for model in weight_models.values():
        model.fit(X_core, y_core)
    quality = "NISKA — BRAK PRZEWAGI"
    validation_auc = 0.5
    validation_brier = 0.25
    if len(X_cal) >= 35 and y_cal.nunique() > 1:
        calibration_predictions = _model_probabilities(weight_models, X_cal)
        weights = _classification_weights(calibration_predictions, y_cal.to_numpy())
        raw_cal = _weighted_prediction(calibration_predictions, weights)
        quality, validation_auc, validation_brier = _quality_from_validation(y_cal, raw_cal)
        calibrator = LogisticRegression(C=0.5, max_iter=1000)
        calibrator.fit(raw_cal.reshape(-1, 1), y_cal)
    else:
        weights = {"linear": 0.45, "boosting": 0.35, "extra_trees": 0.20}
        calibrator = None

    weight_reg_models = _regression_models()
    for model in weight_reg_models.values():
        model.fit(X_core, r_core)
    if len(X_cal) >= 35:
        reg_predictions = {name: model.predict(X_cal) for name, model in weight_reg_models.items()}
        reg_weights = _regression_weights(reg_predictions, r_cal)
    else:
        reg_weights = {"ridge": 0.40, "forest": 0.25, "boosting": 0.20, "extra_trees": 0.15}

    models = _classification_models()
    for model in models.values():
        model.fit(X.iloc[:train_end], y.iloc[:train_end])
    reg_models = _regression_models()
    for model in reg_models.values():
        model.fit(X.iloc[:train_end], returns.iloc[:train_end])
    return _BacktestEnsemble(
        class_models=models,
        class_weights=weights,
        calibrator=calibrator,
        reg_models=reg_models,
        reg_weights=reg_weights,
        quality=quality,
        validation_auc=validation_auc,
        validation_brier=validation_brier,
        train_end=train_end,
        core_end=core_end,
        calibration_start=calibration_start,
        calibration_end=train_end,
    )


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
    X, y, model_forward, execution_forward, prices = _supervised_execution_frame(data, horizon, context)
    start = max(300, int(len(X) * 0.55))
    records = []
    state: _BacktestEnsemble | None = None
    for i in range(start, len(X)):
        if state is None or (i - start) % max(1, refit_every) == 0:
            train_end = i - horizon - 1
            if train_end < 200:
                continue
            state = _fit_backtest_ensemble(X, y, model_forward, train_end, horizon)
        raw_probability = float(
            _weighted_prediction(_model_probabilities(state.class_models, X.iloc[[i]]), state.class_weights)[0]
        )
        probability = (
            float(state.calibrator.predict_proba(np.array([[raw_probability]]))[:, 1][0])
            if state.calibrator is not None else raw_probability
        )
        expected_return = float(
            _weighted_prediction(
                {name: model.predict(X.iloc[[i]]) for name, model in state.reg_models.items()},
                state.reg_weights,
            )[0]
        )
        inputs = SignalInputs(
            probability=probability,
            expected_return=expected_return,
            quality=state.quality,
            auc=state.validation_auc,
            brier=state.validation_brier,
            source="BACKTEST",
        )
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
            "Probability": probability,
            "ExpectedReturn": expected_return,
            "Quality": state.quality,
            "ValidationAUC": state.validation_auc,
            "ValidationBrier": state.validation_brier,
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
