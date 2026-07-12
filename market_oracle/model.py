from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


@dataclass
class Forecast:
    probability_up: float
    expected_return: float
    lower_return: float
    upper_return: float
    accuracy: float
    auc: float
    brier: float
    quality: str
    baseline_accuracy: float
    validation_start: str
    validation_end: str
    linear_weight: float
    model_weights: dict[str, float]
    validation_folds: int
    samples: int
    importance: pd.Series


def _classification_models() -> dict[str, object]:
    return {
        "linear": make_pipeline(
            SimpleImputer(strategy="median"), RobustScaler(),
            LogisticRegression(C=0.35, max_iter=2000, class_weight="balanced"),
        ),
        "boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                learning_rate=0.04, max_iter=180, max_leaf_nodes=15,
                min_samples_leaf=24, l2_regularization=2.5, random_state=42,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=180, max_depth=7, min_samples_leaf=10, max_features=0.65,
                class_weight="balanced", bootstrap=True, n_jobs=-1, random_state=42,
            ),
        ),
    }


def _regression_models() -> dict[str, object]:
    return {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), Ridge(alpha=18.0)),
        "forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=120, max_depth=6, min_samples_leaf=12,
                max_features=0.65, n_jobs=-1, random_state=42,
            ),
        ),
        "boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.04, max_iter=160, max_leaf_nodes=15,
                min_samples_leaf=24, l2_regularization=2.0, random_state=42,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=160, max_depth=7, min_samples_leaf=10,
                max_features=0.65, bootstrap=True, n_jobs=-1, random_state=42,
            ),
        ),
    }


def _model_probabilities(models: dict[str, object], X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: model.predict_proba(X)[:, 1] for name, model in models.items()}


def _weighted_prediction(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    available = {name: values for name, values in predictions.items() if name in weights}
    if not available:
        raise ValueError("Brak predykcji modeli do złożenia ensemble.")
    total = sum(max(0.0, weights.get(name, 0.0)) for name in available)
    if total <= 0:
        equal = 1 / len(available)
        return sum(equal * values for values in available.values())
    return sum(weights[name] / total * values for name, values in available.items())


def _walk_forward_predictions(
    X: pd.DataFrame, y: pd.Series, horizon: int, folds: int = 3,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Expanding-window validation with a purge gap before each test block."""
    minimum_train = max(180, int(len(X) * 0.45))
    remaining = len(X) - minimum_train
    if remaining < max(90, folds * 30):
        return {}, np.array([]), 0

    block = max(35, min(100, remaining // folds))
    starts = np.linspace(minimum_train, max(minimum_train, len(X) - block), folds, dtype=int)
    collected = {name: [] for name in _classification_models()}
    y_parts: list[np.ndarray] = []
    used_folds = 0

    for start in dict.fromkeys(int(s) for s in starts):
        end = min(len(X), start + block)
        train_end = max(120, start - horizon)
        if end - start < 25 or train_end <= 120:
            continue
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[start:end], y.iloc[start:end]
        if y_train.nunique() < 2 or y_test.empty:
            continue
        models = _classification_models()
        try:
            for model in models.values():
                model.fit(X_train, y_train)
            fold_predictions = _model_probabilities(models, X_test)
        except Exception:
            continue
        for name, values in fold_predictions.items():
            collected[name].append(values)
        y_parts.append(y_test.to_numpy())
        used_folds += 1

    if not y_parts:
        return {}, np.array([]), 0
    return {name: np.concatenate(parts) for name, parts in collected.items() if parts}, np.concatenate(y_parts), used_folds


def _classification_weights(predictions: dict[str, np.ndarray], y_true: np.ndarray) -> dict[str, float]:
    default = {"linear": 0.45, "boosting": 0.35, "extra_trees": 0.20}
    if len(y_true) < 50 or not predictions:
        return default
    briers = {
        name: brier_score_loss(y_true, np.clip(values, 1e-4, 1 - 1e-4))
        for name, values in predictions.items()
    }
    best = min(briers.values())
    raw = {name: np.exp(-(score - best) / 0.012) for name, score in briers.items()}
    total = sum(raw.values())
    if not np.isfinite(total) or total <= 0:
        return default
    # Keep every model alive a little; financial regimes change and the recent winner is not always the next winner.
    weights = {name: 0.08 + 0.92 * value / total for name, value in raw.items()}
    normalizer = sum(weights.values())
    return {name: float(value / normalizer) for name, value in weights.items()}


def _regression_weights(predictions: dict[str, np.ndarray], y_true: pd.Series) -> dict[str, float]:
    default = {"ridge": 0.40, "forest": 0.25, "boosting": 0.20, "extra_trees": 0.15}
    if len(y_true) < 40 or not predictions:
        return default
    errors = {name: mean_absolute_error(y_true, values) for name, values in predictions.items()}
    best = min(errors.values())
    scale = max(0.0025, abs(float(y_true.std())) * 0.20)
    raw = {name: np.exp(-(error - best) / scale) for name, error in errors.items()}
    total = sum(raw.values())
    if not np.isfinite(total) or total <= 0:
        return default
    weights = {name: 0.06 + 0.94 * value / total for name, value in raw.items()}
    normalizer = sum(weights.values())
    return {name: float(value / normalizer) for name, value in weights.items()}


def _importance(models: dict[str, object], weights: dict[str, float], columns: pd.Index) -> pd.Series:
    importance = pd.Series(0.0, index=columns, dtype=float)
    if "linear" in models:
        coef = np.abs(models["linear"].named_steps["logisticregression"].coef_[0])
        if coef.sum() > 0:
            importance += weights.get("linear", 0.0) * pd.Series(coef / coef.sum(), index=columns)
    if "extra_trees" in models:
        tree = models["extra_trees"].named_steps["extratreesclassifier"]
        values = getattr(tree, "feature_importances_", np.zeros(len(columns)))
        if values.sum() > 0:
            importance += weights.get("extra_trees", 0.0) * pd.Series(values / values.sum(), index=columns)
    if importance.sum() == 0:
        return pd.Series(1 / len(columns), index=columns).head(8)
    return importance.sort_values(ascending=False).head(8)


def fit_forecast(X: pd.DataFrame, y: pd.Series, returns: pd.Series, latest: pd.DataFrame, horizon: int) -> Forecast:
    if len(X) < 250 or y.nunique() < 2:
        raise ValueError("Za mało zróżnicowanych danych do treningu modelu.")
    split = max(200, int(len(X) * 0.78))
    split = min(split, len(X) - 60)
    # Purge observations whose label reaches into the validation interval.
    train_end = max(100, split - horizon)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    r_train = returns.iloc[:train_end]

    # When history allows it, probability calibration and final assessment use separate time blocks.
    validation_size = len(X) - split
    use_calibration = validation_size >= max(120, 4 * horizon + 40)
    if use_calibration:
        calibration_size = validation_size // 2
        calibration_end = split + calibration_size
        assessment_start = min(calibration_end + horizon, len(X) - 30)
        X_cal, y_cal = X.iloc[split:calibration_end], y.iloc[split:calibration_end]
        X_valid, y_valid = X.iloc[assessment_start:], y.iloc[assessment_start:]
        r_valid = returns.iloc[assessment_start:]
    else:
        X_cal = y_cal = None
        X_valid, y_valid = X.iloc[split:], y.iloc[split:]
        r_valid = returns.iloc[split:]

    oof_predictions, y_oof, validation_folds = _walk_forward_predictions(X_train, y_train, horizon)
    model_weights = _classification_weights(oof_predictions, y_oof)

    eval_models = _classification_models()
    for model in eval_models.values():
        model.fit(X_train, y_train)
    calibrator = None
    if len(y_oof) > 80 and len(np.unique(y_oof)) > 1:
        oof_raw = _weighted_prediction(oof_predictions, model_weights)
        calibrator = LogisticRegression(C=0.5, max_iter=1000)
        calibrator.fit(oof_raw.reshape(-1, 1), y_oof)
    elif use_calibration and y_cal is not None and y_cal.nunique() > 1:
        calibration_raw = _weighted_prediction(_model_probabilities(eval_models, X_cal), model_weights)
        calibrator = LogisticRegression(C=0.5, max_iter=1000)
        calibrator.fit(calibration_raw.reshape(-1, 1), y_cal)

    valid_raw = _weighted_prediction(_model_probabilities(eval_models, X_valid), model_weights)
    valid_prob = calibrator.predict_proba(valid_raw.reshape(-1, 1))[:, 1] if calibrator is not None else valid_raw

    # Shrink probabilities toward 0.5 when validation is weak or small.
    auc = roc_auc_score(y_valid, valid_prob) if y_valid.nunique() > 1 else 0.5
    brier = float(brier_score_loss(y_valid, valid_prob))
    oof_auc = roc_auc_score(y_oof, _weighted_prediction(oof_predictions, model_weights)) if len(y_oof) and len(np.unique(y_oof)) > 1 else 0.5
    discrimination = np.clip((auc - 0.5) / 0.12, 0.0, 1.0)
    calibration = np.clip((0.265 - brier) / 0.055, 0.0, 1.0)
    stability = np.clip((oof_auc - 0.5) / 0.12, 0.0, 1.0)
    skill = float(discrimination * calibration * (0.65 + 0.35 * stability))
    if auc >= 0.60 and brier <= 0.24:
        quality = "WYSOKA"
    elif auc >= 0.55 and brier <= 0.26:
        quality = "UMIARKOWANA"
    else:
        quality = "NISKA — BRAK PRZEWAGI"
    # Evaluation stays untouched, while production models learn from every labeled observation.
    production_models = _classification_models()
    for model in production_models.values():
        model.fit(X, y)
    raw_prob = float(_weighted_prediction(_model_probabilities(production_models, latest), model_weights)[0])
    if calibrator is not None:
        raw_prob = float(calibrator.predict_proba(np.array([[raw_prob]]))[:, 1][0])
    probability = 0.5 + skill * (raw_prob - 0.5)

    eval_reg_models = _regression_models()
    for model in eval_reg_models.values():
        model.fit(X_train, r_train)
    valid_reg_predictions = {name: model.predict(X_valid) for name, model in eval_reg_models.items()}
    reg_weights = _regression_weights(valid_reg_predictions, r_valid)
    valid_expected = _weighted_prediction(valid_reg_predictions, reg_weights)
    residual = r_valid - valid_expected

    production_reg_models = _regression_models()
    for model in production_reg_models.values():
        model.fit(X, returns)
    latest_reg_predictions = {name: model.predict(latest) for name, model in production_reg_models.items()}
    expected = float(_weighted_prediction(latest_reg_predictions, reg_weights)[0])
    expected *= skill
    sigma = float(max(residual.std(), r_valid.std() * 0.35, 1e-4))
    lower = float(expected + residual.quantile(0.05)) if len(residual) >= 30 else expected - 1.645 * sigma
    upper = float(expected + residual.quantile(0.95)) if len(residual) >= 30 else expected + 1.645 * sigma
    if lower >= upper:
        lower, upper = expected - 1.645 * sigma, expected + 1.645 * sigma

    importance = _importance(production_models, model_weights, X.columns)
    accuracy = float(accuracy_score(y_valid, valid_prob >= 0.5))
    positive_rate = float(y_valid.mean())
    return Forecast(
        probability_up=float(np.clip(probability, 0.02, 0.98)), expected_return=expected,
        lower_return=lower, upper_return=upper,
        accuracy=accuracy, auc=float(auc), brier=brier, quality=quality,
        baseline_accuracy=max(positive_rate, 1 - positive_rate),
        validation_start=str(X_valid.index[0].date()), validation_end=str(X_valid.index[-1].date()),
        linear_weight=float(model_weights.get("linear", 0.0)),
        model_weights={name: float(weight) for name, weight in model_weights.items()},
        validation_folds=validation_folds,
        samples=len(X), importance=importance,
    )
