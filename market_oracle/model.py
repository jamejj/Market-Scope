from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
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
    samples: int
    importance: pd.Series


def _models() -> tuple[object, object]:
    linear = make_pipeline(
        SimpleImputer(strategy="median"), RobustScaler(),
        LogisticRegression(C=0.35, max_iter=2000, class_weight="balanced"),
    )
    tree = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            learning_rate=0.045, max_iter=160, max_leaf_nodes=15,
            min_samples_leaf=25, l2_regularization=2.0, random_state=42,
        ),
    )
    return linear, tree


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

    eval_linear, eval_tree = _models()
    eval_linear.fit(X_train, y_train)
    eval_tree.fit(X_train, y_train)
    calibrator = None
    linear_weight = 0.55
    if use_calibration and y_cal is not None and y_cal.nunique() > 1:
        cal_linear = eval_linear.predict_proba(X_cal)[:, 1]
        cal_tree = eval_tree.predict_proba(X_cal)[:, 1]
        candidate_weights = np.linspace(0.0, 1.0, 21)
        linear_weight = float(min(
            candidate_weights,
            key=lambda weight: brier_score_loss(y_cal, weight * cal_linear + (1 - weight) * cal_tree),
        ))
        calibration_raw = linear_weight * cal_linear + (1 - linear_weight) * cal_tree
        calibrator = LogisticRegression(C=0.5, max_iter=1000)
        calibrator.fit(calibration_raw.reshape(-1, 1), y_cal)
    valid_raw = linear_weight * eval_linear.predict_proba(X_valid)[:, 1] + (1 - linear_weight) * eval_tree.predict_proba(X_valid)[:, 1]
    valid_prob = calibrator.predict_proba(valid_raw.reshape(-1, 1))[:, 1] if calibrator is not None else valid_raw

    # Shrink probabilities toward 0.5 when validation is weak or small.
    auc = roc_auc_score(y_valid, valid_prob) if y_valid.nunique() > 1 else 0.5
    brier = float(brier_score_loss(y_valid, valid_prob))
    discrimination = np.clip((auc - 0.5) / 0.12, 0.0, 1.0)
    calibration = np.clip((0.265 - brier) / 0.055, 0.0, 1.0)
    skill = float(discrimination * calibration)
    if auc >= 0.60 and brier <= 0.24:
        quality = "WYSOKA"
    elif auc >= 0.55 and brier <= 0.26:
        quality = "UMIARKOWANA"
    else:
        quality = "NISKA — BRAK PRZEWAGI"
    # Evaluation stays untouched, while production models learn from every labeled observation.
    linear, tree = _models()
    linear.fit(X, y)
    tree.fit(X, y)
    raw_prob = float(linear_weight * linear.predict_proba(latest)[:, 1][0] + (1 - linear_weight) * tree.predict_proba(latest)[:, 1][0])
    if calibrator is not None:
        raw_prob = float(calibrator.predict_proba(np.array([[raw_prob]]))[:, 1][0])
    probability = 0.5 + skill * (raw_prob - 0.5)

    eval_reg_linear = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), Ridge(alpha=18.0))
    eval_reg_tree = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=14, max_features=0.65, n_jobs=-1, random_state=42),
    )
    eval_reg_linear.fit(X_train, r_train)
    eval_reg_tree.fit(X_train, r_train)
    residual = r_valid - (0.65 * eval_reg_linear.predict(X_valid) + 0.35 * eval_reg_tree.predict(X_valid))

    reg_linear = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), Ridge(alpha=18.0))
    reg_tree = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=14, max_features=0.65, n_jobs=-1, random_state=42),
    )
    reg_linear.fit(X, returns)
    reg_tree.fit(X, returns)
    expected = float(0.65 * reg_linear.predict(latest)[0] + 0.35 * reg_tree.predict(latest)[0])
    expected *= skill
    sigma = float(max(residual.std(), r_valid.std() * 0.35, 1e-4))

    coef = linear.named_steps["logisticregression"].coef_[0]
    importance = pd.Series(np.abs(coef), index=X.columns).sort_values(ascending=False).head(8)
    accuracy = float(accuracy_score(y_valid, valid_prob >= 0.5))
    positive_rate = float(y_valid.mean())
    return Forecast(
        probability_up=float(np.clip(probability, 0.02, 0.98)), expected_return=expected,
        lower_return=expected - 1.645 * sigma, upper_return=expected + 1.645 * sigma,
        accuracy=accuracy, auc=float(auc), brier=brier, quality=quality,
        baseline_accuracy=max(positive_rate, 1 - positive_rate),
        validation_start=str(X_valid.index[0].date()), validation_end=str(X_valid.index[-1].date()),
        linear_weight=linear_weight,
        samples=len(X), importance=importance,
    )
