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
    X_valid, y_valid = X.iloc[split:], y.iloc[split:]
    r_train, r_valid = returns.iloc[:train_end], returns.iloc[split:]

    linear, tree = _models()
    linear.fit(X_train, y_train)
    tree.fit(X_train, y_train)
    valid_prob = 0.55 * linear.predict_proba(X_valid)[:, 1] + 0.45 * tree.predict_proba(X_valid)[:, 1]

    # Shrink probabilities toward 0.5 when validation is weak or small.
    auc = roc_auc_score(y_valid, valid_prob) if y_valid.nunique() > 1 else 0.5
    brier = float(brier_score_loss(y_valid, valid_prob))
    discrimination = np.clip((auc - 0.5) / 0.12, 0.0, 1.0)
    calibration = np.clip((0.265 - brier) / 0.055, 0.0, 1.0)
    skill = float(discrimination * calibration)
    if auc >= 0.60 and brier <= 0.24:
        quality = "WYSOKA"
    elif auc >= 0.55 and brier <= 0.255:
        quality = "UMIARKOWANA"
    else:
        quality = "NISKA — BRAK PRZEWAGI"
    raw_prob = float(0.55 * linear.predict_proba(latest)[:, 1][0] + 0.45 * tree.predict_proba(latest)[:, 1][0])
    probability = 0.5 + skill * (raw_prob - 0.5)

    reg_linear = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), Ridge(alpha=18.0))
    reg_tree = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=180, max_depth=5, min_samples_leaf=14, max_features=0.65, n_jobs=-1, random_state=42),
    )
    reg_linear.fit(X_train, r_train)
    reg_tree.fit(X_train, r_train)
    expected = float(0.65 * reg_linear.predict(latest)[0] + 0.35 * reg_tree.predict(latest)[0])
    expected *= skill
    residual = r_valid - (0.65 * reg_linear.predict(X_valid) + 0.35 * reg_tree.predict(X_valid))
    sigma = float(max(residual.std(), r_valid.std() * 0.35, 1e-4))

    coef = linear.named_steps["logisticregression"].coef_[0]
    importance = pd.Series(np.abs(coef), index=X.columns).sort_values(ascending=False).head(8)
    return Forecast(
        probability_up=float(np.clip(probability, 0.02, 0.98)), expected_return=expected,
        lower_return=expected - 1.645 * sigma, upper_return=expected + 1.645 * sigma,
        accuracy=float(accuracy_score(y_valid, valid_prob >= 0.5)), auc=float(auc),
        brier=brier, quality=quality, samples=len(X), importance=importance,
    )
