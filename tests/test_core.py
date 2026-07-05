import numpy as np
import pandas as pd

from market_oracle.backtest import walk_forward_backtest
from market_oracle.catalog import CATEGORIES, CRYPTO, ETF_CATEGORIES
from market_oracle.engine import observation_label, signal_label
from market_oracle.features import build_features, supervised_frame
from market_oracle.model import fit_forecast
from market_oracle.risk import periods_per_year, risk_metrics


def synthetic_data(n=900):
    rng = np.random.default_rng(42)
    returns = 0.00025 + 0.18 * np.sin(np.arange(n) / 17) / 100 + rng.normal(0, 0.011, n)
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.002, n)),
        "High": close * (1 + rng.uniform(0.001, 0.015, n)),
        "Low": close * (1 - rng.uniform(0.001, 0.015, n)),
        "Close": close,
        "Volume": rng.integers(100_000, 2_000_000, n),
    }, index=pd.bdate_range("2020-01-01", periods=n))


def test_features_are_finite_and_past_only():
    data = synthetic_data()
    original = build_features(data).iloc[400].copy()
    changed = data.copy()
    changed.loc[changed.index[500]:, "Close"] *= 2
    after = build_features(changed).iloc[400]
    pd.testing.assert_series_equal(original, after)


def test_forecast_bounds():
    data = synthetic_data()
    X, y, ret = supervised_frame(data, 5)
    latest = build_features(data).dropna().iloc[[-1]]
    result = fit_forecast(X, y, ret, latest, 5)
    assert 0 <= result.probability_up <= 1
    assert result.lower_return < result.upper_return
    assert result.samples > 250
    assert result.validation_start < result.validation_end
    assert result.baseline_accuracy >= 0.5


def test_risk_and_backtest():
    data = synthetic_data()
    metrics = risk_metrics(data.Close)
    assert metrics["annual_volatility"] > 0
    curve, summary = walk_forward_backtest(data, horizon=5)
    assert not curve.empty
    assert np.isfinite(summary["total_return"])
    assert metrics["periods_per_year"] == 252


def test_crypto_uses_365_day_annualization():
    assert periods_per_year(pd.date_range("2024-01-01", periods=500, freq="D")) == 365


def test_catalog_is_broad_and_symbols_are_present():
    assert sum(len(group) for group in CATEGORIES.values()) >= 150
    assert sum(len(group) for group in ETF_CATEGORIES.values()) >= 50
    assert len(CRYPTO) >= 25
    assert CATEGORIES["GPW — największe spółki"]["CD Projekt"] == "CDR.WA"
    assert CATEGORIES["USA — mniejsze i spekulacyjne"]["Rocket Lab"] == "RKLB"


def test_market_context_features_are_added():
    data = synthetic_data()
    context = synthetic_data()
    context["Close"] *= 1.05
    features = build_features(data, context).dropna()
    assert "market_beta_60" in features
    assert "relative_strength_20" in features
    assert np.isfinite(features["market_corr_60"]).all()


def test_low_quality_forecast_never_becomes_buy_signal():
    forecast = {"quality": "NISKA — BRAK PRZEWAGI", "probability_up": 0.9, "expected_return": 0.1}
    assert observation_label(forecast) == "BRAK SYGNAŁU"
    assert signal_label(0.9, forecast["quality"]) == "BRAK PRZEWAGI"
    confirmed = {"quality": "UMIARKOWANA", "probability_up": 0.60, "expected_return": 0.03}
    assert observation_label(confirmed) == "KANDYDAT WZROSTOWY"
