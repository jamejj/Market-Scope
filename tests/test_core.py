import numpy as np
import pandas as pd

from market_oracle.backtest import walk_forward_backtest
from market_oracle.catalog import CATEGORIES, CRYPTO, CRYPTO_CATEGORIES, ETF_CATEGORIES
from market_oracle.engine import observation_label, signal_label
from market_oracle.features import build_features, supervised_frame
from market_oracle.journal import journal_summary, load_journal, paper_portfolio, record_snapshot_signals, refresh_journal_results
from market_oracle.model import fit_forecast
from market_oracle.monitor import default_universe, load_snapshot, run_signal_scan, snapshot_is_stale
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
    assert 0 <= result.linear_weight <= 1


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
    assert "DeFi / giełdy / tokeny protokołów" in CRYPTO_CATEGORIES
    assert CRYPTO["DeXe"] == "DEXE-USD"
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


def test_background_monitor_persists_snapshot(tmp_path, monkeypatch):
    sample = pd.DataFrame([{
        "Symbol": "TEST", "Ocena": "OBSERWUJ", "Score": 1.5,
        "P(wzrost)": 0.52, "Oczekiwany ruch": 0.01,
    }])
    monkeypatch.setattr("market_oracle.monitor.scan_market_multi", lambda symbols, horizons, years: (sample, {}))
    path = tmp_path / "signals.json"
    result = run_signal_scan(["TEST"], path=path)
    loaded = load_snapshot(path)
    assert result["status"] == "complete"
    assert loaded["records"][0]["Symbol"] == "TEST"
    assert len(default_universe()) >= 100


def test_old_signal_snapshot_is_considered_stale():
    old_snapshot = {
        "status": "complete", "updated_at": "2026-07-12T12:00:00+00:00",
        "horizon": 20, "completed": 41, "total": 41,
        "records": [{"Symbol": "SPY", "Ocena": "OBSERWUJ"}],
        "errors": {},
    }
    assert snapshot_is_stale(old_snapshot)


def test_signal_journal_records_and_evaluates(tmp_path, monkeypatch):
    snapshot = {
        "status": "complete", "updated_at": "2026-01-10T12:00:00+00:00",
        "records": [
            {
                "Symbol": "TEST", "Klasa": "USA", "Horyzont": 5, "Data": "2026-01-10",
                "Cena": 100.0, "Setup": "Breakout / momentum",
                "Ocena": "KANDYDAT WZROSTOWY", "P(wzrost)": 0.61,
                "Oczekiwany ruch": 0.03, "AUC walidacji": 0.62,
                "Brier": 0.22, "Jakość modelu": "WYSOKA", "Score": 4.2,
            },
            {"Symbol": "SKIP", "Horyzont": 5, "Cena": 10.0, "Ocena": "BRAK SYGNAŁU"},
        ],
    }
    path = tmp_path / "journal.json"
    assert record_snapshot_signals(snapshot, path=path) == 1
    assert record_snapshot_signals(snapshot, path=path) == 0
    entries = load_journal(path)
    assert len(entries) == 1
    assert entries[0]["direction"] == "LONG"

    dates = pd.bdate_range("2026-01-09", periods=12)
    prices = np.linspace(99, 111, len(dates))
    history = pd.DataFrame({
        "Open": prices, "High": prices + 1, "Low": prices - 1,
        "Close": prices, "Volume": 1_000_000,
    }, index=dates)
    monkeypatch.setattr("market_oracle.journal.download_history", lambda symbol, years=3: history)
    refreshed, errors = refresh_journal_results(path=path)
    assert errors == {}
    assert refreshed[0]["status"] == "closed"
    assert refreshed[0]["hit"] is True
    assert refreshed[0]["strategy_return"] > 0
    summary = journal_summary(refreshed)
    assert summary["profit_factor"] is None
    assert summary["expectancy"] > 0
    assert summary["max_drawdown"] == 0


def test_journal_summary_risk_metrics():
    entries = [
        {"status": "closed", "strategy_return": 0.10, "hit": True},
        {"status": "closed", "strategy_return": -0.04, "hit": False},
        {"status": "closed", "strategy_return": 0.02, "hit": True},
        {"status": "open"},
    ]
    summary = journal_summary(entries)
    assert summary["closed"] == 3
    assert summary["open"] == 1
    assert summary["profit_factor"] == 3.0
    assert summary["payoff_ratio"] == 1.5
    assert summary["max_drawdown"] < 0


def test_paper_portfolio_applies_sizing_and_costs():
    entries = [
        {
            "status": "closed", "signal_date": "2026-01-01", "target_date": "2026-01-06",
            "symbol": "AAA", "asset_class": "USA", "horizon": 5, "direction": "LONG",
            "strategy_return": 0.10,
        },
        {
            "status": "closed", "signal_date": "2026-01-02", "target_date": "2026-01-07",
            "symbol": "BBB", "asset_class": "USA", "horizon": 5, "direction": "LONG",
            "strategy_return": -0.05,
        },
        {"status": "open", "strategy_return": 1.0},
    ]
    curve, summary = paper_portfolio(entries, starting_capital=10_000, position_fraction=0.20, round_trip_cost_bps=25)
    assert len(curve) == 2
    assert curve["Zwrot netto"].iloc[0] == 0.0975
    assert curve["P&L"].iloc[0] == 195
    assert summary["trades"] == 2
    assert summary["final_capital"] < 10_195
    assert summary["total_return"] > 0
    assert summary["max_drawdown"] < 0
