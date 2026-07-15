import numpy as np
import pandas as pd

from market_oracle.backtest import _supervised_execution_frame, walk_forward_backtest
from market_oracle.catalog import CATEGORIES, CRYPTO, CRYPTO_CATEGORIES, ETF_CATEGORIES
from market_oracle.engine import observation_label, risk_reward_metrics, scan_market_fast, setup_intelligence, signal_label
from market_oracle.features import build_features, supervised_frame
from market_oracle.journal import journal_summary, load_journal, paper_portfolio, record_snapshot_signals, refresh_journal_results
from market_oracle.model import fit_forecast
from market_oracle.monitor import default_universe, load_snapshot, run_signal_scan, select_deep_shortlist, snapshot_is_stale
from market_oracle.risk import periods_per_year, risk_metrics
from market_oracle.signals import SignalInputs, signal_decision, signal_inputs_from_forecast
from market_oracle.validation import ValidationConfig, aggregate_summary, aggregate_validate_histories, group_summary


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
    data = synthetic_data(560)
    metrics = risk_metrics(data.Close)
    assert metrics["annual_volatility"] > 0
    curve, summary = walk_forward_backtest(data, horizon=5, refit_every=120)
    assert not curve.empty
    assert np.isfinite(summary["total_return"])
    assert metrics["periods_per_year"] == 252
    assert summary["execution"] == "next_open"
    assert summary["target"] == "close_to_close"
    assert 0 <= summary["auc"] <= 1
    assert "ExpectedReturn" in curve
    assert "Quality" in curve


def test_backtest_target_is_close_to_close_but_pnl_is_next_open():
    data = synthetic_data(460)
    idx = 320
    signal_day = data.index[idx]
    next_day = data.index[idx + 1]
    exit_day = data.index[idx + 2]
    data.loc[signal_day, "Close"] = 100.0
    data.loc[next_day, "Close"] = 120.0
    data.loc[next_day, "Open"] = 100.0
    data.loc[exit_day, "Open"] = 90.0
    X, y, model_return, realized, prices = _supervised_execution_frame(data, horizon=1)
    assert signal_day in X.index
    assert y.loc[signal_day] == 1
    assert model_return.loc[signal_day] > 0
    assert realized.loc[signal_day] < 0
    assert prices.loc[signal_day, "EntryPrice"] == 100.0
    assert prices.loc[signal_day, "ExitPrice"] == 90.0


def test_signal_decision_is_shared_and_blocks_low_quality():
    assert signal_decision(SignalInputs(0.63, 0.02, "WYSOKA"), threshold=0.56) == 1
    assert signal_decision(SignalInputs(0.37, -0.02, "WYSOKA"), threshold=0.56) == -1
    assert signal_decision(SignalInputs(0.70, 0.04, "NISKA — BRAK PRZEWAGI"), threshold=0.56) == 0
    assert signal_decision(SignalInputs(0.63, -0.01, "WYSOKA"), threshold=0.56) == 0
    payload = signal_inputs_from_forecast({"probability_up": 0.61, "expected_return": 0.03, "quality": "UMIARKOWANA"})
    assert payload.source == "ML"
    assert signal_decision(payload, threshold=0.56) == 1


def test_aggregate_validation_keeps_rejected_observations():
    histories = {"AAA": synthetic_data(560), "BBB": synthetic_data(580)}
    config = ValidationConfig(horizons=(1,), initial_train=260, test_size=25, max_folds=1)
    frame = aggregate_validate_histories(histories, markets={"AAA": "USA", "BBB": "ETF"}, config=config)
    assert not frame.empty
    assert set(frame["Symbol"]) == {"AAA", "BBB"}
    assert frame["Fold"].nunique() >= 1
    assert "DecisionReason" in frame
    assert frame["Position"].isin([-1, 0, 1]).all()
    assert (frame["Position"] == 0).any()
    summary = aggregate_summary(frame)
    assert summary["observations"] == len(frame)
    assert summary["rejected"] >= 1
    assert 0 <= summary["auc"] <= 1
    by_market = group_summary(frame, "Market")
    assert set(by_market["Market"]) == {"USA", "ETF"}


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


def test_risk_reward_metrics_prioritize_confirmed_edge():
    forecast = {
        "quality": "WYSOKA", "probability_up": 0.64, "expected_return": 0.06,
        "lower_return": -0.04, "upper_return": 0.14, "auc": 0.63, "brier": 0.22,
    }
    risk = {"annual_volatility": 0.32}
    technical = {"above_sma_50": True, "above_sma_200": True, "near_20d_high": True}
    metrics = risk_reward_metrics(forecast, risk, technical)
    assert metrics["risk_reward"] > 3
    assert metrics["edge_score"] > 4.5
    assert metrics["radar_action"] == "PRIORYTET DO ANALIZY"


def test_setup_intelligence_explains_clean_setup():
    forecast = {
        "quality": "WYSOKA", "probability_up": 0.64, "expected_return": 0.06,
        "lower_return": -0.04, "upper_return": 0.14, "auc": 0.63, "brier": 0.22,
    }
    risk = {"annual_volatility": 0.32, "max_drawdown": -0.18}
    technical = {
        "return_1d": 0.02, "return_5d": 0.08, "return_20d": 0.16, "return_60d": 0.30,
        "momentum_acceleration": 0.04, "rsi_14": 66, "atr_pct": 0.025,
        "relative_volume_20": 0.60, "avg_dollar_volume_20": 25_000_000,
        "drawdown_60": -0.02, "range_position_60": 0.92,
        "above_sma_50": True, "above_sma_200": True, "near_20d_high": True, "near_60d_high": True,
    }
    intelligence = setup_intelligence("TEST", forecast, risk, technical)
    assert intelligence["setup_score"] >= 65
    assert intelligence["setup_grade"].startswith(("A", "B"))
    assert "momentum" in intelligence["thesis"] or "ML" in intelligence["thesis"]


def test_fast_market_scan_creates_non_ml_rows(monkeypatch):
    data = synthetic_data()
    monkeypatch.setattr("market_oracle.engine.download_history", lambda symbol, years=2: data)
    frame, errors = scan_market_fast(["TEST"], horizons=(1, 5), years=2)
    assert errors == {}
    assert set(frame["Horyzont"]) == {1, 5}
    assert set(frame["Tryb analizy"]) == {"FAST"}
    assert "Deep score" in frame
    assert frame["Ocena"].eq("OBSERWUJ").all()


def test_background_monitor_persists_snapshot(tmp_path, monkeypatch):
    sample = pd.DataFrame([{
        "Symbol": "TEST", "Klasa": "USA", "Horyzont": 5, "Ocena": "OBSERWUJ", "Score": 1.5,
        "Deep score": 70.0, "Setup score": 68.0, "Radar score": 5.0,
        "P(wzrost)": 0.52, "Oczekiwany ruch": 0.01, "Tryb analizy": "FAST",
    }])
    ml_sample = sample.copy()
    ml_sample["Tryb analizy"] = "ML"
    monkeypatch.setattr("market_oracle.monitor.scan_market_fast", lambda symbols, horizons, years: (sample, {}))
    monkeypatch.setattr("market_oracle.monitor.scan_market_multi", lambda symbols, horizons, years: (ml_sample, {}))
    path = tmp_path / "signals.json"
    result = run_signal_scan(["TEST"], path=path, deep_limit=1)
    loaded = load_snapshot(path)
    assert result["status"] == "complete"
    assert result["scan_mode"] == "two_stage"
    assert result["shortlist"] == ["TEST"]
    assert loaded["records"][0]["Symbol"] == "TEST"
    assert loaded["records"][0]["Tryb analizy"] == "ML"
    assert len(default_universe()) >= 100


def test_deep_shortlist_keeps_diverse_high_priority_symbols():
    frame = pd.DataFrame([
        {"Symbol": "A", "Klasa": "USA", "Deep score": 90, "Setup score": 80, "Radar score": 5, "Edge score": 3, "Risk control": 70},
        {"Symbol": "B", "Klasa": "USA", "Deep score": 85, "Setup score": 76, "Radar score": 4, "Edge score": 2, "Risk control": 65},
        {"Symbol": "C", "Klasa": "Krypto", "Deep score": 82, "Setup score": 74, "Radar score": 7, "Edge score": 1, "Risk control": 55},
        {"Symbol": "D", "Klasa": "GPW", "Deep score": 78, "Setup score": 71, "Radar score": 3, "Edge score": 2, "Risk control": 60},
    ])
    shortlist = select_deep_shortlist(frame, limit=3)
    assert len(shortlist) == 3
    assert "A" in shortlist
    assert len(set(shortlist)) == len(shortlist)


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
                "Tryb analizy": "ML",
            },
            {
                "Symbol": "FAST", "Klasa": "USA", "Horyzont": 5, "Data": "2026-01-10",
                "Cena": 50.0, "Setup": "Breakout / momentum", "Ocena": "KANDYDAT WZROSTOWY",
                "P(wzrost)": 0.70, "Oczekiwany ruch": 0.05, "Tryb analizy": "FAST",
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
    assert entries[0]["execution"] == "NEXT_OPEN"
    assert entries[0]["signal_price"] == 100.0
    assert entries[0]["entry_price"] is None

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
    assert refreshed[0]["entry_price"] > refreshed[0]["signal_price"]
    assert refreshed[0]["entry_date"] is not None
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
