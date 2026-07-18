import numpy as np
import pandas as pd
import pytest

from market_oracle.backtest import _supervised_execution_frame, walk_forward_backtest
from market_oracle.catalog import CATEGORIES, CRYPTO, CRYPTO_CATEGORIES, ETF_CATEGORIES
from market_oracle.cutoff import available_label_end
from market_oracle.engine import observation_label, risk_reward_metrics, scan_market_fast, setup_intelligence, signal_label
from market_oracle.features import build_features, supervised_frame
from market_oracle.forward import (
    load_candidate_manifest,
    load_forward_events,
    record_snapshot_forward_signals,
    reconstruct_forward_state,
    refresh_forward_ledger,
    forward_summary,
    verify_frozen_hash,
)
from market_oracle.journal import journal_summary, load_journal, paper_portfolio, record_snapshot_signals, refresh_journal_results
from market_oracle.model import fit_forecast, fit_forecast_state
from market_oracle.monitor import default_universe, load_snapshot, run_signal_scan, select_deep_shortlist, snapshot_is_stale
from market_oracle.risk import periods_per_year, risk_metrics
from market_oracle.reality import RealityConfig, reality_check_report, select_non_overlapping_trades
from market_oracle.signals import (
    DEFAULT_SIGNAL_THRESHOLD, SignalInputs, signal_decision, signal_inputs_from_forecast, signal_verdict,
)
from market_oracle.validation import (
    ValidationConfig,
    _fold_ranges,
    aggregate_summary,
    aggregate_validate_histories,
    cost_stress_summary,
    data_fingerprint,
    group_summary,
    save_validation_artifacts,
    validation_report,
)
import market_oracle.validation as validation_module


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


def test_aggregate_validation_keeps_rejected_observations(tmp_path):
    histories = {"AAA": synthetic_data(560), "BBB": synthetic_data(580)}
    config = ValidationConfig(horizons=(1,), initial_train=260, test_size=25, max_folds=1, holdout_size=25, refit_every=25)
    assert config.threshold == DEFAULT_SIGNAL_THRESHOLD
    frame = aggregate_validate_histories(histories, markets={"AAA": "USA", "BBB": "ETF"}, config=config)
    assert not frame.empty
    assert frame.attrs["data_fingerprint"]
    assert set(frame["Symbol"]) == {"AAA", "BBB"}
    assert frame["Fold"].nunique() >= 1
    assert "HOLDOUT" in set(frame["FoldType"])
    assert "DecisionReason" in frame
    assert "TrainEndDate" in frame
    assert "AvailableTrainEndDate" in frame
    assert "RefitEvery" in frame
    assert "CalibrationStartDate" in frame
    assert "AlwaysLongReturn" in frame
    assert "MomentumReturn" in frame
    assert "LinearReturn" in frame
    assert "ActivePositions" not in frame
    assert frame["Position"].isin([-1, 0, 1]).all()
    assert (frame["Position"] == 0).any()
    summary = aggregate_summary(frame)
    assert summary["observations"] == len(frame)
    assert summary["rejected"] >= 1
    assert summary["non_overlapping_trades"] <= summary["trades"]
    assert "benchmark_mean_returns" in summary
    assert "cost_stress" in summary
    assert 0 <= summary["exposure"] <= 1
    assert 0 <= summary["auc"] <= 1
    stress = cost_stress_summary(frame)
    assert set(stress) == {"1x", "2x", "3x"}
    report = validation_report(frame, config, ["AAA", "BBB"], commit_hash="test")
    assert report["manifest"]["commit"] == "test"
    assert report["manifest"]["data_fingerprint"] == frame.attrs["data_fingerprint"]
    assert report["manifest"]["experiment_id"]
    assert report["manifest"]["run_id"].startswith(report["manifest"]["experiment_id"])
    assert report["summary"]["observations"] < report["combined_summary"]["observations"]
    assert report["holdout_summary"]["observations"] > 0
    assert report["by_fold"]
    written = save_validation_artifacts(frame, report, tmp_path)
    written_again = save_validation_artifacts(frame, report, tmp_path)
    assert written_again["records"] != written["records"]
    assert written["records_sha256"]
    assert written["report_sha256"]
    assert pd.read_json(written["manifest_log"], lines=True).iloc[-1]["experiment_id"] == report["manifest"]["experiment_id"]
    fingerprint, ranges = data_fingerprint(histories)
    assert fingerprint == frame.attrs["data_fingerprint"]
    assert ranges["AAA"]["rows"] == len(histories["AAA"])
    context_fingerprint, context_ranges = data_fingerprint(histories, {"AAA": synthetic_data(560)})
    assert context_fingerprint != fingerprint
    assert "context:AAA" in context_ranges
    by_market = group_summary(frame, "Market")
    assert set(by_market["Market"]) == {"USA", "ETF"}

    row = frame[
        (frame["Symbol"] == "AAA")
        & (frame["FoldType"] == "WALK_FORWARD")
        & (frame["TrainEndDate"] == frame["AvailableTrainEndDate"])
    ].iloc[0]
    X, y, model_return, _, _ = _supervised_execution_frame(histories["AAA"], horizon=int(row["Horizon"]))
    train_end = X.index.get_loc(pd.Timestamp(row["TrainEndDate"])) + 1
    state = fit_forecast_state(X.iloc[:train_end], y.iloc[:train_end], model_return.iloc[:train_end], int(row["Horizon"]))
    prediction = state.predict(X.loc[[pd.Timestamp(row["Date"])]])
    verdict = signal_verdict(prediction.signal_inputs(source="TEST"), config.threshold)
    assert np.isclose(row["Probability"], prediction.probability_up)
    assert np.isclose(row["ExpectedReturn"], prediction.expected_return)
    assert np.isclose(row["Skill"], prediction.skill)
    assert np.isclose(row["ValidationAUC"], state.auc)
    assert np.isclose(row["ValidationBrier"], state.brier)
    assert row["Quality"] == state.quality
    assert row["DecisionReason"] == verdict.reason

    raw_cutoff = pd.Timestamp(row["Date"])
    truncated = histories["AAA"].loc[:raw_cutoff]
    latest = build_features(truncated).dropna().loc[[raw_cutoff]]
    X_cut, y_cut, returns_cut = supervised_frame(truncated, int(row["Horizon"]))
    assert X_cut.index[-1] == pd.Timestamp(row["TrainEndDate"])
    production_state = fit_forecast_state(X_cut, y_cut, returns_cut, int(row["Horizon"]))
    production_prediction = production_state.predict(latest)
    production_verdict = signal_verdict(production_prediction.signal_inputs(source="PRODUCTION_TEST"), config.threshold)
    assert np.isclose(row["Probability"], production_prediction.probability_up)
    assert np.isclose(row["ExpectedReturn"], production_prediction.expected_return)
    assert np.isclose(row["Skill"], production_prediction.skill)
    assert np.isclose(row["ValidationAUC"], production_state.auc)
    assert np.isclose(row["ValidationBrier"], production_state.brier)
    assert row["Quality"] == production_state.quality
    assert row["DecisionReason"] == production_verdict.reason

    date_position = X.index.get_loc(pd.Timestamp(row["Date"]))
    assert train_end == available_label_end(date_position, int(row["Horizon"]))


def test_reality_check_filters_overlap_and_builds_daily_curve():
    records = pd.DataFrame([
        {
            "Date": "2024-01-01", "Symbol": "AAA", "Market": "USA", "Horizon": 20, "Fold": 1,
            "Position": 1, "EntryDate": "2024-01-02", "ExitDate": "2024-01-30",
            "EntryPrice": 100.0, "ExitPrice": 106.0, "Return": 0.0585, "RoundTripCost": 0.0015,
            "BuyHoldReturn": 0.06, "ActualUp": 1, "ExecutionUp": 1,
            "Probability": 0.62, "ExpectedReturn": 0.04, "ValidationAUC": 0.67, "ValidationBrier": 0.22,
            "DecisionReason": "LONG_CONFIRMED",
        },
        {
            "Date": "2024-01-05", "Symbol": "AAA", "Market": "USA", "Horizon": 20, "Fold": 1,
            "Position": 1, "EntryDate": "2024-01-08", "ExitDate": "2024-02-05",
            "EntryPrice": 102.0, "ExitPrice": 108.0, "Return": 0.0573, "RoundTripCost": 0.0015,
            "BuyHoldReturn": 0.0588, "ActualUp": 1, "ExecutionUp": 1,
            "Probability": 0.65, "ExpectedReturn": 0.05, "ValidationAUC": 0.69, "ValidationBrier": 0.21,
            "DecisionReason": "LONG_CONFIRMED",
        },
        {
            "Date": "2024-01-02", "Symbol": "BBB", "Market": "ETF", "Horizon": 20, "Fold": 1,
            "Position": 1, "EntryDate": "2024-01-03", "ExitDate": "2024-01-31",
            "EntryPrice": 50.0, "ExitPrice": 51.0, "Return": 0.0185, "RoundTripCost": 0.0015,
            "BuyHoldReturn": 0.02, "ActualUp": 1, "ExecutionUp": 1,
            "Probability": 0.59, "ExpectedReturn": 0.02, "ValidationAUC": 0.61, "ValidationBrier": 0.23,
            "DecisionReason": "LONG_CONFIRMED",
        },
        {
            "Date": "2024-01-31", "Symbol": "AAA", "Market": "USA", "Horizon": 20, "Fold": 2,
            "Position": 1, "EntryDate": "2024-02-01", "ExitDate": "2024-02-29",
            "EntryPrice": 107.0, "ExitPrice": 112.0, "Return": 0.0452, "RoundTripCost": 0.0015,
            "BuyHoldReturn": 0.0467, "ActualUp": 1, "ExecutionUp": 1,
            "Probability": 0.61, "ExpectedReturn": 0.03, "ValidationAUC": 0.66, "ValidationBrier": 0.22,
            "DecisionReason": "LONG_CONFIRMED",
        },
        {
            "Date": "2024-02-01", "Symbol": "CCC", "Market": "CRYPTO", "Horizon": 20, "Fold": 2,
            "Position": 0, "EntryDate": "2024-02-02", "ExitDate": "2024-03-01",
            "EntryPrice": 10.0, "ExitPrice": 9.0, "Return": -0.10, "RoundTripCost": 0.0015,
            "BuyHoldReturn": -0.10, "ActualUp": 0, "ExecutionUp": 0,
            "Probability": 0.51, "ExpectedReturn": 0.0, "ValidationAUC": 0.49, "ValidationBrier": 0.25,
            "DecisionReason": "LOW_QUALITY",
        },
    ])
    dates = pd.bdate_range("2024-01-01", "2024-03-05")
    histories = {
        "AAA": pd.DataFrame({"Open": np.linspace(100, 115, len(dates))}, index=dates),
        "BBB": pd.DataFrame({"Open": np.linspace(50, 52, len(dates))}, index=dates),
        "SPY": pd.DataFrame({"Open": np.linspace(480, 500, len(dates))}, index=dates),
    }
    selected = select_non_overlapping_trades(records, RealityConfig(horizons=(20,)))
    assert list(selected["Symbol"]) == ["AAA", "BBB", "AAA"]
    assert selected["RealityTradeId"].tolist() == [1, 2, 3]

    capped = select_non_overlapping_trades(records, RealityConfig(horizons=(20,), max_positions=1))
    assert list(capped["Symbol"]) == ["AAA", "AAA"]

    report, selected, curve = reality_check_report(
        records,
        histories,
        RealityConfig(horizons=(20,), benchmark_symbol="SPY", strict_history=False),
    )
    assert report["summary"]["observations"] == len(records)
    assert report["summary"]["raw_signals"] == 4
    assert report["summary"]["selected_trades"] == 3
    assert report["summary"]["hit_rate"] == 1.0
    assert report["summary"]["avg_validation_auc"] > 0.6
    assert report["summary"]["exposure_days"] > 0
    assert not curve.empty
    assert curve["Equity"].iloc[-1] > 1
    assert {row["Symbol"] for row in report["by_symbol"]} == {"AAA", "BBB", "CCC"}
    assert next(row for row in report["by_symbol"] if row["Symbol"] == "CCC")["selected_trades"] == 0


def test_reality_check_uses_fixed_slots_costs_and_same_slot_benchmark():
    records = pd.DataFrame([
        {
            "Date": "2024-01-01", "Symbol": "AAA", "Market": "USA", "Horizon": 20, "Fold": 1,
            "Position": 1, "EntryDate": "2024-01-02", "ExitDate": "2024-01-04",
            "EntryPrice": 100.0, "ExitPrice": 121.0, "Return": 0.20, "RoundTripCost": 0.01,
            "BuyHoldReturn": 0.21, "ActualUp": 1, "ExecutionUp": 1,
            "Probability": 0.62, "ExpectedReturn": 0.04, "ValidationAUC": 0.67, "ValidationBrier": 0.22,
            "DecisionReason": "LONG_CONFIRMED",
        },
        {
            "Date": "2024-01-02", "Symbol": "BBB", "Market": "USA", "Horizon": 20, "Fold": 1,
            "Position": 1, "EntryDate": "2024-01-03", "ExitDate": "2024-01-04",
            "EntryPrice": 50.0, "ExitPrice": 45.0, "Return": -0.11, "RoundTripCost": 0.01,
            "BuyHoldReturn": -0.10, "ActualUp": 0, "ExecutionUp": 0,
            "Probability": 0.59, "ExpectedReturn": 0.02, "ValidationAUC": 0.61, "ValidationBrier": 0.23,
            "DecisionReason": "LONG_CONFIRMED",
        },
    ])
    histories = {
        "AAA": pd.DataFrame({"Open": [100.0, 110.0, 121.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])),
        "BBB": pd.DataFrame({"Open": [50.0, 45.0]}, index=pd.to_datetime(["2024-01-03", "2024-01-04"])),
        "SPY": pd.DataFrame({"Open": [200.0, 220.0, 242.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])),
    }
    report, selected, curve = reality_check_report(
        records,
        histories,
        RealityConfig(horizons=(20,), max_positions=2, portfolio_slots=2, benchmark_symbol="SPY"),
    )
    daily = curve.set_index("Date")
    assert selected["Symbol"].tolist() == ["AAA", "BBB"]
    assert np.isclose(daily.loc[pd.Timestamp("2024-01-02"), "StrategyReturn"], -0.005)
    assert daily.loc[pd.Timestamp("2024-01-03"), "StrategyReturn"] == pytest.approx(1.0395 / 0.995 - 1)
    assert daily.loc[pd.Timestamp("2024-01-04"), "StrategyReturn"] == pytest.approx(1.04445 / 1.0395 - 1)
    assert daily.loc[pd.Timestamp("2024-01-02"), "GrossExposure"] == pytest.approx(0.495 / 0.995)
    assert daily.loc[pd.Timestamp("2024-01-03"), "GrossExposure"] == 1.0
    assert curve["Equity"].iloc[-1] == pytest.approx(1.04445)
    assert curve["BenchmarkEquity"].iloc[-1] > curve["Equity"].iloc[-1]
    assert report["summary"]["portfolio_slots"] == 2
    assert report["summary"]["max_active_positions"] == 2
    assert report["summary"]["selection_counts"] == {"SELECTED": 2}
    assert report["price_issues"] == []


def test_reality_check_ledger_does_not_rebalance_winning_slot():
    records = pd.DataFrame([{
        "Date": "2024-01-01", "Symbol": "AAA", "Market": "USA", "Horizon": 20, "Fold": 1,
        "Position": 1, "EntryDate": "2024-01-02", "ExitDate": "2024-01-04",
        "EntryPrice": 100.0, "ExitPrice": 121.0, "Return": 0.21, "RoundTripCost": 0.0,
        "BuyHoldReturn": 0.21, "ActualUp": 1, "ExecutionUp": 1,
        "Probability": 0.62, "ExpectedReturn": 0.04, "ValidationAUC": 0.67, "ValidationBrier": 0.22,
        "DecisionReason": "LONG_CONFIRMED",
    }])
    history = pd.DataFrame({"Open": [100.0, 110.0, 121.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    report, _, curve = reality_check_report(
        records,
        {"AAA": history},
        RealityConfig(horizons=(20,), max_positions=1, portfolio_slots=5, benchmark_symbol=None),
    )
    assert curve["Equity"].iloc[-1] == pytest.approx(0.8 + 0.2 * 1.10 * 1.10)
    assert report["summary"]["portfolio_slots"] == 5
    assert report["summary"]["max_gross_exposure"] == pytest.approx(0.2 * 1.10 / 1.02)


def test_reality_check_uses_session_calendar_and_fails_on_bad_cache():
    records = pd.DataFrame([{
        "Date": "2024-01-04", "Symbol": "AAA", "Market": "USA", "Horizon": 1, "Fold": 1,
        "Position": 1, "EntryDate": "2024-01-05", "ExitDate": "2024-01-08",
        "EntryPrice": 100.0, "ExitPrice": 105.0, "Return": 0.05, "RoundTripCost": 0.0,
        "BuyHoldReturn": 0.05, "ActualUp": 1, "ExecutionUp": 1,
        "Probability": 0.62, "ExpectedReturn": 0.03, "ValidationAUC": 0.62, "ValidationBrier": 0.22,
        "DecisionReason": "LONG_CONFIRMED",
    }])
    history = pd.DataFrame({"Open": [100.0, 105.0]}, index=pd.to_datetime(["2024-01-05", "2024-01-08"]))
    report, _, curve = reality_check_report(
        records,
        {"AAA": history, "SPY": history},
        RealityConfig(horizons=(1,), max_positions=1, portfolio_slots=1, benchmark_symbol="SPY"),
    )
    daily = curve.set_index("Date")
    assert pd.Timestamp("2024-01-06") not in daily.index
    assert pd.Timestamp("2024-01-07") not in daily.index
    assert report["summary"]["annualization_days"] == 252

    with pytest.raises(ValueError, match="MISSING_HISTORY"):
        reality_check_report(records, {}, RealityConfig(horizons=(1,), strict_history=True))

    bad_records = records.copy()
    bad_records.loc[0, "EntryPrice"] = 90.0
    with pytest.raises(ValueError, match="PRICE_MISMATCH"):
        reality_check_report(
            bad_records,
            {"AAA": history},
            RealityConfig(horizons=(1,), benchmark_symbol=None, strict_history=True),
        )


def test_fold_selection_is_evenly_distributed_before_holdout():
    folds = _fold_ranges(length=900, horizon=5, config=ValidationConfig(initial_train=260, test_size=40, max_folds=4, holdout_size=80))
    walk = [fold for fold in folds if fold.fold_type == "WALK_FORWARD"]
    holdout = [fold for fold in folds if fold.fold_type == "HOLDOUT"]
    assert len(walk) == 4
    assert len(holdout) == 1
    assert walk[0].test_start < walk[1].test_start < walk[2].test_start < walk[3].test_start
    assert walk[-1].test_end <= holdout[0].test_start - 5
    assert walk[-1].test_start > walk[1].test_start


def test_validation_uses_all_valid_rows_without_holdout_and_freezes_between_refits(monkeypatch):
    folds = _fold_ranges(
        length=500,
        horizon=5,
        config=ValidationConfig(initial_train=260, test_size=90, max_folds=None, holdout_size=0),
    )
    assert folds[-1].test_end == 500
    assert {fold.fold_type for fold in folds} == {"WALK_FORWARD"}

    class FakeLinear:
        def predict_proba(self, X):
            return np.repeat([[0.45, 0.55]], len(X), axis=0)

    class FakeState:
        def __init__(self, history_end: int):
            self.history_end = history_end
            self.model_train_end = max(1, history_end - 80)
            self.calibration_start = None
            self.calibration_end = None
            self.assessment_start = max(0, history_end - 40)
            self.assessment_end = history_end
            self.class_models = {"linear": FakeLinear()}
            self.quality = "UMIARKOWANA"
            self.auc = 0.60
            self.brier = 0.23

        def predict(self, X):
            from market_oracle.model import ForecastPrediction

            return ForecastPrediction(
                probability_up=0.58,
                expected_return=0.02,
                lower_return=-0.03,
                upper_return=0.05,
                raw_probability=0.58,
                raw_expected_return=0.02,
                skill=0.5,
                quality=self.quality,
                auc=self.auc,
                brier=self.brier,
            )

    train_lengths = []

    def fake_fit_state(X, y, returns, horizon):
        train_lengths.append(len(X))
        return FakeState(len(X))

    monkeypatch.setattr(validation_module, "fit_forecast_state", fake_fit_state)
    config = ValidationConfig(
        horizons=(1,), initial_train=260, test_size=20, max_folds=1, holdout_size=0, refit_every=5,
    )
    observed_records = []
    observed_refits = []
    frame = validation_module.validate_history(
        "AAA",
        synthetic_data(620),
        market="USA",
        config=config,
        record_callback=observed_records.append,
        refit_callback=observed_refits.append,
    )
    walk = frame[frame["FoldType"] == "WALK_FORWARD"].sort_values("Date").reset_index(drop=True)
    assert len(walk) == 20
    assert len(observed_records) == len(walk)
    assert [event["available_train_end"] for event in observed_refits[:4]] == [260, 265, 270, 275]
    assert train_lengths[:4] == [260, 265, 270, 275]
    assert walk.loc[:4, "TrainEndDate"].nunique() == 1
    assert walk.loc[0, "TrainEndDate"] == walk.loc[0, "AvailableTrainEndDate"]
    assert walk.loc[4, "TrainEndDate"] != walk.loc[4, "AvailableTrainEndDate"]
    assert walk.loc[5, "TrainEndDate"] == walk.loc[5, "AvailableTrainEndDate"]


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


def test_candidate_manifest_is_frozen_and_hash_verified():
    manifest = load_candidate_manifest()
    assert manifest["candidate_id"] == "marketscope_20d_long_candidate_v1"
    assert manifest["frozen_commit"] == "60f0a8b"
    assert manifest["decision_contract"]["threshold"] == DEFAULT_SIGNAL_THRESHOLD
    assert manifest["scope"]["horizon_sessions"] == 20
    assert manifest["portfolio_contract"]["portfolio_slots"] == 5
    assert manifest["portfolio_contract"]["max_positions"] == 5
    assert verify_frozen_hash(manifest, "manifest_hash")


def test_forward_ledger_records_only_candidate_rows_and_is_append_only(tmp_path):
    snapshot = {
        "status": "complete",
        "schema_version": 6,
        "updated_at": "2026-01-10T21:00:00+00:00",
        "records": [
            {
                "Symbol": "GOOD", "Klasa": "USA / ETF", "Horyzont": 20, "Data": "2026-01-10",
                "Cena": 100.0, "Setup": "Trend continuation",
                "Ocena": "KANDYDAT WZROSTOWY", "P(wzrost)": 0.61,
                "Oczekiwany ruch": 0.03, "AUC walidacji": 0.62,
                "Brier": 0.22, "Jakość modelu": "WYSOKA", "Score": 4.2,
                "Tryb analizy": "ML",
            },
            {
                "Symbol": "FAST", "Klasa": "USA / ETF", "Horyzont": 20, "Data": "2026-01-10",
                "Cena": 50.0, "Ocena": "KANDYDAT WZROSTOWY", "P(wzrost)": 0.70,
                "Oczekiwany ruch": 0.05, "Jakość modelu": "WYSOKA", "Tryb analizy": "FAST",
            },
            {
                "Symbol": "BTC-USD", "Klasa": "Krypto", "Horyzont": 20, "Data": "2026-01-10",
                "Cena": 100_000.0, "Ocena": "KANDYDAT WZROSTOWY", "P(wzrost)": 0.70,
                "Oczekiwany ruch": 0.05, "Jakość modelu": "WYSOKA", "Tryb analizy": "ML",
            },
            {
                "Symbol": "PKO.WA", "Klasa": "GPW", "Horyzont": 20, "Data": "2026-01-10",
                "Cena": 70.0, "Ocena": "KANDYDAT WZROSTOWY", "P(wzrost)": 0.70,
                "Oczekiwany ruch": 0.05, "Jakość modelu": "WYSOKA", "Tryb analizy": "ML",
            },
            {
                "Symbol": "SHORTER", "Klasa": "USA / ETF", "Horyzont": 5, "Data": "2026-01-10",
                "Cena": 80.0, "Ocena": "KANDYDAT WZROSTOWY", "P(wzrost)": 0.70,
                "Oczekiwany ruch": 0.05, "Jakość modelu": "WYSOKA", "Tryb analizy": "ML",
            },
        ],
    }
    path = tmp_path / "forward.jsonl"
    assert record_snapshot_forward_signals(snapshot, path=path) == 1
    assert record_snapshot_forward_signals(snapshot, path=path) == 0
    events = load_forward_events(path)
    assert len(events) == 1
    assert events[0]["event_type"] == "SIGNAL_OBSERVED"
    assert events[0]["status"] == "PENDING"
    assert events[0]["symbol"] == "GOOD"
    assert events[0]["horizon"] == 20
    assert events[0]["direction"] == "LONG"
    assert events[0]["execution"] == "NEXT_OPEN"
    assert events[0]["probability_up"] == 0.61
    assert verify_frozen_hash(load_candidate_manifest(), "manifest_hash")


def test_forward_ledger_fills_next_open_and_closes_after_horizon(tmp_path):
    snapshot = {
        "status": "complete",
        "schema_version": 6,
        "updated_at": "2026-01-02T21:00:00+00:00",
        "records": [{
            "Symbol": "GOOD", "Klasa": "USA / ETF", "Horyzont": 20, "Data": "2026-01-02",
            "Cena": 100.0, "Setup": "Trend continuation",
            "Ocena": "SILNY KANDYDAT WZROSTOWY", "P(wzrost)": 0.66,
            "Oczekiwany ruch": 0.04, "AUC walidacji": 0.67,
            "Brier": 0.21, "Jakość modelu": "WYSOKA", "Score": 5.0,
            "Tryb analizy": "ML",
        }],
    }
    path = tmp_path / "forward.jsonl"
    assert record_snapshot_forward_signals(snapshot, path=path) == 1
    dates = pd.bdate_range("2026-01-02", periods=26)
    prices = np.linspace(99.0, 125.0, len(dates))
    prices[1] = 100.0
    prices[21] = 110.0
    history = pd.DataFrame({
        "Open": prices,
        "High": prices + 1,
        "Low": prices - 1,
        "Close": prices,
        "Volume": 1_000_000,
    }, index=dates)
    events, state, errors = refresh_forward_ledger(path=path, histories={"GOOD": history})
    assert errors == {}
    assert len(events) == 3
    event_types = [event["event_type"] for event in events]
    assert event_types == ["SIGNAL_OBSERVED", "ENTRY_FILLED", "POSITION_CLOSED"]
    signal_id = events[0]["signal_id"]
    assert state[signal_id]["status"] == "CLOSED"
    assert state[signal_id]["entry_date"] == "2026-01-05"
    assert state[signal_id]["exit_date"] == "2026-02-02"
    assert state[signal_id]["entry_price"] == 100.0
    assert state[signal_id]["exit_price"] == 110.0
    assert state[signal_id]["gross_return"] == pytest.approx(0.10)
    assert state[signal_id]["strategy_return"] == pytest.approx(0.10 - 0.0015)
    assert state[signal_id]["hit"] is True

    events_again, state_again, errors_again = refresh_forward_ledger(path=path, histories={"GOOD": history})
    assert errors_again == {}
    assert len(events_again) == 3
    assert reconstruct_forward_state(events_again) == state_again
    summary = forward_summary(events_again)
    assert summary["signals"] == 1
    assert summary["closed"] == 1
    assert summary["closed_hit_rate"] == 1.0
