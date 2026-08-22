import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_oracle.backtest import _supervised_execution_frame, walk_forward_backtest
from market_oracle.catalog import CATEGORIES, CRYPTO, CRYPTO_CATEGORIES, ETF_CATEGORIES
from market_oracle.cutoff import available_label_end
from market_oracle.engine import observation_label, risk_reward_metrics, scan_market_fast, setup_intelligence, signal_label
from market_oracle.evidence import (
    EvidenceRegistryError,
    EvidenceScope,
    ForecastAvailability,
    ForwardEvidence,
    HistoricalEvidence,
    evidence_copy,
    load_evidence_registry,
    registry_hash,
    resolve_evidence,
    validate_evidence_registry,
    verify_evidence_sources,
)
from market_oracle.features import build_features, supervised_frame
from market_oracle.candidate import build_candidate_snapshot, run_candidate_forward_cycle
from market_oracle.forward import (
    assert_forward_contract_ready,
    append_forward_event,
    build_forward_cockpit,
    format_forward_cli_summary,
    load_forward_cockpit,
    load_forward_universe,
    load_candidate_manifest,
    load_forward_events,
    load_unseen_universe,
    record_snapshot_forward_signals,
    reconstruct_forward_state,
    refresh_forward_ledger,
    forward_summary,
    snapshot_hash,
    verify_frozen_hash,
    verify_pipeline_contract,
)
import market_oracle.forward as forward_module
from market_oracle.auto_forward import (
    AutomationConfig,
    automation_lock,
    build_automation_plan,
    eligible_session_dates,
    execute_automation,
    launchd_status,
    launchd_plist_payload,
    nyse_full_holidays,
)
import market_oracle.auto_forward as auto_forward_module
from market_oracle.journal import journal_summary, load_journal, paper_portfolio, record_snapshot_signals, refresh_journal_results
from market_oracle.model import fit_forecast, fit_forecast_state
from market_oracle.monitor import default_universe, load_snapshot, run_signal_scan, select_deep_shortlist, snapshot_is_stale
from market_oracle.presentation import build_analysis_report, build_start_guidance
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
from market_oracle.watchlist import (
    archive_watch_item,
    compare_watch_item_to_current,
    load_watchlist,
    upsert_watch_item,
    watch_item_current_snapshot,
    watch_item_from_analysis,
    watch_item_lifecycle,
    watchlist_analysis_matches_selection,
    watchlist_summary,
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


def load_aggregate_model_view():
    source_path = Path(__file__).resolve().parents[1] / "app.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "aggregate_model_view"
    )
    namespace = {
        "horizon_text": lambda horizon, crypto: f"{horizon} {'dni' if crypto else 'sesji'}",
        "pct": lambda value: f"{value:.1%}",
        "signed_pct": lambda value: f"{value:+.1%}",
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["aggregate_model_view"]


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
                "Tryb analizy": "ML", "DecisionReason": "LONG_CONFIRMED",
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


def test_watchlist_upsert_deduplicates_active_symbol_horizon(tmp_path):
    path = tmp_path / "watchlist.json"
    item = {
        "symbol": "xtb.wa",
        "horizon": 20,
        "source": "ML",
        "verdict": "LONG_CONFIRMED",
        "probability_up": 0.66,
        "expected_return": 0.035,
        "quality": "WYSOKA",
    }

    first, created = upsert_watch_item(item, path=path)
    duplicate, created_again = upsert_watch_item({**item, "probability_up": 0.70}, path=path)

    assert created is True
    assert created_again is False
    assert first["id"] == duplicate["id"]
    assert first["symbol"] == "XTB.WA"
    assert len(load_watchlist(path)) == 1
    assert watchlist_summary(load_watchlist(path))["active"] == 1


def test_watchlist_archive_allows_fresh_observation(tmp_path):
    path = tmp_path / "watchlist.json"
    first, _ = upsert_watch_item({"symbol": "SPY", "horizon": 20}, path=path)

    assert archive_watch_item(first["id"], path=path, archived_at="2026-08-10T22:00:00+02:00") is True
    second, created = upsert_watch_item({"symbol": "SPY", "horizon": 20}, path=path)

    entries = load_watchlist(path)
    assert created is True
    assert first["id"] != second["id"]
    assert watchlist_summary(entries) == {"total": 2, "active": 1, "archived": 1, "symbols": 1}


def test_watchlist_item_from_analysis_preserves_seen_snapshot():
    result = {
        "symbol": "XTB.WA",
        "last_date": pd.Timestamp("2026-08-07"),
        "last_price": 168.0,
        "benchmark": "ETFBW20TR.WA",
        "forecasts": {
            20: {"probability_up": 0.658, "expected_return": 0.035, "quality": "WYSOKA"},
        },
    }
    report = {
        "symbol": "XTB.WA",
        "primary_horizon": 20,
        "headline": "XTB.WA: potwierdzony kandydat wzrostowy na horyzoncie 20 sesji.",
        "verdict": {"label": "LONG", "reason": "LONG_CONFIRMED", "decision": 1},
        "evidence": ["Horyzont roboczy raportu: 20 sesji; jakość walidacji: WYSOKA."],
        "counterpoints": ["RSI wysoko — możliwe przegrzanie."],
        "freshness": {"radar": "2026-08-07 20:26", "analysis": "2026-08-07", "benchmark": "ETFBW20TR.WA"},
    }

    item = watch_item_from_analysis(result, report, {"radar_updated_at": "2026-08-07T20:26:00+02:00"})

    assert item["symbol"] == "XTB.WA"
    assert item["horizon"] == 20
    assert item["verdict"] == "LONG_CONFIRMED"
    assert item["probability_up"] == pytest.approx(0.658)
    assert item["expected_return"] == pytest.approx(0.035)
    assert item["risk_note"] == "RSI wysoko — możliwe przegrzanie."
    assert item["data_as_of"] == "2026-08-07"
    assert item["asset_class"] == "GPW"
    assert item["calendar_kind"] == "GPW"


def test_watchlist_analysis_must_match_selected_item():
    xtb_item = {"id": "watch-xtb", "symbol": "XTB.WA", "horizon": 20}
    spy_item = {"id": "watch-spy", "symbol": "SPY", "horizon": 20}
    saved_xtb = {
        "item_id": "watch-xtb",
        "symbol": "XTB.WA",
        "horizon": 20,
        "years": 5,
        "result": {"symbol": "XTB.WA"},
    }

    assert watchlist_analysis_matches_selection(saved_xtb, xtb_item, current_years=5) is True
    assert watchlist_analysis_matches_selection(saved_xtb, spy_item, current_years=5) is False
    assert watchlist_analysis_matches_selection(saved_xtb, xtb_item, current_years=3) is False
    assert watchlist_analysis_matches_selection({**saved_xtb, "item_id": ""}, xtb_item, current_years=5) is False


def test_watchlist_current_snapshot_uses_saved_horizon_and_shared_gate():
    item = {"id": "watch-xtb", "symbol": "XTB.WA", "horizon": 20}
    result = {
        "symbol": "XTB.WA",
        "last_date": pd.Timestamp("2026-08-11"),
        "last_price": 170.0,
        "forecasts": {
            5: {"probability_up": 0.90, "expected_return": 0.10, "quality": "WYSOKA", "auc": 0.80, "brier": 0.10},
            20: {"probability_up": 0.60, "expected_return": 0.02, "quality": "WYSOKA", "auc": 0.70, "brier": 0.20},
        },
    }

    current = watch_item_current_snapshot(result, item)

    assert current["available"] is True
    assert current["symbol"] == "XTB.WA"
    assert current["horizon"] == 20
    assert current["probability_up"] == pytest.approx(0.60)
    assert current["expected_return"] == pytest.approx(0.02)
    assert current["verdict"] == "LONG_CONFIRMED"

    conflict = watch_item_current_snapshot(
        {
            **result,
            "forecasts": {
                20: {"probability_up": 0.63, "expected_return": -0.01, "quality": "WYSOKA", "auc": 0.70, "brier": 0.20},
            },
        },
        item,
    )

    assert conflict["verdict"] == "EXPECTED_RETURN_CONFLICT"
    assert conflict["verdict_decision"] == 0


def test_watchlist_current_snapshot_rejects_wrong_symbol_or_missing_horizon():
    item = {"symbol": "XTB.WA", "horizon": 20}

    wrong_symbol = watch_item_current_snapshot({"symbol": "SPY", "forecasts": {20: {"probability_up": 0.60}}}, item)
    missing_horizon = watch_item_current_snapshot(
        {"symbol": "XTB.WA", "forecasts": {5: {"probability_up": 0.60}}},
        item,
    )

    assert wrong_symbol["available"] is False
    assert wrong_symbol["reason"] == "SYMBOL_MISMATCH"
    assert missing_horizon["available"] is False
    assert missing_horizon["reason"] == "HORIZON_NOT_AVAILABLE"


def test_watchlist_comparison_uses_verdict_transition_not_delta_thresholds():
    item = {
        "symbol": "XTB.WA",
        "horizon": 20,
        "created_at": "2026-08-03T10:00:00+02:00",
        "verdict_label": "LONG",
        "verdict_decision": 1,
        "probability_up": 0.66,
        "expected_return": 0.05,
        "quality": "WYSOKA",
    }
    still_long = {
        "available": True,
        "symbol": "XTB.WA",
        "horizon": 20,
        "verdict_label": "LONG",
        "verdict_decision": 1,
        "probability_up": 0.56,
        "expected_return": 0.01,
        "quality": "WYSOKA",
    }

    comparison = compare_watch_item_to_current(item, still_long, now="2026-08-05T10:00:00+02:00")

    assert comparison["comparison_status"] == "STILL_CONFIRMED"
    assert comparison["delta_probability"] == pytest.approx(-0.10)
    assert comparison["delta_expected_return"] == pytest.approx(-0.04)

    weakened = {**still_long, "verdict_label": "OBSERWUJ", "verdict_decision": 0}
    neutral_item = {**item, "verdict_label": "OBSERWUJ", "verdict_decision": 0}
    reversed_now = {**still_long, "verdict_label": "SHORT", "verdict_decision": -1}

    assert compare_watch_item_to_current(item, weakened)["comparison_status"] == "WEAKENED"
    assert compare_watch_item_to_current(neutral_item, still_long)["comparison_status"] == "GAINED_CONFIRMATION"
    assert compare_watch_item_to_current(item, reversed_now)["comparison_status"] == "REVERSED"


def test_watchlist_lifecycle_expiry_is_separate_from_current_verdict():
    item = {
        "symbol": "XTB.WA",
        "horizon": 5,
        "created_at": "2026-07-01T10:00:00+02:00",
        "verdict_label": "LONG",
        "verdict_decision": 1,
    }
    current = {
        "available": True,
        "symbol": "XTB.WA",
        "horizon": 5,
        "verdict_label": "LONG",
        "verdict_decision": 1,
    }

    lifecycle = watch_item_lifecycle(item, now="2026-07-15T10:00:00+02:00")
    comparison = compare_watch_item_to_current(item, current, now="2026-07-15T10:00:00+02:00")

    assert lifecycle["status"] == "HORIZON_ENDED"
    assert comparison["comparison_status"] == "STILL_CONFIRMED"
    assert comparison["lifecycle"]["status"] == "HORIZON_ENDED"


def test_watchlist_lifecycle_crypto_uses_7_day_calendar_and_data_anchor():
    item = {
        "symbol": "BTC-USD",
        "horizon": 2,
        "data_as_of": "2026-08-07",
        "created_at": "2026-08-11T00:00:00+02:00",
        "asset_class": "Krypto",
        "calendar_kind": "CRYPTO_24_7",
        "verdict_label": "LONG",
        "verdict_decision": 1,
    }

    lifecycle = watch_item_lifecycle(item, now="2026-08-09T12:00:00+02:00")

    assert lifecycle["status"] == "HORIZON_ENDED"
    assert lifecycle["elapsed"] == 2
    assert lifecycle["remaining"] == 0
    assert lifecycle["unit"] == "dni"
    assert lifecycle["calendar_kind"] == "CRYPTO_24_7"
    assert lifecycle["anchor_source"] == "data_as_of"
    assert lifecycle["is_approximate"] is False


def test_watchlist_lifecycle_nyse_skips_exchange_holiday():
    item = {
        "symbol": "SPY",
        "horizon": 1,
        "data_as_of": "2026-07-02",
        "created_at": "2026-07-06T00:00:00+02:00",
        "asset_class": "USA / ETF",
        "calendar_kind": "NYSE",
    }

    lifecycle = watch_item_lifecycle(item, now="2026-07-06T12:00:00+02:00")

    assert lifecycle["status"] == "HORIZON_ENDED"
    assert lifecycle["elapsed"] == 1
    assert lifecycle["remaining"] == 0
    assert lifecycle["unit"] == "sesji"
    assert lifecycle["calendar_kind"] == "NYSE"
    assert lifecycle["anchor_source"] == "data_as_of"


def test_watchlist_lifecycle_gpw_skips_official_exchange_holiday():
    item = {
        "symbol": "XTB.WA",
        "horizon": 1,
        "data_as_of": "2026-12-23",
        "created_at": "2026-12-24T10:00:00+01:00",
        "asset_class": "GPW",
        "calendar_kind": "GPW",
    }

    christmas_eve = watch_item_lifecycle(item, now="2026-12-24T12:00:00+01:00")
    year_end = watch_item_lifecycle(item, now="2026-12-31T12:00:00+01:00")
    first_counted_session = watch_item_lifecycle(item, now="2026-12-28T12:00:00+01:00")
    good_friday = watch_item_lifecycle(
        {**item, "data_as_of": "2026-04-02", "created_at": "2026-04-03T10:00:00+02:00"},
        now="2026-04-03T12:00:00+02:00",
    )

    assert christmas_eve["status"] == "ACTIVE"
    assert christmas_eve["elapsed"] == 0
    assert christmas_eve["remaining"] == 1
    assert christmas_eve["unit"] == "sesji"
    assert christmas_eve["calendar_kind"] == "GPW"
    assert christmas_eve["anchor_source"] == "data_as_of"
    assert christmas_eve["is_approximate"] is False
    assert year_end["elapsed"] == 3
    assert good_friday["status"] == "ACTIVE"
    assert good_friday["elapsed"] == 0
    assert first_counted_session["status"] == "HORIZON_ENDED"
    assert first_counted_session["elapsed"] == 1
    assert first_counted_session["remaining"] == 0


def test_watchlist_lifecycle_gpw_unknown_outside_verified_calendar():
    item = {
        "symbol": "XTB.WA",
        "horizon": 5,
        "data_as_of": "2029-01-02",
        "asset_class": "GPW",
        "calendar_kind": "GPW",
    }

    lifecycle = watch_item_lifecycle(item, now="2029-01-10T12:00:00+01:00")

    assert lifecycle["status"] == "UNKNOWN"
    assert lifecycle["elapsed"] is None
    assert lifecycle["remaining"] is None
    assert lifecycle["is_approximate"] is True
    assert lifecycle["calendar_kind"] == "GPW"
    assert lifecycle["calendar_label"] == "sesje GPW"


def test_watchlist_lifecycle_unknown_calendar_does_not_fake_sessions():
    item = {
        "symbol": "ABC.L",
        "horizon": 5,
        "created_at": "2026-08-01T10:00:00+02:00",
        "calendar_kind": "UNKNOWN",
    }

    lifecycle = watch_item_lifecycle(item, now="2026-08-05T10:00:00+02:00")

    assert lifecycle["status"] == "UNKNOWN"
    assert lifecycle["elapsed"] is None
    assert lifecycle["remaining"] is None
    assert lifecycle["is_approximate"] is True
    assert lifecycle["calendar_label"] == "kalendarz nieznany"


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
    forward_universe = load_forward_universe()
    unseen = load_unseen_universe()
    assert manifest["candidate_id"] == "marketscope_20d_long_candidate_v1"
    assert manifest["frozen_commit"] == "60f0a8b"
    assert manifest["decision_contract"]["threshold"] == DEFAULT_SIGNAL_THRESHOLD
    assert manifest["scope"]["horizon_sessions"] == 20
    assert manifest["portfolio_contract"]["portfolio_slots"] == 5
    assert manifest["portfolio_contract"]["max_positions"] == 5
    assert verify_frozen_hash(manifest, "manifest_hash")
    assert verify_pipeline_contract(manifest)
    assert verify_frozen_hash(forward_universe, "universe_hash")
    assert forward_universe["symbols"] == ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
    assert verify_frozen_hash(unseen, "universe_hash")
    assert len(unseen["symbols"]) == 30


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
                "Tryb analizy": "ML", "DecisionReason": "LONG_CONFIRMED",
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
    assert record_snapshot_forward_signals(
        snapshot,
        path=path,
        allow_historical=True,
        enforce_pipeline=False,
        require_clean_tree=False,
        require_closed_bar=False,
        require_full_universe=False,
    ) == 1
    assert record_snapshot_forward_signals(
        snapshot,
        path=path,
        allow_historical=True,
        enforce_pipeline=False,
        require_clean_tree=False,
        require_closed_bar=False,
        require_full_universe=False,
    ) == 0
    events = load_forward_events(path)
    assert len(events) == 3
    assert [event["event_type"] for event in events] == ["SNAPSHOT_AUDIT", "SIGNAL_OBSERVED", "POSITION_ACCEPTED"]
    assert events[0]["candidate_rows"] == 1
    assert events[1]["status"] == "OBSERVED"
    assert events[1]["symbol"] == "GOOD"
    assert events[1]["horizon"] == 20
    assert events[1]["direction"] == "LONG"
    assert events[1]["execution"] == "NEXT_OPEN"
    assert events[1]["probability_up"] == 0.61
    assert events[1]["decision_reason"] == "LONG_CONFIRMED"
    assert events[1]["decision_reason_source"] == "SNAPSHOT_EXPLICIT"
    assert events[2]["status"] == "ACCEPTED"
    assert events[2]["slot"] == 1
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
            "Tryb analizy": "ML", "DecisionReason": "LONG_CONFIRMED",
        }],
    }
    path = tmp_path / "forward.jsonl"
    assert record_snapshot_forward_signals(
        snapshot,
        path=path,
        allow_historical=True,
        enforce_pipeline=False,
        require_clean_tree=False,
        require_closed_bar=False,
        require_full_universe=False,
    ) == 1
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
    events, state, errors = refresh_forward_ledger(
        path=path,
        histories={"GOOD": history},
        enforce_pipeline=False,
        require_clean_tree=False,
    )
    assert errors == {}
    assert len(events) == 5
    event_types = [event["event_type"] for event in events]
    assert event_types == ["SNAPSHOT_AUDIT", "SIGNAL_OBSERVED", "POSITION_ACCEPTED", "ENTRY_FILLED", "POSITION_CLOSED"]
    signal_id = events[1]["signal_id"]
    assert state[signal_id]["status"] == "CLOSED"
    assert state[signal_id]["slot"] == 1
    assert state[signal_id]["entry_date"] == "2026-01-05"
    assert state[signal_id]["exit_date"] == "2026-02-02"
    assert state[signal_id]["entry_price"] == 100.0
    assert state[signal_id]["exit_price"] == 110.0
    assert state[signal_id]["gross_return"] == pytest.approx(0.10)
    assert state[signal_id]["strategy_return"] == pytest.approx(0.10 - 0.0015)
    assert state[signal_id]["hit"] is True

    events_again, state_again, errors_again = refresh_forward_ledger(
        path=path,
        histories={"GOOD": history},
        enforce_pipeline=False,
        require_clean_tree=False,
    )
    assert errors_again == {}
    assert len(events_again) == 5
    assert reconstruct_forward_state(events_again) == state_again
    summary = forward_summary(events_again)
    assert summary["signals"] == 1
    assert summary["closed"] == 1
    assert summary["closed_hit_rate"] == 1.0


def test_forward_ledger_rejects_old_and_unclosed_snapshots(tmp_path):
    old_snapshot = {
        "status": "complete",
        "schema_version": 6,
        "updated_at": "2026-07-17T23:00:00+02:00",
        "records": [{
            "Symbol": "GOOD", "Klasa": "USA / ETF", "Horyzont": 20, "Data": "2026-07-17",
            "Cena": 100.0, "Ocena": "KANDYDAT WZROSTOWY", "P(wzrost)": 0.61,
            "Oczekiwany ruch": 0.03, "AUC walidacji": 0.62, "Brier": 0.22,
            "Jakość modelu": "WYSOKA", "Tryb analizy": "ML", "DecisionReason": "LONG_CONFIRMED",
        }],
    }
    with pytest.raises(ValueError, match="older than Candidate"):
        record_snapshot_forward_signals(
            old_snapshot,
            path=tmp_path / "old.jsonl",
            enforce_pipeline=False,
            require_clean_tree=False,
            require_full_universe=False,
        )

    intraday_snapshot = {
        **old_snapshot,
        "updated_at": "2026-07-20T20:30:00+02:00",
        "records": [{**old_snapshot["records"][0], "Data": "2026-07-20"}],
    }
    with pytest.raises(ValueError, match="before the daily bar"):
        record_snapshot_forward_signals(
            intraday_snapshot,
            path=tmp_path / "intraday.jsonl",
            enforce_pipeline=False,
            require_clean_tree=False,
            require_full_universe=False,
        )


def test_forward_ledger_requires_full_universe_and_explicit_reason(tmp_path):
    universe = load_forward_universe()
    row = {
        "Symbol": "AAPL", "Klasa": "USA / ETF", "Horyzont": 20, "Data": "2026-07-20",
        "Cena": 100.0, "Ocena": "KANDYDAT WZROSTOWY", "P(wzrost)": 0.61,
        "Oczekiwany ruch": 0.03, "AUC walidacji": 0.62, "Brier": 0.22,
        "Jakość modelu": "WYSOKA", "Tryb analizy": "ML", "DecisionReason": "LONG_CONFIRMED",
    }
    snapshot = {
        "status": "complete",
        "schema_version": 1,
        "scan_mode": "candidate_v1_full_ml",
        "updated_at": "2026-07-20T22:30:00+02:00",
        "records": [row],
    }
    with pytest.raises(ValueError, match="forward universe hash"):
        record_snapshot_forward_signals(
            snapshot,
            path=tmp_path / "missing_universe.jsonl",
            enforce_pipeline=False,
            require_clean_tree=False,
        )

    full_snapshot = {
        **snapshot,
        "forward_universe": {
            "universe_id": universe["universe_id"],
            "universe_hash": universe["universe_hash"],
            "requested_symbols": universe["symbols"],
            "completed_symbols": universe["symbols"][:-1],
            "failed_symbols": [universe["symbols"][-1]],
            "full_coverage": False,
        },
    }
    with pytest.raises(ValueError, match="failed Candidate"):
        record_snapshot_forward_signals(
            full_snapshot,
            path=tmp_path / "failed_universe.jsonl",
            enforce_pipeline=False,
            require_clean_tree=False,
        )

    no_full_coverage_snapshot = {
        **full_snapshot,
        "forward_universe": {
            **full_snapshot["forward_universe"],
            "completed_symbols": universe["symbols"],
            "failed_symbols": [],
            "full_coverage": False,
        },
    }
    with pytest.raises(ValueError, match="full_coverage"):
        record_snapshot_forward_signals(
            no_full_coverage_snapshot,
            path=tmp_path / "no_full_coverage.jsonl",
            enforce_pipeline=False,
            require_clean_tree=False,
        )

    missing_reason_snapshot = {
        **full_snapshot,
        "forward_universe": {
            **full_snapshot["forward_universe"],
            "completed_symbols": universe["symbols"],
            "failed_symbols": [],
            "full_coverage": True,
        },
        "records": [{key: value for key, value in row.items() if key != "DecisionReason"}],
    }
    with pytest.raises(ValueError, match="missing explicit DecisionReason"):
        record_snapshot_forward_signals(
            missing_reason_snapshot,
            path=tmp_path / "missing_reason.jsonl",
            enforce_pipeline=False,
            require_clean_tree=False,
        )


def test_forward_contract_blocks_dirty_tree_and_changed_pipeline(monkeypatch):
    manifest = load_candidate_manifest()
    monkeypatch.setattr(forward_module, "git_dirty_paths", lambda: [" M market_oracle/model.py"])
    with pytest.raises(ValueError, match="dirty"):
        assert_forward_contract_ready(manifest, enforce_pipeline=False, require_clean_tree=True)

    monkeypatch.setattr(forward_module, "git_dirty_paths", lambda: [])
    original = forward_module.pipeline_fingerprint

    def changed_pipeline():
        payload = original()
        payload["pipeline_hash"] = "changed"
        return payload

    monkeypatch.setattr(forward_module, "pipeline_fingerprint", changed_pipeline)
    with pytest.raises(ValueError, match="pipeline hash mismatch"):
        assert_forward_contract_ready(manifest, enforce_pipeline=True, require_clean_tree=False)


def test_forward_ledger_portfolio_gate_and_priority_are_frozen(tmp_path):
    rows = []
    for symbol, probability in [("CCC", 0.90), ("AAA", 0.56), ("BBB", 0.70), ("DDD", 0.60), ("EEE", 0.59), ("FFF", 0.95)]:
        rows.append({
            "Symbol": symbol, "Klasa": "USA / ETF", "Horyzont": 20, "Data": "2026-07-20",
            "Cena": 100.0, "Ocena": "KANDYDAT WZROSTOWY", "P(wzrost)": probability,
            "Oczekiwany ruch": 0.03, "AUC walidacji": 0.62, "Brier": 0.22,
            "Jakość modelu": "WYSOKA", "Tryb analizy": "ML", "DecisionReason": "LONG_CONFIRMED",
        })
    snapshot = {
        "status": "complete",
        "schema_version": 6,
        "updated_at": "2026-07-20T22:30:00+02:00",
        "records": rows,
    }
    path = tmp_path / "portfolio.jsonl"
    assert record_snapshot_forward_signals(
        snapshot,
        path=path,
        enforce_pipeline=False,
        require_clean_tree=False,
        require_full_universe=False,
    ) == 6
    events = load_forward_events(path)
    accepted = [event for event in events if event["event_type"] == "POSITION_ACCEPTED"]
    skipped = [event for event in events if event["event_type"] == "POSITION_SKIPPED"]
    assert [event["symbol"] for event in accepted] == ["AAA", "BBB", "CCC", "DDD", "EEE"]
    assert [event["slot"] for event in accepted] == [1, 2, 3, 4, 5]
    assert skipped[0]["symbol"] == "FFF"
    assert skipped[0]["skip_reason"] == "POSITION_SKIPPED_NO_FREE_SLOT"


def test_forward_ledger_hash_chain_detects_tampering(tmp_path):
    snapshot = {
        "status": "complete",
        "schema_version": 6,
        "updated_at": "2026-07-20T22:30:00+02:00",
        "records": [{
            "Symbol": "GOOD", "Klasa": "USA / ETF", "Horyzont": 20, "Data": "2026-07-20",
            "Cena": 100.0, "Ocena": "KANDYDAT WZROSTOWY", "P(wzrost)": 0.61,
            "Oczekiwany ruch": 0.03, "AUC walidacji": 0.62, "Brier": 0.22,
            "Jakość modelu": "WYSOKA", "Tryb analizy": "ML", "DecisionReason": "LONG_CONFIRMED",
        }],
    }
    path = tmp_path / "tamper.jsonl"
    record_snapshot_forward_signals(
        snapshot,
        path=path,
        enforce_pipeline=False,
        require_clean_tree=False,
        require_full_universe=False,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = lines[1].replace('"signal_price":100.0', '"signal_price":101.0')
    path.write_text("\n".join([lines[0], tampered, *lines[2:]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="event hash mismatch"):
        load_forward_events(path)


def test_forward_cockpit_reads_open_skipped_and_corrupt_ledger(tmp_path):
    manifest = load_candidate_manifest()
    ledger = tmp_path / "cockpit.jsonl"
    snapshot_path = tmp_path / "candidate_snapshot.json"
    universe = load_forward_universe()
    snapshot = {
        "status": "complete",
        "updated_at": "2026-07-22T22:31:00+02:00",
        "records": [{"Symbol": "SPY"}, {"Symbol": "AAPL"}],
        "errors": {},
        "pre_scan_refresh_errors": {},
        "forward_universe": {
            "universe_id": universe["universe_id"],
            "universe_hash": universe["universe_hash"],
            "requested_symbols": universe["symbols"],
            "completed_symbols": universe["symbols"],
            "failed_symbols": [],
            "full_coverage": True,
        },
    }
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    append_forward_event({
        "event_type": "SNAPSHOT_AUDIT",
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_hash": manifest["manifest_hash"],
        "snapshot_hash": snapshot_hash(snapshot),
        "status": "AUDITED",
        "snapshot_updated_at": snapshot["updated_at"],
    }, ledger)
    append_forward_event({
        "event_type": "SIGNAL_OBSERVED",
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_hash": manifest["manifest_hash"],
        "signal_id": "spy-2026-07-20",
        "status": "OBSERVED",
        "symbol": "SPY",
        "asset_class": "USA / ETF",
        "horizon": 20,
        "direction": "LONG",
        "signal_date": "2026-07-20",
        "decision_reason": "LONG_CONFIRMED",
        "probability_up": 0.57,
        "expected_return": 0.01,
        "quality": "WYSOKA",
    }, ledger)
    append_forward_event({
        "event_type": "POSITION_ACCEPTED",
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_hash": manifest["manifest_hash"],
        "signal_id": "spy-2026-07-20",
        "status": "ACCEPTED",
        "symbol": "SPY",
        "direction": "LONG",
        "signal_date": "2026-07-20",
        "slot": 1,
    }, ledger)
    append_forward_event({
        "event_type": "ENTRY_FILLED",
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_hash": manifest["manifest_hash"],
        "signal_id": "spy-2026-07-20",
        "status": "OPEN",
        "symbol": "SPY",
        "direction": "LONG",
        "signal_date": "2026-07-20",
        "slot": 1,
        "entry_date": "2026-07-21",
        "entry_price": 746.29,
    }, ledger)
    append_forward_event({
        "event_type": "SIGNAL_OBSERVED",
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_hash": manifest["manifest_hash"],
        "signal_id": "spy-2026-07-22",
        "status": "OBSERVED",
        "symbol": "SPY",
        "asset_class": "USA / ETF",
        "horizon": 20,
        "direction": "LONG",
        "signal_date": "2026-07-22",
        "decision_reason": "LONG_CONFIRMED",
        "probability_up": 0.576,
        "expected_return": 0.008,
        "quality": "WYSOKA",
    }, ledger)
    append_forward_event({
        "event_type": "POSITION_SKIPPED",
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_hash": manifest["manifest_hash"],
        "signal_id": "spy-2026-07-22",
        "status": "SKIPPED",
        "symbol": "SPY",
        "direction": "LONG",
        "signal_date": "2026-07-22",
        "skip_reason": "POSITION_SKIPPED_SYMBOL_OPEN",
    }, ledger)

    cockpit = load_forward_cockpit(path=ledger, snapshot_path=snapshot_path, now="2026-07-22")
    assert cockpit["healthy"] is True
    assert cockpit["coverage"]["completed"] == 5
    assert cockpit["audit_days"] == 1
    assert cockpit["signal_days"] == 2
    assert cockpit["forward_days"] == 1
    assert cockpit["portfolio"]["open"] == 1
    assert cockpit["portfolio"]["free_slots"] == 4
    assert cockpit["latest_signal_date"] == "2026-07-22"
    assert cockpit["open_positions"][0]["Symbol"] == "SPY"
    assert cockpit["open_positions"][0]["Sesje do wyjścia"] == 19
    assert cockpit["latest_observations"][0]["Status"] == "SKIPPED"
    assert cockpit["latest_observations"][0]["Powód pominięcia"] == "POSITION_SKIPPED_SYMBOL_OPEN"

    summary_text = format_forward_cli_summary(
        snapshot,
        {
            "added_signals": 1,
            "refresh_errors": {},
            "snapshot_path": str(snapshot_path),
            "run_event_counts": {"SIGNAL_OBSERVED": 1, "POSITION_SKIPPED": 1},
            "run_skipped": [{"symbol": "SPY", "skip_reason": "POSITION_SKIPPED_SYMBOL_OPEN"}],
            "run_entries": [],
            "run_closed": [],
            "run_accepted": [],
        },
        path=ledger,
    )
    assert "Candidate v1 forward run: OK" in summary_text
    assert "Universe coverage: 5/5" in summary_text
    assert "Skipped: SPY — POSITION_SKIPPED_SYMBOL_OPEN" in summary_text

    stale_summary_text = format_forward_cli_summary(
        snapshot,
        {
            "added_signals": 0,
            "refresh_errors": {},
            "snapshot_path": str(snapshot_path),
            "run_event_counts": {},
            "run_skipped": [],
            "run_entries": [],
            "run_closed": [],
            "run_accepted": [],
        },
        path=ledger,
    )
    assert "New positions: 0" in stale_summary_text
    assert "Skipped: none" in stale_summary_text

    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not-json}\n", encoding="utf-8")
    broken_view = load_forward_cockpit(path=broken, snapshot_path=snapshot_path)
    assert broken_view["healthy"] is False
    assert any("Ledger error" in problem for problem in broken_view["problems"])


def test_forward_cockpit_counts_audit_days_without_signals_and_same_day_reruns(tmp_path):
    path = tmp_path / "audit_days.jsonl"
    first_snapshot = {
        "status": "complete",
        "schema_version": 6,
        "updated_at": "2026-07-20T22:30:00+02:00",
        "records": [{
            "Symbol": "GOOD", "Klasa": "USA / ETF", "Horyzont": 20, "Data": "2026-07-20",
            "Cena": 100.0, "Ocena": "KANDYDAT WZROSTOWY", "P(wzrost)": 0.61,
            "Oczekiwany ruch": 0.03, "AUC walidacji": 0.62, "Brier": 0.22,
            "Jakość modelu": "WYSOKA", "Tryb analizy": "ML", "DecisionReason": "LONG_CONFIRMED",
        }],
    }
    second_same_day = {
        **first_snapshot,
        "updated_at": "2026-07-20T22:35:00+02:00",
    }
    next_day_no_signal = {
        "status": "complete",
        "schema_version": 6,
        "updated_at": "2026-07-21T22:30:00+02:00",
        "records": [{
            "Symbol": "GOOD", "Klasa": "USA / ETF", "Horyzont": 20, "Data": "2026-07-21",
            "Cena": 101.0, "Ocena": "BRAK SYGNAŁU", "P(wzrost)": 0.50,
            "Oczekiwany ruch": 0.00, "AUC walidacji": 0.49, "Brier": 0.25,
            "Jakość modelu": "NISKA — BRAK PRZEWAGI", "Tryb analizy": "ML", "DecisionReason": "LOW_QUALITY",
        }],
    }
    assert record_snapshot_forward_signals(
        first_snapshot,
        path=path,
        allow_historical=True,
        enforce_pipeline=False,
        require_clean_tree=False,
        require_closed_bar=False,
        require_full_universe=False,
    ) == 1
    assert record_snapshot_forward_signals(
        second_same_day,
        path=path,
        allow_historical=True,
        enforce_pipeline=False,
        require_clean_tree=False,
        require_closed_bar=False,
        require_full_universe=False,
    ) == 0
    assert record_snapshot_forward_signals(
        next_day_no_signal,
        path=path,
        allow_historical=True,
        enforce_pipeline=False,
        require_clean_tree=False,
        require_closed_bar=False,
        require_full_universe=False,
    ) == 0
    events = load_forward_events(path)
    event_types = [event["event_type"] for event in events]
    assert event_types.count("SNAPSHOT_AUDIT") == 3
    assert event_types.count("SIGNAL_OBSERVED") == 1
    cockpit = build_forward_cockpit(events, manifest=load_candidate_manifest(), now="2026-07-21")
    assert cockpit["forward_days"] == 2
    assert cockpit["audit_days"] == 2
    assert cockpit["signal_days"] == 1


def test_candidate_snapshot_scans_full_frozen_universe_without_fast_shortlist():
    universe = load_forward_universe()

    def fake_scan(symbols, horizon, years):
        symbol = symbols[0]
        frame = pd.DataFrame([{
            "Symbol": symbol, "Klasa": "USA / ETF", "Horyzont": horizon, "Data": "2026-07-20",
            "Cena": 100.0, "Ocena": "KANDYDAT WZROSTOWY", "P(wzrost)": 0.61,
            "Oczekiwany ruch": 0.03, "AUC walidacji": 0.62, "Brier": 0.22,
            "Jakość modelu": "WYSOKA", "Tryb analizy": "ML",
        }])
        return frame, {}

    snapshot = build_candidate_snapshot(scan_fn=fake_scan, updated_at="2026-07-20T22:30:00+02:00")
    assert snapshot["status"] == "complete"
    assert snapshot["scan_mode"] == "candidate_v1_full_ml"
    assert snapshot["forward_universe"]["universe_hash"] == universe["universe_hash"]
    assert snapshot["forward_universe"]["requested_symbols"] == universe["symbols"]
    assert snapshot["forward_universe"]["completed_symbols"] == universe["symbols"]
    assert snapshot["forward_universe"]["failed_symbols"] == []
    assert len(snapshot["records"]) == len(universe["symbols"])
    assert {row["DecisionReason"] for row in snapshot["records"]} == {"LONG_CONFIRMED"}
    assert {row["Tryb analizy"] for row in snapshot["records"]} == {"ML"}


def test_candidate_cycle_refreshes_before_record_and_blocks_same_day_reentry(tmp_path):
    manifest = load_candidate_manifest()
    ledger = tmp_path / "cycle.jsonl"

    def seed_open(symbol: str, slot: int) -> None:
        signal_id = f"{symbol.lower()}_seed"
        append_forward_event({
            "event_type": "SIGNAL_OBSERVED",
            "candidate_id": manifest["candidate_id"],
            "candidate_manifest_hash": manifest["manifest_hash"],
            "signal_id": signal_id,
            "status": "OBSERVED",
            "symbol": symbol,
            "asset_class": "USA / ETF",
            "horizon": 20,
            "direction": "LONG",
            "execution": "NEXT_OPEN",
            "signal_date": "2026-06-23",
            "decision_label": "KANDYDAT WZROSTOWY",
            "decision_reason": "LONG_CONFIRMED",
            "signal_price": 100.0,
            "probability_up": 0.61,
            "expected_return": 0.03,
            "quality": "WYSOKA",
        }, ledger)
        append_forward_event({
            "event_type": "POSITION_ACCEPTED",
            "candidate_id": manifest["candidate_id"],
            "candidate_manifest_hash": manifest["manifest_hash"],
            "signal_id": signal_id,
            "status": "ACCEPTED",
            "symbol": symbol,
            "direction": "LONG",
            "signal_date": "2026-06-23",
            "slot": slot,
        }, ledger)
        append_forward_event({
            "event_type": "ENTRY_FILLED",
            "candidate_id": manifest["candidate_id"],
            "candidate_manifest_hash": manifest["manifest_hash"],
            "signal_id": signal_id,
            "status": "OPEN",
            "symbol": symbol,
            "direction": "LONG",
            "signal_date": "2026-06-23",
            "slot": slot,
            "entry_date": "2026-06-24",
            "entry_price": 100.0,
            "price_source": "Open",
        }, ledger)

    for slot, symbol in enumerate(["AAPL", "OLD1", "OLD2", "OLD3", "OLD4"], start=1):
        seed_open(symbol, slot)

    close_dates = pd.bdate_range("2026-06-23", periods=22)
    close_prices = np.linspace(99.0, 121.0, len(close_dates))
    close_prices[1] = 100.0
    close_prices[21] = 110.0
    close_history = pd.DataFrame({"Open": close_prices}, index=close_dates)
    short_dates = pd.bdate_range("2026-06-23", periods=10)
    short_history = pd.DataFrame({"Open": np.linspace(100.0, 105.0, len(short_dates))}, index=short_dates)
    histories = {"AAPL": close_history, "OLD1": short_history, "OLD2": short_history, "OLD3": short_history, "OLD4": short_history}

    def fake_scan(symbols, horizon, years):
        symbol = symbols[0]
        is_signal = symbol in {"AAPL", "MSFT"}
        frame = pd.DataFrame([{
            "Symbol": symbol, "Klasa": "USA / ETF", "Horyzont": horizon, "Data": "2026-07-22",
            "Cena": 100.0, "Ocena": "KANDYDAT WZROSTOWY" if is_signal else "OBSERWUJ",
            "P(wzrost)": 0.61 if is_signal else 0.50,
            "Oczekiwany ruch": 0.03 if is_signal else 0.0,
            "AUC walidacji": 0.62, "Brier": 0.22,
            "Jakość modelu": "WYSOKA", "Tryb analizy": "ML",
        }])
        return frame, {}

    snapshot, result = run_candidate_forward_cycle(
        ledger_path=ledger,
        snapshot_path=tmp_path / "candidate_snapshot.json",
        scan_fn=fake_scan,
        refresh_histories=histories,
        enforce_pipeline=False,
        require_clean_tree=False,
        require_closed_bar=False,
        updated_at="2026-07-22T22:30:00+02:00",
    )
    assert snapshot["status"] == "complete"
    assert result["added_signals"] == 2
    events = load_forward_events(ledger)
    state = reconstruct_forward_state(events)
    aapl_new = [
        item for item in state.values()
        if item.get("symbol") == "AAPL" and item.get("signal_date") == "2026-07-22"
    ][0]
    msft_new = [
        item for item in state.values()
        if item.get("symbol") == "MSFT" and item.get("signal_date") == "2026-07-22"
    ][0]
    assert aapl_new["status"] == "SKIPPED"
    assert aapl_new["skip_reason"] == "POSITION_SKIPPED_SAME_DAY_REENTRY"
    assert msft_new["status"] == "ACCEPTED"
    assert msft_new["slot"] == 1


def test_forward_automation_plan_dedupes_and_catches_up():
    config = AutomationConfig()
    cockpit = {"latest_audit_date": "2026-07-23"}

    waiting = build_automation_plan(cockpit=cockpit, now="2026-07-24T21:00:00+02:00", config=config)
    assert waiting["should_run"] is False
    assert waiting["reason"] == "ALREADY_AUDITED"
    assert waiting["target_session_date"] == "2026-07-23"

    ready = build_automation_plan(cockpit=cockpit, now="2026-07-24T22:40:00+02:00", config=config)
    assert ready["should_run"] is True
    assert ready["reason"] == "SCHEDULED_SESSION_READY"
    assert ready["target_session_date"] == "2026-07-24"

    catch_up = build_automation_plan(cockpit=cockpit, now="2026-07-25T10:00:00+02:00", config=config)
    assert catch_up["should_run"] is True
    assert catch_up["reason"] == "CATCH_UP_MISSED_SESSION"
    assert catch_up["target_session_date"] == "2026-07-24"
    assert "2026-07-24" in catch_up["missed_session_warning"]


def test_forward_automation_respects_no_session_day():
    config = AutomationConfig(closed_dates=frozenset({"2026-07-24"}))
    cockpit = {"latest_audit_date": "2026-07-23"}
    plan = build_automation_plan(cockpit=cockpit, now="2026-07-24T22:50:00+02:00", config=config)
    assert plan["should_run"] is False
    assert plan["target_session_date"] == "2026-07-23"
    assert plan["reason"] == "ALREADY_AUDITED"


def test_forward_automation_uses_nyse_holiday_calendar():
    assert "2026-07-03" in nyse_full_holidays(2026)
    config = AutomationConfig()
    cockpit = {"latest_audit_date": "2026-07-02"}
    plan = build_automation_plan(cockpit=cockpit, now="2026-07-03T22:50:00+02:00", config=config)
    assert plan["should_run"] is False
    assert plan["target_session_date"] == "2026-07-02"
    assert plan["target_session_is_nyse_session"] is True
    assert plan["reason"] == "ALREADY_AUDITED"


def test_forward_automation_three_day_gap_is_reported_without_backfill():
    config = AutomationConfig()
    cockpit = {"latest_audit_date": "2026-07-20"}
    plan = build_automation_plan(cockpit=cockpit, now="2026-07-24T22:50:00+02:00", config=config)
    assert plan["should_run"] is True
    assert plan["reason"] == "SCHEDULED_SESSION_READY_WITH_GAP"
    assert plan["target_session_date"] == "2026-07-24"
    assert plan["missing_sessions"] == ["2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24"]
    assert plan["missed_sessions"] == ["2026-07-21", "2026-07-22", "2026-07-23"]
    assert plan["missed_sessions_count"] == 3
    assert "Wrapper wykona tylko najnowszą sesję 2026-07-24" in plan["missed_session_warning"]
    assert eligible_session_dates(latest_audit_date="2026-07-20", now="2026-07-24T22:50:00+02:00", config=config)[-1] == "2026-07-24"


def test_forward_automation_skips_when_session_already_audited(tmp_path, monkeypatch):
    config = AutomationConfig(
        status_path=tmp_path / "status.json",
        lock_path=tmp_path / "run.lock",
        log_dir=tmp_path / "logs",
        candidate_command=("python", "run_candidate_forward.py"),
    )
    monkeypatch.setattr(auto_forward_module, "load_forward_cockpit", lambda: {"latest_audit_date": "2026-07-24"})

    def should_not_run(command):
        raise AssertionError("dedupe should prevent the runner")

    payload = execute_automation(
        config=config,
        now="2026-07-24T22:50:00+02:00",
        runner=should_not_run,
    )
    assert payload["automation_status"] == "SKIPPED"
    assert payload["exit_code"] == 0
    assert json.loads(config.status_path.read_text())["automation_status"] == "SKIPPED"


def test_forward_automation_rechecks_plan_after_lock(tmp_path, monkeypatch):
    config = AutomationConfig(
        status_path=tmp_path / "status.json",
        lock_path=tmp_path / "run.lock",
        log_dir=tmp_path / "logs",
        candidate_command=("python", "run_candidate_forward.py"),
    )
    cockpits = [
        {"latest_audit_date": "2026-07-23"},
        {"latest_audit_date": "2026-07-24"},
    ]

    def next_cockpit():
        return cockpits.pop(0)

    monkeypatch.setattr(auto_forward_module, "load_forward_cockpit", next_cockpit)
    payload = execute_automation(
        config=config,
        now="2026-07-24T22:50:00+02:00",
        runner=lambda command: pytest.fail("race re-check should prevent duplicate run"),
    )
    stored = json.loads(config.status_path.read_text())
    assert payload["automation_status"] == "SKIPPED"
    assert payload["race_recheck"] is True
    assert payload["plan"]["reason"] == "SCHEDULED_SESSION_READY"
    assert payload["plan_after_lock"]["reason"] == "ALREADY_AUDITED"
    assert stored["race_recheck"] is True


def test_forward_automation_lock_blocks_parallel_runs(tmp_path, monkeypatch):
    config = AutomationConfig(
        status_path=tmp_path / "status.json",
        lock_path=tmp_path / "run.lock",
        log_dir=tmp_path / "logs",
        candidate_command=("python", "run_candidate_forward.py"),
    )
    monkeypatch.setattr(auto_forward_module, "load_forward_cockpit", lambda: {"latest_audit_date": "2026-07-23"})

    with automation_lock(config.lock_path):
        payload = execute_automation(
            config=config,
            now="2026-07-24T22:50:00+02:00",
            runner=lambda command: pytest.fail("locked runner must not execute"),
        )
    assert payload["automation_status"] == "LOCKED"
    assert payload["exit_code"] == 75
    assert json.loads(config.status_path.read_text())["automation_status"] == "LOCKED"


def test_forward_automation_failed_run_writes_logs_and_status(tmp_path, monkeypatch):
    config = AutomationConfig(
        status_path=tmp_path / "status.json",
        lock_path=tmp_path / "run.lock",
        log_dir=tmp_path / "logs",
        candidate_command=("python", "run_candidate_forward.py"),
    )
    config.status_path.parent.mkdir(parents=True, exist_ok=True)
    config.status_path.write_text(json.dumps({
        "automation_status": "OK",
        "target_session_date": "2026-07-23",
        "ended_at": "2026-07-23T21:00:00+00:00",
        "exit_code": 0,
        "last_successful_run": {
            "target_session_date": "2026-07-23",
            "ended_at": "2026-07-23T21:00:00+00:00",
            "exit_code": 0,
        },
    }))
    monkeypatch.setattr(auto_forward_module, "load_forward_cockpit", lambda: {"latest_audit_date": "2026-07-23"})

    def failing_runner(command):
        return auto_forward_module.subprocess.CompletedProcess(
            command,
            2,
            '{"status":"error","run_event_counts":{"SNAPSHOT_AUDIT":1}}\nHuman summary',
            "boom",
        )

    payload = execute_automation(
        config=config,
        now="2026-07-24T22:50:00+02:00",
        runner=failing_runner,
    )
    stored = json.loads(config.status_path.read_text())
    assert payload["automation_status"] == "FAILED"
    assert stored["exit_code"] == 2
    assert Path(stored["stdout_log"]).read_text().startswith('{"status":"error"')
    assert Path(stored["stderr_log"]).read_text() == "boom"
    assert stored["runner_payload"]["status"] == "error"
    assert stored["runner_summary_text"] == "Human summary"
    assert stored["last_successful_run"]["target_session_date"] == "2026-07-23"


def test_forward_automation_launchd_plist_has_weekday_schedule(tmp_path):
    config = AutomationConfig(status_path=tmp_path / "status.json", lock_path=tmp_path / "lock", log_dir=tmp_path / "logs")
    payload = launchd_plist_payload(config=config, python_path=tmp_path / "python")
    assert payload["RunAtLoad"] is True
    assert payload["ProgramArguments"][-1] == "run-now"
    assert [item["Weekday"] for item in payload["StartCalendarInterval"]] == [1, 2, 3, 4, 5]
    assert {item["Hour"] for item in payload["StartCalendarInterval"]} == {22}
    assert {item["Minute"] for item in payload["StartCalendarInterval"]} == {35}


def test_forward_automation_launchd_status_uses_gui_domain_and_detects_privacy_block(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "launchd.stderr.log").write_text(
        "PermissionError: [Errno 1] Operation not permitted: "
        "'/Users/jakubjaworski/Documents/Codex/2026-07-03/jo/.venv/pyvenv.cfg'",
        encoding="utf-8",
    )
    monkeypatch.setattr(auto_forward_module, "AUTOMATION_LOG_DIR", log_dir)
    monkeypatch.setattr(auto_forward_module.os, "getuid", lambda: 501)

    def fake_run(command, capture_output, text, check):
        assert command == ["launchctl", "print", "gui/501/com.jamejj.marketscope.candidate-forward"]
        return auto_forward_module.subprocess.CompletedProcess(
            command,
            0,
            "gui/501/com.jamejj.marketscope.candidate-forward = {\n"
            "\tstate = not running\n"
            "\truns = 1\n"
            "\tlast exit code = 1\n"
            "}\n",
            "",
        )

    monkeypatch.setattr(auto_forward_module.subprocess, "run", fake_run)
    status = launchd_status(tmp_path / "agent.plist")
    assert status["loaded"] is True
    assert status["domain"] == "gui/501"
    assert status["state"] == "not running"
    assert status["runs"] == "1"
    assert status["last_exit_code"] == "1"
    assert status["privacy_block_detected"] is True
    assert "Full Disk Access" in status["privacy_hint"]


def verdict_integrity_result(probability: float, expected_return: float) -> dict:
    return {
        "symbol": "TEST",
        "last_date": pd.Timestamp("2026-08-07"),
        "benchmark": "^GSPC",
        "technical": {
            "return_1d": 0.01,
            "return_5d": 0.03,
            "return_20d": 0.10,
            "rsi_14": 62.0,
            "above_sma_50": True,
            "above_sma_200": True,
        },
        "risk": {"max_drawdown": -0.18},
        "forecasts": {
            20: {
                "probability_up": probability,
                "expected_return": expected_return,
                "lower_return": -0.06,
                "upper_return": 0.08,
                "auc": 0.66,
                "brier": 0.21,
                "quality": "WYSOKA",
            }
        },
    }


def multi_horizon_analysis_result() -> dict:
    result = verdict_integrity_result(0.63, 0.03)
    result["forecasts"] = {
        5: {
            "probability_up": 0.60,
            "expected_return": -0.01,
            "lower_return": -0.05,
            "upper_return": 0.06,
            "auc": 0.64,
            "brier": 0.22,
            "quality": "WYSOKA",
            "importance": {"ret_5": 0.6},
        },
        20: {
            "probability_up": 0.63,
            "expected_return": 0.03,
            "lower_return": -0.06,
            "upper_return": 0.12,
            "auc": 0.66,
            "brier": 0.21,
            "quality": "WYSOKA",
            "importance": {"ret_20": 0.7},
        },
        60: {
            "probability_up": 0.54,
            "expected_return": 0.02,
            "lower_return": -0.10,
            "upper_return": 0.18,
            "auc": 0.59,
            "brier": 0.24,
            "quality": "UMIARKOWANA",
            "importance": {"ret_60": 0.5},
        },
    }
    return result


def test_analysis_report_manual_horizon_uses_same_shared_verdict_and_auto_is_unchanged():
    result = multi_horizon_analysis_result()

    automatic = build_analysis_report(result)
    manual = build_analysis_report(result, selected_horizon=5)

    assert automatic["selection_mode"] == "AUTO"
    assert automatic["primary_horizon"] == 20
    assert automatic["verdict"]["label"] == "LONG"
    assert manual["selection_mode"] == "MANUAL"
    assert manual["requested_horizon"] == 5
    assert manual["primary_horizon"] == 5
    assert manual["verdict"]["label"] == "OBSERWUJ"
    assert manual["verdict"]["reason"] == "EXPECTED_RETURN_CONFLICT"
    assert "wybrany ręcznie" in manual["cards"][2][2].lower()


def test_analysis_report_manual_horizon_has_no_silent_fallback():
    with pytest.raises(ValueError, match="Horyzont 1 nie jest dostępny"):
        build_analysis_report(multi_horizon_analysis_result(), selected_horizon=1)


def test_aggregate_model_view_uses_manually_selected_report_horizon():
    result = multi_horizon_analysis_result()
    report = build_analysis_report(result, selected_horizon=60)
    view = load_aggregate_model_view()(result, report)

    assert report["primary_horizon"] == 60
    assert view["verdict"] == report["headline"].rstrip(".")
    assert view["best_label"].startswith("60 sesji")
    assert view["tone"] == "info"


def test_watchlist_saves_exact_manual_horizon_forecast_and_verdict():
    result = multi_horizon_analysis_result()
    report = build_analysis_report(result, selected_horizon=5)

    item = watch_item_from_analysis(result, report, {"radar_updated_at": "2026-08-07T22:00:00+02:00"})

    assert item["horizon"] == 5
    assert item["probability_up"] == 0.60
    assert item["expected_return"] == -0.01
    assert item["verdict_label"] == "OBSERWUJ"
    assert item["verdict"] == "EXPECTED_RETURN_CONFLICT"


def test_evidence_registry_maps_only_exact_protocol_symbols():
    registry = load_evidence_registry()
    candidate_claim = next(
        claim for claim in registry["historical_claims"]
        if claim["claim_id"] == "candidate_v1_h20_research_candidate"
    )
    forward_claim = registry["forward_claims"][0]
    frozen_universe = load_forward_universe()
    unseen_universe = load_unseen_universe()
    unseen_claim = next(
        claim for claim in registry["historical_claims"]
        if claim["claim_id"] == "unseen_usa_etf_v1_h20_not_run"
    )
    assert candidate_claim["exact_symbols"] == frozen_universe["symbols"]
    assert candidate_claim["candidate_manifest_hash"] == load_candidate_manifest()["manifest_hash"]
    assert forward_claim["exact_symbols"] == frozen_universe["symbols"]
    assert forward_claim["forward_universe_hash"] == frozen_universe["universe_hash"]
    assert unseen_claim["exact_symbols"] == unseen_universe["symbols"]
    assert unseen_claim["universe_hash"] == unseen_universe["universe_hash"]

    spy = resolve_evidence("spy", 20, [1, 5, 20, 60], registry)
    assert spy.forecast_availability is ForecastAvailability.AVAILABLE
    assert spy.historical_evidence is HistoricalEvidence.RESEARCH_CANDIDATE
    assert spy.forward_evidence is ForwardEvidence.IN_PROGRESS
    assert spy.evidence_scope is EvidenceScope.AGGREGATE_UNIVERSE

    aapl_5d = resolve_evidence("AAPL", 5, [1, 5, 20, 60], registry)
    assert aapl_5d.historical_evidence is HistoricalEvidence.NO_EDGE
    assert aapl_5d.forward_evidence is ForwardEvidence.NOT_STARTED

    btc_20d = resolve_evidence("BTC-USD", 20, [1, 5, 20, 60], registry)
    assert btc_20d.historical_evidence is HistoricalEvidence.INSUFFICIENT_EVIDENCE
    assert btc_20d.forward_evidence is ForwardEvidence.NOT_STARTED

    btc_60d = resolve_evidence("BTC-USD", 60, [1, 5, 20, 60], registry)
    assert btc_60d.forecast_availability is ForecastAvailability.AVAILABLE
    assert btc_60d.historical_evidence is HistoricalEvidence.UNTESTED

    amzn = resolve_evidence("AMZN", 20, [20], registry)
    assert amzn.historical_evidence is HistoricalEvidence.UNTESTED
    assert amzn.historical_protocol_id == "unseen_usa_etf_v1"

    for symbol in ("XTB.WA", "AAPL.US"):
        unknown = resolve_evidence(symbol, 20, [20], registry)
        assert unknown.historical_evidence is HistoricalEvidence.UNTESTED
        assert unknown.historical_protocol_id is None
        assert unknown.forward_evidence is ForwardEvidence.NOT_STARTED


def test_evidence_copy_preserves_aggregate_scope_without_symbol_overclaim():
    registry = load_evidence_registry()

    aapl_copy = evidence_copy(resolve_evidence("AAPL", 5, [5], registry))
    assert aapl_copy.title == "Brak wykazanej przewagi w protokole zbiorczym"
    assert aapl_copy.summary == "AAPL należało do badanego koszyka; status nie jest indywidualną oceną instrumentu."
    assert "AAPL było częścią badanego koszyka" in aapl_copy.detail
    assert "nie jest to osobna ocena skuteczności AAPL" in aapl_copy.detail

    spy_copy = evidence_copy(resolve_evidence("SPY", 20, [20], registry))
    assert spy_copy.title == "Kandydat badawczy · Forward trwa"
    assert spy_copy.summary == "Wynik pochodzi z protokołu zbiorczego, nie z indywidualnej walidacji SPY."
    assert "wynik protokołu zbiorczego" in spy_copy.detail
    assert "nie jest to indywidualnie potwierdzony edge" in spy_copy.detail
    assert "Forward tego koszyka trwa" in spy_copy.detail
    assert "potwierdzonej przewagi" in spy_copy.detail
    assert len(spy_copy.summary) < len(spy_copy.detail)


def test_evidence_copy_distinguishes_insufficient_unrun_and_unregistered():
    registry = load_evidence_registry()

    btc_20d = evidence_copy(resolve_evidence("BTC-USD", 20, [20], registry))
    assert btc_20d.title == "Za mało dowodów w badanym zakresie"
    assert "zbyt mała do wiarygodnego wniosku" in btc_20d.detail

    btc_60d = evidence_copy(resolve_evidence("BTC-USD", 60, [60], registry))
    assert btc_60d.title == "Eksperymentalny forecast"
    assert "Brak zarejestrowanej walidacji" in btc_60d.detail

    amzn = evidence_copy(resolve_evidence("AMZN", 20, [20], registry))
    assert amzn.title == "Eksperymentalny forecast · protokół jeszcze nieuruchomiony"
    assert "prerejestrowanego koszyka" in amzn.detail

    xtb = evidence_copy(resolve_evidence("XTB.WA", 20, [20], registry))
    assert xtb.title == "Eksperymentalny forecast"
    assert "Brak zarejestrowanej walidacji" in xtb.detail


def test_evidence_registry_hash_is_canonical_and_detects_tampering():
    registry = load_evidence_registry()
    reordered = dict(reversed(list(registry.items())))
    assert registry_hash(reordered) == registry["registry_hash"]

    tampered = json.loads(json.dumps(registry))
    tampered["historical_claims"][0]["status"] = "RESEARCH_CANDIDATE"
    with pytest.raises(EvidenceRegistryError, match="hash mismatch"):
        validate_evidence_registry(tampered)

    changed_hash = json.loads(json.dumps(registry))
    changed_hash["registry_hash"] = "0" * 64
    with pytest.raises(EvidenceRegistryError, match="hash mismatch"):
        validate_evidence_registry(changed_hash)


def test_evidence_registry_rejects_overlapping_exact_symbol_claims():
    registry = load_evidence_registry()
    overlapping = json.loads(json.dumps(registry))
    duplicate = json.loads(json.dumps(overlapping["historical_claims"][0]))
    duplicate["claim_id"] = "overlapping_aapl_h1"
    duplicate["exact_symbols"] = ["AAPL"]
    overlapping["historical_claims"].append(duplicate)
    overlapping["registry_hash"] = registry_hash(overlapping)

    with pytest.raises(EvidenceRegistryError, match=r"Overlapping historical claims.*AAPL 1"):
        validate_evidence_registry(overlapping)


def test_evidence_registry_provenance_is_structurally_complete():
    registry = load_evidence_registry()

    for claim in registry["historical_claims"]:
        assert claim["artifact_refs"]
        for artifact in claim["artifact_refs"]:
            assert artifact["path"]
            assert len(artifact["sha256"]) == 64

    forward_claim = registry["forward_claims"][0]
    assert forward_claim["ledger_id"] == "forward_ledger_candidate_v1"
    assert forward_claim["ledger_path"] == "data/forward_ledger_candidate_v1.jsonl"
    assert len(forward_claim["first_event_hash"]) == 64
    assert forward_claim["started_at"]

    incomplete = json.loads(json.dumps(registry))
    incomplete["historical_claims"][0]["artifact_refs"][0].pop("sha256")
    incomplete["registry_hash"] = registry_hash(incomplete)
    with pytest.raises(EvidenceRegistryError, match="artifact requires a canonical sha256"):
        validate_evidence_registry(incomplete)


def test_explicit_evidence_source_audit_uses_portable_fixtures(tmp_path):
    artifact_path = tmp_path / "research" / "report.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text('{"result":"fixture"}', encoding="utf-8")
    artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    first_event = {
        "event_hash": "a" * 64,
        "event_time_utc": "2026-01-02T20:00:00+00:00",
        "candidate_manifest_hash": "b" * 64,
    }
    ledger_path = tmp_path / "proof" / "ledger.jsonl"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps(first_event) + "\n", encoding="utf-8")

    registry = {
        "schema_version": 1,
        "registry_id": "fixture_registry",
        "evidence_updated_at": "2026-01-02T20:00:00+00:00",
        "hash_method": "sha256(canonical_json(registry_without_top_level_registry_hash))",
        "historical_claims": [{
            "claim_id": "fixture_historical",
            "protocol_id": "fixture_protocol",
            "horizon": 5,
            "exact_symbols": ["TEST"],
            "market_scope": ["TEST"],
            "evidence_scope": "SYMBOL_SPECIFIC",
            "status": "NO_EDGE",
            "artifact_refs": [{"path": "research/report.json", "sha256": artifact_sha}],
            "evidence_updated_at": "2026-01-02T20:00:00+00:00",
        }],
        "forward_claims": [{
            "claim_id": "fixture_forward",
            "protocol_id": "fixture_forward_protocol",
            "horizon": 20,
            "exact_symbols": ["TEST"],
            "market_scope": ["TEST"],
            "evidence_scope": "SYMBOL_SPECIFIC",
            "status": "IN_PROGRESS",
            "candidate_manifest_hash": "b" * 64,
            "forward_universe_hash": "c" * 64,
            "ledger_id": "fixture_ledger",
            "ledger_path": "proof/ledger.jsonl",
            "first_event_hash": "a" * 64,
            "started_at": "2026-01-02T20:00:00+00:00",
            "evidence_updated_at": "2026-01-02T20:00:00+00:00",
        }],
    }
    registry["registry_hash"] = registry_hash(registry)

    assert verify_evidence_sources(registry, root=tmp_path) == {
        "artifacts": 1,
        "forward_checkpoints": 1,
    }
    artifact_path.write_text('{"result":"tampered"}', encoding="utf-8")
    with pytest.raises(EvidenceRegistryError, match="artifact hash mismatch"):
        verify_evidence_sources(registry, root=tmp_path)


def test_selected_report_horizon_is_the_only_input_to_evidence_resolution():
    result = multi_horizon_analysis_result()
    result["symbol"] = "SPY"
    automatic = build_analysis_report(result)
    manual = build_analysis_report(result, selected_horizon=60)

    auto_evidence = resolve_evidence("SPY", automatic["primary_horizon"], result["forecasts"].keys())
    manual_evidence = resolve_evidence("SPY", manual["primary_horizon"], result["forecasts"].keys())

    assert automatic["primary_horizon"] == 20
    assert auto_evidence.historical_evidence is HistoricalEvidence.RESEARCH_CANDIDATE
    assert manual["primary_horizon"] == 60
    assert manual_evidence.historical_evidence is HistoricalEvidence.UNTESTED


@pytest.mark.parametrize("probability,expected_return", [(0.60, -0.01), (0.54, 0.02)])
def test_aggregate_model_view_does_not_create_bullish_verdict_outside_shared_gate(probability, expected_return):
    result = verdict_integrity_result(probability, expected_return)
    report = build_analysis_report(result)
    view = load_aggregate_model_view()(result, report)

    assert report["verdict"]["decision"] == 0
    assert report["verdict"]["label"] == "OBSERWUJ"
    assert view["verdict"] == report["headline"].rstrip(".")
    assert view["tone"] == "info"
    assert "model wskazuje scenariusz wzrostowy" not in f"{view['verdict']} {view['detail']}".lower()


def test_aggregate_model_view_matches_shared_confirmed_long():
    result = verdict_integrity_result(0.60, 0.02)
    report = build_analysis_report(result)
    view = load_aggregate_model_view()(result, report)

    assert report["verdict"]["decision"] == 1
    assert report["verdict"]["label"] == "LONG"
    assert view["verdict"] == report["headline"].rstrip(".")
    assert view["tone"] == "success"
    assert "scenariusz wzrostowy spełnia warunki MarketScope" in view["verdict"]


def test_analysis_report_separates_radar_and_full_analysis_dates():
    result = {
        "symbol": "LPP.WA",
        "last_date": pd.Timestamp("2026-08-07"),
        "benchmark": "ETFBW20TR.WA",
        "technical": {
            "return_1d": 0.01,
            "return_5d": 0.04,
            "return_20d": 0.08,
            "rsi_14": 61.0,
            "above_sma_50": True,
            "above_sma_200": True,
        },
        "risk": {"max_drawdown": -0.22},
        "forecasts": {
            5: {
                "probability_up": 0.51,
                "expected_return": 0.002,
                "lower_return": -0.03,
                "upper_return": 0.04,
                "auc": 0.51,
                "brier": 0.25,
                "quality": "NISKA — BRAK PRZEWAGI",
            },
            20: {
                "probability_up": 0.635,
                "expected_return": 0.029,
                "lower_return": -0.05,
                "upper_return": 0.12,
                "auc": 0.64,
                "brier": 0.22,
                "quality": "WYSOKA",
            },
        },
    }
    report = build_analysis_report(result, {}, {"radar_updated_at": "2026-08-05T07:13:00+02:00"})

    assert "20 sesji" in report["headline"]
    assert "scenariusz wzrostowy spełnia warunki MarketScope" in report["headline"]
    assert report["cards"][0][1] == "Scenariusz wzrostowy spełnia warunki MarketScope"
    assert report["cards"][1][0] == "Wsparcie w walidacji"
    assert report["freshness"]["radar"] == "2026-08-05 07:13"
    assert report["freshness"]["analysis"] == "2026-08-07"
    assert "Candidate v1" not in report["headline"]
    assert "Candidate v1" not in report["body"]
    assert any("Dolny zakres 90%" in item for item in report["counterpoints"])


def test_analysis_report_uses_shared_verdict_for_expected_return_conflict():
    result = {
        "symbol": "TEST",
        "last_date": pd.Timestamp("2026-08-07"),
        "benchmark": "^GSPC",
        "technical": {
            "return_1d": 0.01,
            "return_5d": 0.03,
            "return_20d": 0.10,
            "rsi_14": 62.0,
            "above_sma_50": True,
            "above_sma_200": True,
        },
        "risk": {"max_drawdown": -0.18},
        "forecasts": {
            20: {
                "probability_up": 0.63,
                "expected_return": -0.01,
                "lower_return": -0.06,
                "upper_return": 0.08,
                "auc": 0.66,
                "brier": 0.21,
                "quality": "WYSOKA",
            }
        },
    }
    report = build_analysis_report(result, {}, {"radar_updated_at": "2026-08-05T07:13:00+02:00"})

    assert "obserwuj" in report["headline"].lower()
    assert "reguły MarketScope nie potwierdzają kierunku" in report["headline"]
    assert "kandydat wzrostowy" not in report["headline"].lower()
    assert report["cards"][0][1] == "Obserwuj"
    assert any("konflikt" in item.lower() for item in report["evidence"] + report["counterpoints"])


def test_analysis_report_describes_running_radar_snapshot_without_dash():
    result = {
        "symbol": "TEST",
        "last_date": pd.Timestamp("2026-08-07"),
        "benchmark": "^GSPC",
        "technical": {"above_sma_50": True, "above_sma_200": True},
        "risk": {},
        "forecasts": {
            20: {
                "probability_up": 0.57,
                "expected_return": 0.02,
                "quality": "UMIARKOWANA",
            }
        },
    }
    report = build_analysis_report(
        result,
        {},
        {"radar_status": "running", "radar_started_at": "2026-08-08T20:05:12+02:00"},
    )

    assert report["freshness"]["radar"] == "skan w toku od 2026-08-08 20:05"
    assert report["freshness"]["radar"] != "—"
    assert "trakcie odświeżania" in report["freshness"]["note"]


def guidance_row(symbol="XTB.WA", *, mode="ML", action="PRIORYTET DO ANALIZY", probability=0.63,
                 expected_return=0.025, quality="WYSOKA", grade="B — watchlist",
                 thesis="silne momentum · trend 50/200 wspiera ruch"):
    return {
        "Symbol": symbol,
        "Klasa": "GPW" if symbol.endswith(".WA") else "USA / ETF",
        "Tryb analizy": mode,
        "Horyzont": 20,
        "Akcja radaru": action,
        "Setup grade": grade,
        "Teza radaru": thesis,
        "P(wzrost)": probability,
        "Oczekiwany ruch": expected_return,
        "Jakość modelu": quality,
        "AUC walidacji": 0.64,
        "Brier": 0.21,
        "Deep score": 140,
        "Setup score": 88,
        "Radar score": 12,
        "Edge score": 8,
    }


def test_start_guidance_empty_state_has_safe_cards():
    guidance = build_start_guidance(
        snapshot={},
        cockpit={},
        automation={},
        proof_state={"label": "OK", "klass": "", "detail": "healthy"},
        universe_size=163,
    )

    ids = [card["id"] for card in guidance["cards"]]
    assert ids == ["forward_empty", "radar_empty", "methodology_guardrail"]
    text = " ".join(card["title"] + " " + card["body"] for card in guidance["cards"]).lower()
    assert "kup" not in text
    assert "sprzedaj" not in text


def test_start_guidance_running_scan_warns_about_partial_data():
    guidance = build_start_guidance(
        snapshot={
            "status": "running",
            "started_at": "2026-08-08T20:05:12+02:00",
            "records": [],
            "universe_total": 163,
            "fast_completed": 163,
            "completed": 165,
            "total": 199,
        },
        cockpit={},
        automation={},
        proof_state={"label": "OK", "klass": "", "detail": "healthy"},
        universe_size=163,
    )

    assert guidance["freshness"] == "skan w toku od 2026-08-08 20:05"
    assert "częściow" in guidance["warning"]
    assert guidance["cards"][0]["id"] == "radar_running"
    assert "aktualizowany" in guidance["cards"][0]["title"]
    assert "mieli" not in guidance["cards"][0]["title"]
    assert "FAST 163/163" in guidance["cards"][0]["body"]
    assert "Deep ML" in guidance["cards"][0]["body"]
    assert "199 instrument" not in guidance["cards"][0]["body"]


def test_start_guidance_prioritizes_proof_problem():
    guidance = build_start_guidance(
        snapshot={"status": "complete", "updated_at": "2026-08-08T07:13:00+02:00", "records": [guidance_row()]},
        cockpit={},
        automation={},
        proof_state={"label": "Wymaga uwagi", "klass": "bad", "detail": "hash snapshotu nie pasuje"},
    )

    assert guidance["cards"][0]["id"] == "proof_attention"
    assert guidance["cards"][0]["action"] == "show_forward_details"
    assert "hash snapshotu" in guidance["cards"][0]["body"]


def test_start_guidance_deduplicates_forward_and_ml_symbol():
    cockpit = {
        "open_positions": [{"Symbol": "SPY", "Data wejścia": "2026-07-21", "Cena wejścia": 746.29, "Sesje do wyjścia": 10}],
        "portfolio": {"open": 1, "slots": 5},
    }
    guidance = build_start_guidance(
        snapshot={"status": "complete", "updated_at": "2026-08-08T07:13:00+02:00", "records": [guidance_row("SPY")]},
        cockpit=cockpit,
        automation={},
        proof_state={"label": "OK", "klass": "", "detail": "healthy"},
    )

    spy_cards = [card for card in guidance["cards"] if card.get("symbol") == "SPY"]
    assert len(spy_cards) == 1
    assert spy_cards[0]["id"] == "ml_candidate"
    assert spy_cards[0]["action"] == "full_analysis"


def test_start_guidance_fast_only_is_not_ml_confirmation():
    guidance = build_start_guidance(
        snapshot={
            "status": "complete",
            "updated_at": "2026-08-08T07:13:00+02:00",
            "records": [guidance_row("LPP.WA", mode="FAST", action="FAST SHORTLIST", probability=None, quality="FAST — BEZ ML")],
        },
        cockpit={},
        automation={},
        proof_state={"label": "OK", "klass": "", "detail": "healthy"},
    )

    fast = next(card for card in guidance["cards"] if card["id"] == "fast_setup")
    assert fast["source"] == "FAST Radar"
    assert fast["status"] == "bez potwierdzenia ML"
    assert "nie jest potwierdzeniem ML" in fast["body"]


def test_start_guidance_risk_alert_wins_before_ml_candidate():
    risk = guidance_row(
        "DEXE-USD",
        mode="FAST",
        action="RYZYKO / UNIKAJ",
        expected_return=-0.18,
        quality="FAST — BEZ ML",
        grade="R — ryzyko dominuje",
        thesis="ML bez przewagi · wysokie ryzyko/zmienność",
    )
    guidance = build_start_guidance(
        snapshot={"status": "complete", "updated_at": "2026-08-08T07:13:00+02:00", "records": [guidance_row("XTB.WA"), risk]},
        cockpit={},
        automation={},
        proof_state={"label": "OK", "klass": "", "detail": "healthy"},
    )

    assert guidance["cards"][0]["id"] == "risk_alert"
    assert guidance["cards"][0]["symbol"] == "DEXE-USD"
    assert guidance["cards"][0]["status"] == "Podwyższone ryzyko"


def test_start_guidance_stale_snapshot_warns():
    guidance = build_start_guidance(
        snapshot={"status": "complete", "updated_at": "2026-08-01T07:13:00+02:00", "records": [guidance_row()]},
        cockpit={},
        automation={},
        proof_state={"label": "OK", "klass": "", "detail": "healthy"},
        radar_stale=True,
    )

    assert guidance["warning"]
    assert guidance["cards"][0]["id"] == "radar_stale"
