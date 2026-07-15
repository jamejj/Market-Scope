from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .backtest import _supervised_execution_frame
from .data import download_history
from .model import fit_forecast_state
from .signals import DEFAULT_SIGNAL_THRESHOLD, signal_verdict


@dataclass(frozen=True)
class ValidationConfig:
    """Frozen experiment contract for aggregate validation."""

    horizons: tuple[int, ...] = (1, 5, 20)
    threshold: float = DEFAULT_SIGNAL_THRESHOLD
    cost_bps: float = 10.0
    slippage_bps: float = 5.0
    initial_train: int = 420
    test_size: int = 90
    max_folds: int | None = 4
    holdout_size: int = 90


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_end: int
    test_start: int
    test_end: int
    fold_type: str = "WALK_FORWARD"


def _fold_ranges(length: int, horizon: int, config: ValidationConfig) -> list[FoldSpec]:
    """Return chronological folds with a purge gap before each test block."""
    minimum_train = max(250, config.initial_train)
    test_size = max(20, config.test_size)
    candidates: list[FoldSpec] = []
    test_start = minimum_train + horizon
    fold_id = 1
    holdout_size = max(0, int(config.holdout_size))
    holdout_start: int | None = None
    regular_end = length - horizon
    if holdout_size >= 20 and length - holdout_size >= minimum_train + horizon + 20:
        holdout_start = length - holdout_size
        regular_end = max(minimum_train + horizon, holdout_start - horizon)

    while test_start < regular_end:
        train_end = test_start - horizon
        test_end = min(regular_end, test_start + test_size)
        if train_end >= 250 and test_end - test_start >= 20:
            candidates.append(FoldSpec(fold_id, train_end, test_start, test_end))
            fold_id += 1
        test_start += test_size

    if config.max_folds is not None and len(candidates) > config.max_folds:
        selected_positions = np.linspace(0, len(candidates) - 1, config.max_folds, dtype=int)
        candidates = [candidates[int(position)] for position in dict.fromkeys(selected_positions)]
        candidates = [
            FoldSpec(fold_id=index + 1, train_end=fold.train_end, test_start=fold.test_start, test_end=fold.test_end)
            for index, fold in enumerate(candidates)
        ]
    ranges: list[FoldSpec] = candidates
    fold_id = len(ranges) + 1
    if holdout_start is not None:
        train_end = holdout_start - horizon
        if train_end >= 250 and length - holdout_start >= 20:
            ranges.append(FoldSpec(fold_id, train_end, holdout_start, length, "HOLDOUT"))
    return ranges


def _signal_record(
    *,
    symbol: str,
    market: str,
    horizon: int,
    fold: FoldSpec,
    index_position: int,
    X: pd.DataFrame,
    y: pd.Series,
    model_forward: pd.Series,
    execution_forward: pd.Series,
    prices: pd.DataFrame,
    state,
    threshold: float,
    total_cost: float,
) -> dict:
    prediction = state.predict(X.iloc[[index_position]])
    inputs = prediction.signal_inputs(source="AGGREGATE_VALIDATION")
    verdict = signal_verdict(inputs, threshold)
    position = verdict.decision
    gross = position * float(execution_forward.iloc[index_position])
    net = gross - abs(position) * total_cost
    linear_probability = float(state.class_models["linear"].predict_proba(X.iloc[[index_position]])[:, 1][0])
    linear_position = 1 if linear_probability >= threshold else (-1 if linear_probability <= 1 - threshold else 0)
    momentum_position = 1 if float(X.iloc[index_position].get("ret_20", 0.0)) > 0 else -1
    always_long_return = float(execution_forward.iloc[index_position]) - total_cost
    buy_hold_return = float(execution_forward.iloc[index_position])
    momentum_return = momentum_position * float(execution_forward.iloc[index_position]) - total_cost
    linear_return = linear_position * float(execution_forward.iloc[index_position]) - abs(linear_position) * total_cost
    price_row = prices.iloc[index_position]
    return {
        "Date": X.index[index_position],
        "Symbol": symbol,
        "Market": market,
        "Horizon": int(horizon),
        "Fold": int(fold.fold_id),
        "FoldType": fold.fold_type,
        "TrainEndDate": X.index[state.history_end - 1],
        "CoreEndDate": X.index[state.model_train_end - 1],
        "CalibrationStartDate": X.index[state.calibration_start] if state.calibration_start is not None else None,
        "CalibrationEndDate": X.index[state.calibration_end - 1] if state.calibration_end is not None else None,
        "AssessmentStartDate": X.index[state.assessment_start],
        "AssessmentEndDate": X.index[state.assessment_end - 1],
        "TestStartDate": X.index[fold.test_start],
        "TestEndDate": X.index[fold.test_end - 1],
        "PurgeGap": int(horizon),
        "EntryDate": price_row["EntryDate"],
        "ExitDate": price_row["ExitDate"],
        "EntryPrice": float(price_row["EntryPrice"]),
        "ExitPrice": float(price_row["ExitPrice"]),
        "Probability": prediction.probability_up,
        "ExpectedReturn": prediction.expected_return,
        "RawProbability": prediction.raw_probability,
        "RawExpectedReturn": prediction.raw_expected_return,
        "Skill": prediction.skill,
        "Quality": state.quality,
        "ValidationAUC": state.auc,
        "ValidationBrier": state.brier,
        "Position": int(position),
        "DecisionReason": verdict.reason,
        "DecisionLabel": verdict.label,
        "ModelReturn": float(model_forward.iloc[index_position]),
        "ExecutionReturn": float(execution_forward.iloc[index_position]),
        "GrossReturn": gross,
        "Return": net,
        "RoundTripCost": total_cost,
        "AlwaysLongReturn": always_long_return,
        "BuyHoldReturn": buy_hold_return,
        "MomentumPosition": int(momentum_position),
        "MomentumReturn": momentum_return,
        "LinearProbability": linear_probability,
        "LinearPosition": int(linear_position),
        "LinearReturn": linear_return,
        "ActualUp": int(y.iloc[index_position]),
        "ExecutionUp": int(execution_forward.iloc[index_position] > 0),
        "Target": "close_to_close",
        "Execution": "next_open",
    }


def validate_history(
    symbol: str,
    data: pd.DataFrame,
    *,
    market: str = "Unknown",
    context: pd.DataFrame | None = None,
    config: ValidationConfig = ValidationConfig(),
) -> pd.DataFrame:
    """Validate one instrument across horizons and chronological folds.

    Every potential decision point is retained, including rejected signals. This
    makes the output suitable for aggregate edge analysis and rejection auditing.
    """
    records: list[dict] = []
    total_cost = (config.cost_bps + config.slippage_bps) / 10_000
    for horizon in config.horizons:
        X, y, model_forward, execution_forward, prices = _supervised_execution_frame(data, horizon, context)
        for fold in _fold_ranges(len(X), horizon, config):
            try:
                state = fit_forecast_state(
                    X.iloc[:fold.train_end],
                    y.iloc[:fold.train_end],
                    model_forward.iloc[:fold.train_end],
                    horizon,
                )
            except ValueError:
                continue
            for i in range(fold.test_start, fold.test_end):
                records.append(
                    _signal_record(
                        symbol=symbol,
                        market=market,
                        horizon=horizon,
                        fold=fold,
                        index_position=i,
                        X=X,
                        y=y,
                        model_forward=model_forward,
                        execution_forward=execution_forward,
                        prices=prices,
                        state=state,
                        threshold=config.threshold,
                        total_cost=total_cost,
                    )
                )
    return pd.DataFrame(records)


def _fingerprint_frame(digest, key: str, frame: pd.DataFrame, ranges: dict[str, dict]) -> None:
    frame = frame.sort_index()
    cols = [col for col in ("Open", "High", "Low", "Close", "Volume") if col in frame]
    start = str(frame.index.min().date()) if len(frame) else None
    end = str(frame.index.max().date()) if len(frame) else None
    ranges[key] = {"start": start, "end": end, "rows": int(len(frame))}
    digest.update(key.encode())
    digest.update(json.dumps(ranges[key], sort_keys=True).encode())
    if cols:
        digest.update(pd.util.hash_pandas_object(frame[cols], index=True).values.tobytes())


def data_fingerprint(
    histories: dict[str, pd.DataFrame], contexts: dict[str, pd.DataFrame | None] | None = None,
) -> tuple[str, dict[str, dict]]:
    """Hash the exact input data used by an experiment."""
    digest = hashlib.sha256()
    ranges: dict[str, dict] = {}
    for symbol in sorted(histories):
        _fingerprint_frame(digest, symbol, histories[symbol], ranges)
    for symbol in sorted((contexts or {}).keys()):
        context = (contexts or {}).get(symbol)
        if context is not None and not context.empty:
            _fingerprint_frame(digest, f"context:{symbol}", context, ranges)
    return digest.hexdigest()[:24], ranges


def aggregate_validate_histories(
    histories: dict[str, pd.DataFrame],
    *,
    markets: dict[str, str] | None = None,
    contexts: dict[str, pd.DataFrame | None] | None = None,
    config: ValidationConfig = ValidationConfig(),
) -> pd.DataFrame:
    frames = []
    for symbol, data in histories.items():
        frames.append(
            validate_history(
                symbol,
                data,
                market=(markets or {}).get(symbol, "Unknown"),
                context=(contexts or {}).get(symbol),
                config=config,
            )
        )
    if not frames:
        return pd.DataFrame()
    non_empty = [frame for frame in frames if not frame.empty]
    result = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    fingerprint, ranges = data_fingerprint(histories, contexts)
    result.attrs["data_fingerprint"] = fingerprint
    result.attrs["data_ranges"] = ranges
    return result


def aggregate_validate_universe(
    symbols: dict[str, str] | list[str] | tuple[str, ...],
    *,
    years: int = 8,
    config: ValidationConfig = ValidationConfig(),
    loader: Callable[[str, int], pd.DataFrame] = download_history,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Download and validate a universe. Intended for explicit, long-running jobs."""
    if isinstance(symbols, dict):
        symbol_to_market = symbols
    else:
        symbol_to_market = {symbol: "Unknown" for symbol in symbols}

    histories: dict[str, pd.DataFrame] = {}
    contexts: dict[str, pd.DataFrame | None] = {}
    errors: dict[str, str] = {}
    try:
        from .engine import _benchmark_for
    except Exception:
        _benchmark_for = lambda symbol: None  # type: ignore[assignment]

    for symbol in symbol_to_market:
        try:
            histories[symbol] = loader(symbol, years)
            benchmark = _benchmark_for(symbol)
            if benchmark:
                try:
                    contexts[symbol] = loader(benchmark, years)
                except Exception:
                    contexts[symbol] = None
            else:
                contexts[symbol] = None
        except Exception as exc:
            errors[symbol] = str(exc)
    frame = aggregate_validate_histories(histories, markets=symbol_to_market, contexts=contexts, config=config)
    return frame, errors


def _portfolio_timeline(
    frame: pd.DataFrame, return_col: str = "Return", position_col: str = "Position",
) -> pd.DataFrame:
    """Approximate daily portfolio path by keeping positions open from entry to exit."""
    if frame.empty or return_col not in frame or position_col not in frame:
        return pd.DataFrame(columns=["Return", "ActivePositions", "Entries"])
    active = frame[frame[position_col] != 0].copy()
    if active.empty:
        return pd.DataFrame(columns=["Return", "ActivePositions", "Entries"])
    rows: list[dict] = []
    entry_counts: dict[pd.Timestamp, int] = {}
    for _, trade in active.iterrows():
        entry = pd.Timestamp(trade["EntryDate"]).normalize()
        exit_date = pd.Timestamp(trade["ExitDate"]).normalize()
        if pd.isna(entry) or pd.isna(exit_date):
            continue
        if exit_date < entry:
            exit_date = entry
        holding_dates = pd.date_range(entry, exit_date, freq="D")
        if holding_dates.empty:
            holding_dates = pd.DatetimeIndex([entry])
        daily_piece = float(trade[return_col]) / len(holding_dates)
        entry_counts[entry] = entry_counts.get(entry, 0) + 1
        for date in holding_dates:
            rows.append({"Date": date, "PositionReturn": daily_piece})
    if not rows:
        return pd.DataFrame(columns=["Return", "ActivePositions", "Entries"])
    path = pd.DataFrame(rows)
    grouped = path.groupby("Date")["PositionReturn"].agg(["mean", "count"]).rename(
        columns={"mean": "Return", "count": "ActivePositions"}
    )
    all_days = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    grouped = grouped.reindex(all_days, fill_value=0.0)
    grouped["Entries"] = [entry_counts.get(pd.Timestamp(day), 0) for day in grouped.index]
    return grouped


def _risk_stats(daily: pd.Series) -> dict[str, float]:
    if daily.empty:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0}
    std = float(daily.std())
    downside = float(daily[daily < 0].std())
    equity = (1 + daily).cumprod()
    return {
        "sharpe": float(daily.mean() / std * np.sqrt(252)) if np.isfinite(std) and std > 0 else 0.0,
        "sortino": float(daily.mean() / downside * np.sqrt(252)) if np.isfinite(downside) and downside > 0 else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()) if not equity.empty else 0.0,
    }


def _non_overlapping_trades(frame: pd.DataFrame) -> int:
    active = frame[frame["Position"] != 0].sort_values(["Symbol", "EntryDate", "ExitDate"])
    count = 0
    last_exit_by_symbol: dict[str, pd.Timestamp] = {}
    for _, row in active.iterrows():
        symbol = str(row["Symbol"])
        entry = pd.Timestamp(row["EntryDate"])
        exit_date = pd.Timestamp(row["ExitDate"])
        if symbol not in last_exit_by_symbol or entry > last_exit_by_symbol[symbol]:
            count += 1
            last_exit_by_symbol[symbol] = exit_date
    return count


def _concentration_stats(frame: pd.DataFrame) -> dict[str, float | None]:
    active = frame[frame["Position"] != 0]
    if active.empty:
        return {
            "top_symbol_profit_share": None,
            "top_1pct_trade_profit_share": None,
            "return_without_best_1pct": None,
            "return_without_best_5pct": None,
            "return_without_best_10pct": None,
        }
    positive_by_symbol = active.groupby("Symbol")["Return"].sum().clip(lower=0)
    positive_total = float(positive_by_symbol.sum())
    returns = active["Return"].astype(float).sort_values(ascending=False).reset_index(drop=True)
    positive_returns = returns[returns > 0]
    positive_trade_total = float(positive_returns.sum())

    def without_best(percent: float) -> float | None:
        if returns.empty:
            return None
        drop_n = max(1, int(np.ceil(len(returns) * percent)))
        kept = returns.iloc[drop_n:]
        return float(kept.mean()) if len(kept) else 0.0

    top_1pct_n = max(1, int(np.ceil(len(positive_returns) * 0.01))) if len(positive_returns) else 0
    return {
        "top_symbol_profit_share": float(positive_by_symbol.max() / positive_total) if positive_total > 0 else None,
        "top_1pct_trade_profit_share": (
            float(positive_returns.iloc[:top_1pct_n].sum() / positive_trade_total)
            if positive_trade_total > 0 and top_1pct_n else None
        ),
        "return_without_best_1pct": without_best(0.01),
        "return_without_best_5pct": without_best(0.05),
        "return_without_best_10pct": without_best(0.10),
    }


def cost_stress_summary(frame: pd.DataFrame, multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)) -> dict[str, dict]:
    if frame.empty:
        return {f"{multiplier:g}x": {"mean_return": 0.0, "profit_factor": None} for multiplier in multipliers}
    out: dict[str, dict] = {}
    for multiplier in multipliers:
        stressed = frame.copy()
        stressed["Return"] = stressed["GrossReturn"] - stressed["Position"].abs() * stressed["RoundTripCost"] * multiplier
        active_returns = stressed.loc[stressed["Position"] != 0, "Return"].astype(float)
        gains = active_returns[active_returns > 0].sum()
        losses = -active_returns[active_returns < 0].sum()
        out[f"{multiplier:g}x"] = {
            "mean_return": float(active_returns.mean()) if len(active_returns) else 0.0,
            "median_return": float(active_returns.median()) if len(active_returns) else 0.0,
            "profit_factor": float(gains / losses) if losses > 0 else (None if gains == 0 else float("inf")),
        }
    return out


def _current_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def experiment_manifest(
    config: ValidationConfig,
    symbols: list[str] | tuple[str, ...] | dict[str, str],
    *,
    commit_hash: str | None = None,
    data_fingerprint_value: str | None = None,
    data_ranges: dict | None = None,
) -> dict:
    universe = sorted(symbols.keys() if isinstance(symbols, dict) else symbols)
    commit = commit_hash or _current_commit()
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "commit": commit,
        "config": asdict(config),
        "universe": universe,
        "data_fingerprint": data_fingerprint_value,
        "data_ranges": data_ranges or {},
        "engine": "FittedForecastState+SignalInputs shared pipeline",
        "target": "close_to_close",
        "execution": "next_open",
    }
    experiment_id = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    run_id = f"{experiment_id}_{timestamp.replace(':', '').replace('-', '').replace('.', '')}"
    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "timestamp_utc": timestamp,
        **payload,
    }


def aggregate_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "observations": 0,
            "trades": 0,
            "rejected": 0,
            "non_overlapping_trades": 0,
            "mean_return": 0.0,
            "median_return": 0.0,
            "net_expectancy": 0.0,
            "profit_factor": None,
            "hit_rate": None,
            "exposure": 0.0,
            "avg_concurrent_positions": 0.0,
            "turnover_per_day": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "auc": 0.5,
            "brier": 0.25,
            "rejection_reasons": {},
            "cost_stress": {},
        }

    active = frame["Position"] != 0
    returns = frame.loc[active, "Return"].astype(float)
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    probability = frame["Probability"].clip(1e-4, 1 - 1e-4)
    rejection_reasons = frame.loc[~active, "DecisionReason"].value_counts().to_dict()
    timeline = _portfolio_timeline(frame)
    risk = _risk_stats(timeline["Return"] if not timeline.empty else pd.Series(dtype=float))
    ci = bootstrap_mean_return(frame, samples=250)
    active_days = timeline["ActivePositions"] > 0 if not timeline.empty else pd.Series(dtype=bool)
    return {
        "observations": int(len(frame)),
        "trades": int(active.sum()),
        "rejected": int((~active).sum()),
        "non_overlapping_trades": _non_overlapping_trades(frame),
        "mean_return": float(returns.mean()) if len(returns) else 0.0,
        "median_return": float(returns.median()) if len(returns) else 0.0,
        "net_expectancy": float(returns.mean()) if len(returns) else 0.0,
        "expectancy_ci_95": ci,
        "profit_factor": float(gains / losses) if losses > 0 else (None if gains == 0 else float("inf")),
        "hit_rate": float((returns > 0).mean()) if len(returns) else None,
        "exposure": float(active_days.mean()) if len(active_days) else 0.0,
        "avg_concurrent_positions": float(timeline["ActivePositions"].mean()) if not timeline.empty else 0.0,
        "turnover_per_day": float(timeline["Entries"].sum() / max(1, len(timeline))) if not timeline.empty else 0.0,
        **risk,
        "auc": float(roc_auc_score(frame["ActualUp"], probability)) if frame["ActualUp"].nunique() > 1 else 0.5,
        "brier": float(brier_score_loss(frame["ActualUp"], probability)),
        "avg_validation_auc": float(frame["ValidationAUC"].mean()),
        "avg_validation_brier": float(frame["ValidationBrier"].mean()),
        "rejection_reasons": {str(key): int(value) for key, value in rejection_reasons.items()},
        "benchmark_mean_returns": {
            "always_long": float(frame["AlwaysLongReturn"].mean()),
            "buy_hold_proxy": float(frame["BuyHoldReturn"].mean()),
            "momentum": float(frame["MomentumReturn"].mean()),
            "logistic_regression": float(frame["LinearReturn"].mean()),
        },
        "cost_stress": cost_stress_summary(frame),
        **_concentration_stats(frame),
    }


def group_summary(frame: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    grouped = frame.groupby(by, dropna=False)
    for key, group in grouped:
        summary = aggregate_summary(group)
        if not isinstance(key, tuple):
            key = (key,)
        keys = by if isinstance(by, list) else [by]
        rows.append({**dict(zip(keys, key)), **{k: v for k, v in summary.items() if k != "rejection_reasons"}})
    return pd.DataFrame(rows)


def bootstrap_mean_return(
    frame: pd.DataFrame, *, samples: int = 500, random_state: int = 42,
) -> tuple[float, float] | None:
    """Simple signal-level CI for active returns; conservative block bootstrap comes next."""
    returns = frame.loc[frame["Position"] != 0, "Return"].astype(float).to_numpy()
    if len(returns) < 20:
        return None
    rng = np.random.default_rng(random_state)
    means = [float(rng.choice(returns, size=len(returns), replace=True).mean()) for _ in range(samples)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def validation_report(
    frame: pd.DataFrame,
    config: ValidationConfig,
    symbols: list[str] | tuple[str, ...] | dict[str, str],
    *,
    commit_hash: str | None = None,
    data_fingerprint_value: str | None = None,
    data_ranges: dict | None = None,
) -> dict:
    """Produce an auditable experiment report without hiding weak groups."""
    walk_forward = frame[frame["FoldType"] != "HOLDOUT"].copy() if not frame.empty else frame
    holdout = frame[frame["FoldType"] == "HOLDOUT"].copy() if not frame.empty else frame
    fingerprint = data_fingerprint_value or frame.attrs.get("data_fingerprint")
    ranges = data_ranges or frame.attrs.get("data_ranges")
    return {
        "manifest": experiment_manifest(
            config,
            symbols,
            commit_hash=commit_hash,
            data_fingerprint_value=fingerprint,
            data_ranges=ranges,
        ),
        "summary": aggregate_summary(walk_forward),
        "holdout_summary": aggregate_summary(holdout),
        "combined_summary": aggregate_summary(frame),
        "by_market": group_summary(walk_forward, "Market").to_dict("records"),
        "by_symbol": group_summary(walk_forward, "Symbol").to_dict("records"),
        "by_horizon": group_summary(walk_forward, "Horizon").to_dict("records"),
        "by_fold": group_summary(frame, ["FoldType", "Fold"]).to_dict("records"),
    }


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_validation_artifacts(frame: pd.DataFrame, report: dict, output_dir: str | Path) -> dict[str, str]:
    """Persist raw records and report; manifest log is append-only."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(report["manifest"])
    run_id = str(manifest.get("run_id") or manifest["experiment_id"])
    records_path = _unique_path(output / f"records_{run_id}.csv")
    report_path = _unique_path(output / f"report_{run_id}.json")
    manifest_log = output / "manifest.jsonl"
    frame.to_csv(records_path, index=False)
    records_checksum = _file_sha256(records_path)
    report_to_write = dict(report)
    report_to_write["manifest"] = {
        **manifest,
        "artifacts": {
            "records": str(records_path),
            "records_sha256": records_checksum,
        },
    }
    report_path.write_text(json.dumps(report_to_write, ensure_ascii=False, indent=2, default=str))
    report_checksum = _file_sha256(report_path)
    manifest_entry = {
        **report_to_write["manifest"],
        "artifacts": {
            **report_to_write["manifest"]["artifacts"],
            "report": str(report_path),
            "report_sha256": report_checksum,
        },
    }
    with manifest_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest_entry, ensure_ascii=False, default=str) + "\n")
    return {
        "records": str(records_path),
        "report": str(report_path),
        "manifest_log": str(manifest_log),
        "records_sha256": records_checksum,
        "report_sha256": report_checksum,
    }
