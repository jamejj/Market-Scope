from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import pandas as pd

import market_oracle.validation as validation_module
from market_oracle.data import download_history
from market_oracle.engine import _benchmark_for
from market_oracle.validation import (
    ValidationConfig,
    aggregate_summary,
    data_fingerprint,
    group_summary,
    save_validation_artifacts,
    validate_history,
    validation_report,
)


DEFAULT_SYMBOLS = {
    "AAPL": "USA",
    "MSFT": "USA",
    "NVDA": "USA",
    "SPY": "ETF",
    "QQQ": "ETF",
    "BTC-USD": "CRYPTO",
    "ETH-USD": "CRYPTO",
}

PIPELINE_FILES = (
    "run_validation.py",
    "market_oracle/model.py",
    "market_oracle/features.py",
    "market_oracle/validation.py",
    "market_oracle/backtest.py",
    "market_oracle/signals.py",
    "market_oracle/cutoff.py",
    "market_oracle/engine.py",
)

REQUIRED_RECORD_COLUMNS = {
    "Date", "Symbol", "Market", "Horizon", "Fold", "FoldType",
    "TrainEndDate", "AvailableTrainEndDate", "Position", "DecisionReason",
    "Probability", "ExpectedReturn", "ValidationAUC", "ValidationBrier",
    "Return", "ActualUp", "Target", "Execution",
}


def _parse_symbols(raw: str | None) -> dict[str, str]:
    if not raw:
        return DEFAULT_SYMBOLS
    out: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            symbol, market = item.split(":", 1)
            out[symbol.strip().upper()] = market.strip().upper()
        else:
            out[item.upper()] = "Unknown"
    return out


def _parse_horizons(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.upper())


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{int(time.time() * 1_000_000)}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_replace_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_paths(cache_dir: Path, symbol: str, years: int) -> tuple[Path, Path]:
    name = f"{_safe_name(symbol)}_{years}y"
    return cache_dir / f"{name}.pkl", cache_dir / f"{name}.json"


def _load_or_download_history(symbol: str, years: int, cache_dir: Path, refresh_cache: bool) -> pd.DataFrame:
    data_path, meta_path = _cache_paths(cache_dir, symbol, years)
    meta = _read_json(meta_path)
    if not refresh_cache and data_path.exists() and meta and meta.get("symbol") == symbol and meta.get("years") == years:
        frame = pd.read_pickle(data_path)
        print(f"CACHE_HIT {symbol} rows={len(frame)} path={data_path}", flush=True)
        return frame

    print(f"DOWNLOAD_START {symbol}", flush=True)
    frame = download_history(symbol, years=years)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(data_path)
    _write_json(meta_path, {
        "symbol": symbol,
        "years": int(years),
        "rows": int(len(frame)),
        "start": str(frame.index.min().date()),
        "end": str(frame.index.max().date()),
        "cached_at_utc": _utc_now(),
    })
    print(f"DOWNLOAD_DONE {symbol} rows={len(frame)} cache={data_path}", flush=True)
    return frame


def _cache_metadata(symbol: str, years: int, cache_dir: Path) -> dict[str, Any]:
    _, meta_path = _cache_paths(cache_dir, symbol, years)
    meta = _read_json(meta_path) or {}
    return {
        "symbol": symbol,
        "years": int(years),
        "rows": meta.get("rows"),
        "start": meta.get("start"),
        "end": meta.get("end"),
        "cached_at_utc": meta.get("cached_at_utc"),
    }


def _current_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _pipeline_fingerprint() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for relative in PIPELINE_FILES:
        path = root / relative
        if not path.exists():
            continue
        data = path.read_bytes()
        file_hash = hashlib.sha256(data).hexdigest()[:16]
        files[relative] = file_hash
        digest.update(relative.encode())
        digest.update(data)
    return {
        "git_commit": _current_commit(),
        "pipeline_hash": digest.hexdigest()[:24],
        "files": files,
    }


def _job_key(symbol: str, horizon: int) -> str:
    return f"{_safe_name(symbol)}_h{horizon}"


def _job_paths(job_dir: Path, job_key: str) -> dict[str, Path]:
    return {
        "records": job_dir / f"{job_key}.csv",
        "status": job_dir / f"{job_key}.json",
    }


def _records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_records(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{int(time.time() * 1_000_000)}.tmp")
    _records_frame(records).to_csv(tmp, index=False)
    tmp.replace(path)
    return {
        "rows": int(len(records)),
        "records_sha256": _file_sha256(path),
    }


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    current = _read_json(path) or {}
    _write_json(path, {**current, **payload, "updated_at_utc": _utc_now()})


def _write_jobs_index(job_dir: Path, statuses: dict[str, dict[str, Any]], experiment: dict[str, Any]) -> Path:
    path = job_dir / "jobs_status.json"
    counts = pd.Series([status.get("status", "PENDING") for status in statuses.values()]).value_counts().to_dict()
    _write_json(path, {
        "experiment": experiment,
        "counts": {str(key): int(value) for key, value in counts.items()},
        "jobs": list(statuses.values()),
        "updated_at_utc": _utc_now(),
    })
    return path


def _read_and_validate_job_records(
    path: Path, rows: int, expected_sha256: str | None,
) -> tuple[bool, pd.DataFrame, str | None]:
    if rows <= 0:
        return True, pd.DataFrame(), None
    if not path.exists():
        return False, pd.DataFrame(), "records_missing"
    if expected_sha256 and _file_sha256(path) != expected_sha256:
        return False, pd.DataFrame(), "records_checksum_mismatch"
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return False, pd.DataFrame(), f"records_unreadable:{exc}"
    if len(frame) != rows:
        return False, pd.DataFrame(), f"row_count_mismatch:status={rows}:file={len(frame)}"
    missing = sorted(REQUIRED_RECORD_COLUMNS - set(frame.columns))
    if missing:
        return False, pd.DataFrame(), f"missing_columns:{','.join(missing)}"
    return True, frame, None


def _valid_completed_job(status: dict[str, Any] | None, records_path: Path, config_hash: str, fingerprint: str) -> bool:
    if not status or status.get("status") != "DONE":
        return False
    if status.get("config_hash") != config_hash or status.get("data_fingerprint") != fingerprint:
        return False
    rows = int(status.get("rows", 0))
    ok, _, _ = _read_and_validate_job_records(records_path, rows, status.get("records_sha256"))
    return ok


def _read_job_records(path: Path, rows: int, sha256: str | None) -> pd.DataFrame:
    ok, frame, reason = _read_and_validate_job_records(path, rows, sha256)
    if not ok:
        raise RuntimeError(f"Nieprawidłowy plik joba {path}: {reason}")
    return frame


def _job_config(
    symbol: str,
    market: str,
    horizon: int,
    years: int,
    benchmark: str | None,
    config: ValidationConfig,
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "market": market,
        "horizon": int(horizon),
        "years": int(years),
        "benchmark": benchmark,
        "config": asdict(config),
        "pipeline": pipeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable MarketScope Aggregate Validation.")
    parser.add_argument("--symbols", help="Comma list, e.g. AAPL:USA,SPY:ETF,BTC-USD:CRYPTO")
    parser.add_argument("--horizons", default="1,5,20")
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--initial-train", type=int, default=420)
    parser.add_argument("--test-size", type=int, default=90)
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--holdout-size", type=int, default=0)
    parser.add_argument("--refit-every", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="outputs/validation")
    parser.add_argument("--cache-dir", help="Default: <output-dir>/cache")
    parser.add_argument("--refresh-cache", action="store_true", help="Download fresh histories even when cache exists.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore completed job files and recompute all jobs.")
    parser.add_argument("--log-every-fits", type=int, default=5)
    parser.add_argument("--save-every-records", type=int, default=25)
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    horizons = _parse_horizons(args.horizons)
    config = ValidationConfig(
        horizons=horizons,
        initial_train=args.initial_train,
        test_size=args.test_size,
        max_folds=args.max_folds,
        holdout_size=args.holdout_size,
        refit_every=args.refit_every,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
    )
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    pipeline = _pipeline_fingerprint()
    experiment = {
        "id": _json_hash({
            "symbols": symbols,
            "horizons": horizons,
            "years": args.years,
            "config": asdict(config),
            "pipeline": pipeline,
        }),
        "requested_universe": symbols,
        "horizons": horizons,
        "years": int(args.years),
        "config": asdict(config),
        "pipeline": pipeline,
    }
    job_dir = output_dir / "jobs" / experiment["id"]
    job_dir.mkdir(parents=True, exist_ok=True)

    print("EXPERIMENT", json.dumps(experiment, ensure_ascii=False, default=str), flush=True)
    print(
        "UWAGA: refit_every=1/5 jest ciężki obliczeniowo. Runner ma cache i job-level resume dla jobów symbol×horizon.",
        flush=True,
    )

    histories: dict[str, pd.DataFrame] = {}
    contexts: dict[str, pd.DataFrame | None] = {}
    benchmarks: dict[str, str | None] = {}
    context_required: dict[str, bool] = {}
    context_ok: dict[str, bool] = {}
    cache_metadata: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for symbol in symbols:
        try:
            histories[symbol] = _load_or_download_history(symbol, args.years, cache_dir, args.refresh_cache)
            cache_metadata[symbol] = _cache_metadata(symbol, args.years, cache_dir)
            benchmark = _benchmark_for(symbol)
            benchmarks[symbol] = benchmark
            context_required[symbol] = bool(benchmark)
            if benchmark:
                try:
                    print(f"CONTEXT_START {symbol} benchmark={benchmark}", flush=True)
                    contexts[symbol] = _load_or_download_history(benchmark, args.years, cache_dir, args.refresh_cache)
                    cache_metadata[f"context:{symbol}"] = _cache_metadata(benchmark, args.years, cache_dir)
                    context_ok[symbol] = True
                except Exception as exc:
                    contexts[symbol] = None
                    context_ok[symbol] = False
                    errors[f"context:{symbol}"] = str(exc)
                    print(f"CONTEXT_ERROR {symbol} {exc}", flush=True)
            else:
                contexts[symbol] = None
                context_ok[symbol] = True
        except Exception as exc:
            context_required[symbol] = False
            context_ok[symbol] = False
            errors[symbol] = str(exc)
            print(f"DOWNLOAD_ERROR {symbol} {exc}", flush=True)

    jobs: list[dict[str, Any]] = []
    statuses: dict[str, dict[str, Any]] = {}
    for symbol, market in symbols.items():
        if symbol not in histories:
            for horizon in horizons:
                cfg = ValidationConfig(
                    horizons=(horizon,),
                    initial_train=args.initial_train,
                    test_size=args.test_size,
                    max_folds=args.max_folds,
                    holdout_size=args.holdout_size,
                    refit_every=args.refit_every,
                    cost_bps=args.cost_bps,
                    slippage_bps=args.slippage_bps,
                )
                key = _job_key(symbol, horizon)
                paths = _job_paths(job_dir, key)
                status = {
                    "job_key": key,
                    "symbol": symbol,
                    "market": market,
                    "horizon": int(horizon),
                    "status": "FAILED",
                    "error": errors.get(symbol, "history_unavailable"),
                    "reason": "HISTORY_UNAVAILABLE",
                    "config_hash": _json_hash(_job_config(symbol, market, horizon, args.years, None, cfg, pipeline)),
                    "data_fingerprint": None,
                    "pipeline": pipeline,
                    "records_path": str(paths["records"]),
                    "status_path": str(paths["status"]),
                    "rows": 0,
                    "records_sha256": None,
                }
                statuses[key] = status
                _write_status(paths["status"], status)
            continue
        if context_required.get(symbol) and not context_ok.get(symbol):
            for horizon in horizons:
                cfg = ValidationConfig(
                    horizons=(horizon,),
                    initial_train=args.initial_train,
                    test_size=args.test_size,
                    max_folds=args.max_folds,
                    holdout_size=args.holdout_size,
                    refit_every=args.refit_every,
                    cost_bps=args.cost_bps,
                    slippage_bps=args.slippage_bps,
                )
                fingerprint, ranges = data_fingerprint({symbol: histories[symbol]}, {})
                key = _job_key(symbol, horizon)
                paths = _job_paths(job_dir, key)
                config_payload = _job_config(symbol, market, horizon, args.years, benchmarks.get(symbol), cfg, pipeline)
                status = {
                    "job_key": key,
                    "symbol": symbol,
                    "market": market,
                    "horizon": int(horizon),
                    "status": "FAILED",
                    "error": errors.get(f"context:{symbol}", "context_unavailable"),
                    "reason": "CONTEXT_UNAVAILABLE",
                    "config_hash": _json_hash(config_payload),
                    "data_fingerprint": fingerprint,
                    "data_ranges": ranges,
                    "data_as_of": str(histories[symbol].index.max().date()),
                    "context_required": True,
                    "context_available": False,
                    "pipeline": pipeline,
                    "records_path": str(paths["records"]),
                    "status_path": str(paths["status"]),
                    "rows": 0,
                    "records_sha256": None,
                }
                statuses[key] = status
                _write_status(paths["status"], status)
            continue
        for horizon in horizons:
            cfg = ValidationConfig(
                horizons=(horizon,),
                initial_train=args.initial_train,
                test_size=args.test_size,
                max_folds=args.max_folds,
                holdout_size=args.holdout_size,
                refit_every=args.refit_every,
                cost_bps=args.cost_bps,
                slippage_bps=args.slippage_bps,
            )
            fingerprint, ranges = data_fingerprint({symbol: histories[symbol]}, {symbol: contexts.get(symbol)})
            config_payload = _job_config(symbol, market, horizon, args.years, benchmarks.get(symbol), cfg, pipeline)
            config_hash = _json_hash(config_payload)
            key = _job_key(symbol, horizon)
            paths = _job_paths(job_dir, key)
            previous_status = _read_json(paths["status"]) or {}
            completed_valid = _valid_completed_job(previous_status, paths["records"], config_hash, fingerprint)
            status = {
                "job_key": key,
                "symbol": symbol,
                "market": market,
                "horizon": int(horizon),
                "status": "DONE" if completed_valid else "PENDING",
                "config_hash": config_hash,
                "data_fingerprint": fingerprint,
                "data_ranges": ranges,
                "data_as_of": str(histories[symbol].index.max().date()),
                "context_as_of": (
                    str(contexts[symbol].index.max().date())
                    if contexts.get(symbol) is not None and not contexts[symbol].empty else None
                ),
                "context_required": bool(context_required.get(symbol)),
                "context_available": bool(context_ok.get(symbol)),
                "cache_metadata": {
                    "symbol": cache_metadata.get(symbol),
                    "context": cache_metadata.get(f"context:{symbol}"),
                },
                "pipeline": pipeline,
                "records_path": str(paths["records"]),
                "status_path": str(paths["status"]),
                "rows": int(previous_status.get("rows", 0)) if completed_valid else 0,
                "records_sha256": previous_status.get("records_sha256") if completed_valid else None,
            }
            jobs.append({
                "symbol": symbol,
                "market": market,
                "horizon": int(horizon),
                "config": cfg,
                "fingerprint": fingerprint,
                "config_hash": config_hash,
                "paths": paths,
                "key": key,
            })
            statuses[key] = status
            _write_status(paths["status"], status)
    index_path = _write_jobs_index(job_dir, statuses, experiment)
    print(f"JOB_INDEX {index_path}", flush=True)

    original_fit = validation_module.fit_forecast_state
    fit_counter = {"n": 0}
    current = {"symbol": "?", "horizon": "?"}

    def logged_fit(X, y, returns, horizon):
        fit_counter["n"] += 1
        n = fit_counter["n"]
        if n == 1 or n % max(1, args.log_every_fits) == 0:
            end = X.index[-1].date() if len(X) else "none"
            print(
                f"FIT {n} symbol={current['symbol']} horizon={current['horizon']} train={len(X)} train_end={end}",
                flush=True,
            )
        return original_fit(X, y, returns, horizon)

    validation_module.fit_forecast_state = logged_fit
    frames: list[pd.DataFrame] = []
    started = time.time()
    for job in jobs:
        symbol = str(job["symbol"])
        market = str(job["market"])
        horizon = int(job["horizon"])
        key = str(job["key"])
        paths = job["paths"]
        previous_status = _read_json(paths["status"])
        if not args.no_resume and _valid_completed_job(previous_status, paths["records"], job["config_hash"], job["fingerprint"]):
            rows = int(previous_status.get("rows", 0))
            print(f"RESUME_SKIP {key} rows={rows}", flush=True)
            part = _read_job_records(paths["records"], rows, previous_status.get("records_sha256"))
            if not part.empty:
                frames.append(part)
            statuses[key] = {**statuses[key], **previous_status, "status": "DONE"}
            _write_jobs_index(job_dir, statuses, experiment)
            continue

        current["symbol"] = symbol
        current["horizon"] = str(horizon)
        job_records: list[dict[str, Any]] = []
        t0 = time.time()
        _write_status(paths["status"], {
            **statuses[key],
            "status": "RUNNING",
            "started_at_utc": _utc_now(),
            "rows": 0,
            "records_sha256": None,
            "error": None,
        })
        statuses[key] = {**statuses[key], "status": "RUNNING", "rows": 0, "records_sha256": None}
        _write_jobs_index(job_dir, statuses, experiment)
        print(f"VALIDATE_START {key} market={market}", flush=True)

        def on_record(record: dict[str, Any]) -> None:
            job_records.append(record)
            if len(job_records) % max(1, args.save_every_records) == 0:
                record_meta = _write_records(paths["records"], job_records)
                _write_status(paths["status"], {"status": "RUNNING", **record_meta})

        def on_refit(info: dict[str, Any]) -> None:
            if job_records:
                record_meta = _write_records(paths["records"], job_records)
                _write_status(paths["status"], {
                    "status": "RUNNING",
                    **record_meta,
                    "last_refit": info,
                })
            print(
                f"REFIT {key} fold={info['fold']} date={pd.Timestamp(info['date']).date()} train={info['available_train_end']}",
                flush=True,
            )

        try:
            part = validate_history(
                symbol,
                histories[symbol],
                market=market,
                context=contexts.get(symbol),
                config=job["config"],
                record_callback=on_record,
                refit_callback=on_refit,
            )
            job_records = part.to_dict("records")
            record_meta = _write_records(paths["records"], job_records)
            _write_status(paths["status"], {
                **statuses[key],
                "status": "DONE",
                **record_meta,
                "seconds": round(time.time() - t0, 2),
                "finished_at_utc": _utc_now(),
                "error": None,
            })
            statuses[key] = {**statuses[key], "status": "DONE", **record_meta}
            if not part.empty:
                frames.append(part)
            trades = int((part["Position"] != 0).sum()) if not part.empty else 0
            print(f"VALIDATE_DONE {key} rows={len(part)} trades={trades} seconds={time.time() - t0:.1f}", flush=True)
        except KeyboardInterrupt:
            if job_records:
                record_meta = _write_records(paths["records"], job_records)
            else:
                record_meta = {"rows": 0, "records_sha256": None}
            _write_status(paths["status"], {
                **statuses[key],
                "status": "INTERRUPTED",
                **record_meta,
                "interrupted_at_utc": _utc_now(),
            })
            statuses[key] = {**statuses[key], "status": "INTERRUPTED", **record_meta}
            _write_jobs_index(job_dir, statuses, experiment)
            print(f"INTERRUPTED {key} rows_saved={len(job_records)}", flush=True)
            raise
        except Exception as exc:
            errors[f"{symbol}:{horizon}"] = str(exc)
            if job_records:
                record_meta = _write_records(paths["records"], job_records)
            else:
                record_meta = {"rows": 0, "records_sha256": None}
            _write_status(paths["status"], {
                **statuses[key],
                "status": "FAILED",
                **record_meta,
                "error": str(exc),
                "failed_at_utc": _utc_now(),
            })
            statuses[key] = {**statuses[key], "status": "FAILED", **record_meta, "error": str(exc)}
            print(f"VALIDATE_ERROR {key} {exc}", flush=True)
        finally:
            _write_jobs_index(job_dir, statuses, experiment)

    frame = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    coverage: dict[str, dict[str, str]] = {
        symbol: {str(horizon): statuses.get(_job_key(symbol, horizon), {}).get("status", "MISSING") for horizon in horizons}
        for symbol in symbols
    }
    complete_symbols = {
        symbol: market
        for symbol, market in symbols.items()
        if all(coverage.get(symbol, {}).get(str(horizon)) == "DONE" for horizon in horizons)
    }
    symbols_with_rows = (
        {str(symbol): symbols.get(str(symbol), "Unknown") for symbol in sorted(frame["Symbol"].unique())}
        if not frame.empty else {}
    )
    completed_histories = {symbol: histories[symbol] for symbol in symbols_with_rows if symbol in histories}
    completed_contexts = {symbol: contexts.get(symbol) for symbol in symbols_with_rows}
    fingerprint, ranges = data_fingerprint(completed_histories, completed_contexts)
    frame.attrs["data_fingerprint"] = fingerprint
    frame.attrs["data_ranges"] = ranges
    data_as_of = {
        symbol: str(histories[symbol].index.max().date())
        for symbol in histories
    }
    runner_manifest = {
        "experiment": experiment,
        "requested_universe": symbols,
        "completed_universe": complete_symbols,
        "symbols_with_rows": symbols_with_rows,
        "job_coverage": coverage,
        "data_as_of": data_as_of,
        "cache_metadata": cache_metadata,
        "errors": errors,
        "job_index": str(index_path),
        "rows": int(len(frame)),
        "seconds": round(time.time() - started, 2),
        "updated_at_utc": _utc_now(),
    }
    runner_manifest_path = job_dir / "runner_manifest.json"
    _write_json(runner_manifest_path, runner_manifest)
    print("DOWNLOAD_OR_RUNTIME_ERRORS", json.dumps(errors, ensure_ascii=False, indent=2, default=str), flush=True)
    print("COMPLETED_UNIVERSE", json.dumps(complete_symbols, ensure_ascii=False, default=str), flush=True)
    print("JOB_COVERAGE", json.dumps(coverage, ensure_ascii=False, default=str), flush=True)
    print("ROWS", len(frame), "SECONDS", round(time.time() - started, 1), flush=True)
    print(f"RUNNER_MANIFEST {runner_manifest_path}", flush=True)
    if frame.empty:
        raise SystemExit("No validation rows produced.")

    report = validation_report(frame, config, symbols_with_rows)
    report["manifest"] = {
        **report["manifest"],
        "requested_universe": symbols,
        "completed_universe": complete_symbols,
        "symbols_with_rows": symbols_with_rows,
        "job_coverage": coverage,
        "runner_manifest": str(runner_manifest_path),
        "data_as_of": data_as_of,
        "cache_metadata": cache_metadata,
        "pipeline": pipeline,
    }
    written = save_validation_artifacts(frame, report, output_dir)
    summary = aggregate_summary(frame)
    print("ARTIFACTS", json.dumps(written, ensure_ascii=False, indent=2, default=str), flush=True)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    print("BY_MARKET", group_summary(frame, "Market").to_json(orient="records", force_ascii=False), flush=True)
    print("BY_SYMBOL", group_summary(frame, "Symbol").to_json(orient="records", force_ascii=False), flush=True)
    print("BY_HORIZON", group_summary(frame, "Horizon").to_json(orient="records", force_ascii=False), flush=True)
    print("BY_FOLD", group_summary(frame, ["FoldType", "Fold"]).to_json(orient="records", force_ascii=False), flush=True)
    print("DECISION_REASONS", frame["DecisionReason"].value_counts().to_json(force_ascii=False), flush=True)
    print("QUALITY_COUNTS", frame["Quality"].value_counts().to_json(force_ascii=False), flush=True)
    print("POSITIONS", frame["Position"].value_counts().sort_index().to_json(force_ascii=False), flush=True)


if __name__ == "__main__":
    main()
