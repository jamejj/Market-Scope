from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import pandas as pd

from market_oracle.reality import RealityConfig, reality_check_report


DEFAULT_RECORDS_GLOB = "records_*.csv"
PIPELINE_FILES = ("run_reality_check.py", "market_oracle/reality.py")


def _parse_csv_tuple(raw: str | None, *, upper: bool = False) -> tuple[str, ...] | None:
    if not raw:
        return None
    values = []
    for item in raw.split(","):
        value = item.strip()
        if value:
            values.append(value.upper() if upper else value)
    return tuple(values) or None


def _parse_horizons(raw: str | None) -> tuple[int, ...] | None:
    if not raw:
        return None
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def _latest_records(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob(DEFAULT_RECORDS_GLOB), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"Nie znalazłem pliku {DEFAULT_RECORDS_GLOB} w {output_dir}. Podaj --records.")
    return candidates[0]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.upper())


def _cache_path(cache_dir: Path, symbol: str, years: int) -> Path:
    return cache_dir / f"{_safe_name(symbol)}_{years}y.pkl"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        file_hash = hashlib.sha256(data).hexdigest()
        files[relative] = file_hash
        digest.update(relative.encode())
        digest.update(data)
    return {
        "git_commit": _current_commit(),
        "pipeline_hash": digest.hexdigest(),
        "files": files,
    }


def _load_cached_history(cache_dir: Path, symbol: str, years: int) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    path = _cache_path(cache_dir, symbol, years)
    if not path.exists():
        return None, None
    frame = pd.read_pickle(path)
    meta = {
        "symbol": symbol,
        "path": str(path),
        "sha256": _file_sha256(path),
        "rows": int(len(frame)),
        "start": str(pd.to_datetime(frame.index).min().date()) if len(frame) else None,
        "end": str(pd.to_datetime(frame.index).max().date()) if len(frame) else None,
    }
    return frame, meta


def _load_histories(
    records: pd.DataFrame,
    cache_dir: Path,
    years: int,
    benchmark_symbol: str | None,
) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, Any]]:
    symbols = sorted({str(symbol) for symbol in records["Symbol"].dropna().unique()})
    if benchmark_symbol:
        symbols.append(benchmark_symbol)
    histories: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    metadata: dict[str, Any] = {}
    for symbol in sorted(set(symbols)):
        history, meta = _load_cached_history(cache_dir, symbol, years)
        if history is None or history.empty:
            missing.append(symbol)
            continue
        histories[symbol] = history
        metadata[symbol] = meta
    return histories, missing, metadata


def _json_safe(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    if isinstance(payload, (pd.Timestamp, datetime)):
        return payload.isoformat()
    if isinstance(payload, float):
        return payload if pd.notna(payload) and payload not in {float("inf"), float("-inf")} else None
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{int(time.time() * 1_000_000)}.tmp")
    tmp.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f%z")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MarketScope Reality Check on saved Aggregate Validation records.")
    parser.add_argument("--records", help="Path to records_*.csv. Default: newest in --validation-dir.")
    parser.add_argument("--validation-dir", default="outputs/validation")
    parser.add_argument("--cache-dir", default="outputs/validation/cache")
    parser.add_argument("--output-dir", default="outputs/reality")
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--horizons", default="20", help="Comma list. Default focuses on the candidate 20d edge.")
    parser.add_argument("--symbols", help="Optional comma list of symbols.")
    parser.add_argument("--markets", help="Optional comma list of markets/classes.")
    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Global concurrent-position cap. Use 0 for no extra cap beyond the number of capital slots.",
    )
    parser.add_argument("--portfolio-slots", type=int, default=5, help="Fixed capital slots; 5 means 20% per new position.")
    parser.add_argument("--benchmark-symbol", default="SPY", help="Use SPY for USA/ETF, BTC-USD for crypto-only checks.")
    parser.add_argument("--annualization-days", type=int, help="Override inferred Sharpe annualization.")
    parser.add_argument("--allow-same-day-reentry", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--no-strict-history", action="store_true", help="Do not fail on missing/mismatched cached Open prices.")
    args = parser.parse_args()

    validation_dir = Path(args.validation_dir)
    records_path = Path(args.records) if args.records else _latest_records(validation_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)

    records = pd.read_csv(records_path)
    max_positions = None if args.max_positions == 0 else args.max_positions
    config = RealityConfig(
        horizons=_parse_horizons(args.horizons),
        symbols=_parse_csv_tuple(args.symbols, upper=True),
        markets=_parse_csv_tuple(args.markets, upper=True),
        allow_same_day_reentry=args.allow_same_day_reentry,
        max_positions=max_positions,
        portfolio_slots=args.portfolio_slots,
        benchmark_symbol=args.benchmark_symbol or None,
        annualization_days=args.annualization_days,
        bootstrap_samples=args.bootstrap_samples,
        strict_history=not args.no_strict_history,
    )
    histories, missing_cache, cache_metadata = _load_histories(records, cache_dir, args.years, config.benchmark_symbol)
    report, selected, curve = reality_check_report(records, histories, config)
    source_sha256 = _file_sha256(records_path)
    pipeline = _pipeline_fingerprint()
    report["manifest"] = {
        "source_records": str(records_path),
        "source_records_sha256": source_sha256,
        "source_rows": int(len(records)),
        "cache_dir": str(cache_dir),
        "missing_cache": missing_cache,
        "cache_metadata": cache_metadata,
        "pipeline": pipeline,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    stem = f"{records_path.stem}_h{'-'.join(str(h) for h in (config.horizons or (0,)))}_{_stamp()}"
    trades_path = output_dir / f"reality_trades_{stem}.csv"
    curve_path = output_dir / f"reality_curve_{stem}.csv"
    report_path = output_dir / f"reality_report_{stem}.json"
    manifest_path = output_dir / f"reality_manifest_{stem}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(trades_path, index=False)
    curve.to_csv(curve_path, index=False)
    report["manifest"]["artifacts"] = {
        "trades": {"path": str(trades_path), "sha256": _file_sha256(trades_path), "rows": int(len(selected))},
        "curve": {"path": str(curve_path), "sha256": _file_sha256(curve_path), "rows": int(len(curve))},
    }
    _write_json(report_path, report)
    manifest = {
        **report["manifest"],
        "report": {"path": str(report_path), "sha256": _file_sha256(report_path)},
    }
    _write_json(manifest_path, manifest)

    print("REALITY_ARTIFACTS", json.dumps({
        "report": str(report_path),
        "trades": str(trades_path),
        "curve": str(curve_path),
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2), flush=True)
    print("SUMMARY", json.dumps(report["summary"], ensure_ascii=False, indent=2, default=str), flush=True)
    print("BY_HORIZON", json.dumps(report["by_horizon"], ensure_ascii=False, default=str), flush=True)
    print("BY_SYMBOL", json.dumps(report["by_symbol"], ensure_ascii=False, default=str), flush=True)
    print("BY_FOLD", json.dumps(report["by_fold"], ensure_ascii=False, default=str), flush=True)
    if missing_cache or report["price_issues"]:
        print("PRICE_OR_CACHE_AUDIT", json.dumps({
            "cache": missing_cache,
            "price_issues": report["price_issues"],
        }, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
