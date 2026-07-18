from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from market_oracle.reality import RealityConfig, reality_check_report


DEFAULT_RECORDS_GLOB = "records_*.csv"


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


def _load_cached_history(cache_dir: Path, symbol: str, years: int) -> pd.DataFrame | None:
    path = _cache_path(cache_dir, symbol, years)
    if not path.exists():
        return None
    return pd.read_pickle(path)


def _load_histories(records: pd.DataFrame, cache_dir: Path, years: int, benchmark_symbol: str | None) -> tuple[dict[str, pd.DataFrame], list[str]]:
    symbols = sorted({str(symbol) for symbol in records["Symbol"].dropna().unique()})
    if benchmark_symbol:
        symbols.append(benchmark_symbol)
    histories: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in sorted(set(symbols)):
        history = _load_cached_history(cache_dir, symbol, years)
        if history is None or history.empty:
            missing.append(symbol)
            continue
        histories[symbol] = history
    return histories, missing


def _json_safe(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    if isinstance(payload, (pd.Timestamp, datetime)):
        return payload.isoformat()
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
    parser.add_argument("--max-positions", type=int, help="Optional global concurrent-position cap.")
    parser.add_argument("--benchmark-symbol", default="SPY")
    parser.add_argument("--allow-same-day-reentry", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    validation_dir = Path(args.validation_dir)
    records_path = Path(args.records) if args.records else _latest_records(validation_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)

    records = pd.read_csv(records_path)
    config = RealityConfig(
        horizons=_parse_horizons(args.horizons),
        symbols=_parse_csv_tuple(args.symbols, upper=True),
        markets=_parse_csv_tuple(args.markets, upper=True),
        allow_same_day_reentry=args.allow_same_day_reentry,
        max_positions=args.max_positions,
        benchmark_symbol=args.benchmark_symbol or None,
        bootstrap_samples=args.bootstrap_samples,
    )
    histories, missing_cache = _load_histories(records, cache_dir, args.years, config.benchmark_symbol)
    report, selected, curve = reality_check_report(records, histories, config)
    report["manifest"] = {
        "source_records": str(records_path),
        "source_rows": int(len(records)),
        "cache_dir": str(cache_dir),
        "missing_cache": missing_cache,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    stem = f"{records_path.stem}_h{'-'.join(str(h) for h in (config.horizons or (0,)))}_{_stamp()}"
    trades_path = output_dir / f"reality_trades_{stem}.csv"
    curve_path = output_dir / f"reality_curve_{stem}.csv"
    report_path = output_dir / f"reality_report_{stem}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(trades_path, index=False)
    curve.to_csv(curve_path, index=False)
    _write_json(report_path, report)

    print("REALITY_ARTIFACTS", json.dumps({
        "report": str(report_path),
        "trades": str(trades_path),
        "curve": str(curve_path),
    }, ensure_ascii=False, indent=2), flush=True)
    print("SUMMARY", json.dumps(report["summary"], ensure_ascii=False, indent=2, default=str), flush=True)
    print("BY_HORIZON", json.dumps(report["by_horizon"], ensure_ascii=False, default=str), flush=True)
    print("BY_SYMBOL", json.dumps(report["by_symbol"], ensure_ascii=False, default=str), flush=True)
    print("BY_FOLD", json.dumps(report["by_fold"], ensure_ascii=False, default=str), flush=True)
    if missing_cache or report["missing_histories"]:
        print("MISSING_HISTORIES", json.dumps({
            "cache": missing_cache,
            "used_by_trades": report["missing_histories"],
        }, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
