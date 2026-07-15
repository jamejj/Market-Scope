from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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


def _write_partial(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run auditable MarketScope Aggregate Validation.")
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
    parser.add_argument("--log-every-fits", type=int, default=5)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / f"partial_{int(time.time())}.csv"

    print("CONFIG", json.dumps(config.__dict__, ensure_ascii=False, default=str), flush=True)
    print("SYMBOLS", json.dumps(symbols, ensure_ascii=False), flush=True)
    print(
        "UWAGA: refit_every=1/5 jest ciężki obliczeniowo. Ten runner zapisuje częściowe rekordy po każdym symbolu/horyzoncie.",
        flush=True,
    )

    histories: dict[str, pd.DataFrame] = {}
    contexts: dict[str, pd.DataFrame | None] = {}
    errors: dict[str, str] = {}
    for symbol in symbols:
        print(f"DOWNLOAD_START {symbol}", flush=True)
        try:
            histories[symbol] = download_history(symbol, years=args.years)
            benchmark = _benchmark_for(symbol)
            if benchmark:
                try:
                    print(f"CONTEXT_START {symbol} benchmark={benchmark}", flush=True)
                    contexts[symbol] = download_history(benchmark, years=args.years)
                except Exception as exc:
                    contexts[symbol] = None
                    errors[f"context:{symbol}"] = str(exc)
            else:
                contexts[symbol] = None
            print(f"DOWNLOAD_DONE {symbol} rows={len(histories[symbol])}", flush=True)
        except Exception as exc:
            errors[symbol] = str(exc)
            print(f"DOWNLOAD_ERROR {symbol} {exc}", flush=True)

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
    for symbol, market in symbols.items():
        if symbol not in histories:
            continue
        for horizon in horizons:
            current["symbol"] = symbol
            current["horizon"] = str(horizon)
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
            print(f"VALIDATE_START {symbol} market={market} horizon={horizon}", flush=True)
            t0 = time.time()
            try:
                part = validate_history(
                    symbol,
                    histories[symbol],
                    market=market,
                    context=contexts.get(symbol),
                    config=cfg,
                )
                frames.append(part)
                partial = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
                _write_partial(partial, partial_path)
                trades = int((part["Position"] != 0).sum()) if not part.empty else 0
                print(
                    f"VALIDATE_DONE {symbol} horizon={horizon} rows={len(part)} trades={trades} seconds={time.time() - t0:.1f}",
                    flush=True,
                )
                print(f"PARTIAL_RECORDS {partial_path}", flush=True)
            except Exception as exc:
                errors[f"{symbol}:{horizon}"] = str(exc)
                print(f"VALIDATE_ERROR {symbol} horizon={horizon} {exc}", flush=True)

    frame = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    fingerprint, ranges = data_fingerprint(histories, contexts)
    frame.attrs["data_fingerprint"] = fingerprint
    frame.attrs["data_ranges"] = ranges
    print("DOWNLOAD_OR_RUNTIME_ERRORS", json.dumps(errors, ensure_ascii=False, indent=2, default=str), flush=True)
    print("ROWS", len(frame), "SECONDS", round(time.time() - started, 1), flush=True)
    if frame.empty:
        raise SystemExit("No validation rows produced.")

    report = validation_report(frame, config, symbols)
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
