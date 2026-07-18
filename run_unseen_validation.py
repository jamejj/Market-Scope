from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from market_oracle.forward import (
    UNSEEN_UNIVERSE_PATH,
    assert_forward_contract_ready,
    current_commit,
    load_candidate_manifest,
    load_unseen_universe,
    pipeline_fingerprint,
    verify_frozen_hash,
)


ETF_SYMBOLS = {"TLT", "IWM", "DIA", "XLK", "XLF", "XLV", "XLE", "XLY"}


def symbol_market(symbol: str) -> str:
    return "ETF" if symbol.upper() in ETF_SYMBOLS else "USA"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pre-registered unseen USA/ETF Candidate v1 validation basket.")
    parser.add_argument("--universe", default=str(UNSEEN_UNIVERSE_PATH))
    parser.add_argument("--candidate", default="configs/marketscope_20d_long_candidate_v1.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--initial-train", type=int, default=420)
    parser.add_argument("--test-size", type=int, default=90)
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--holdout-size", type=int, default=0)
    parser.add_argument("--refit-every", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Only write the preflight manifest and print the command.")
    args = parser.parse_args()

    universe_path = Path(args.universe)
    candidate_path = Path(args.candidate)
    universe = load_unseen_universe(universe_path)
    manifest = load_candidate_manifest(candidate_path)
    if not verify_frozen_hash(universe, "universe_hash"):
        raise SystemExit(f"Universe hash mismatch: {universe_path}")
    assert_forward_contract_ready(manifest, require_clean_tree=True, enforce_pipeline=True)

    symbols = [str(symbol).upper() for symbol in universe["symbols"]]
    symbol_arg = ",".join(f"{symbol}:{symbol_market(symbol)}" for symbol in symbols)
    output_dir = Path(args.output_dir or f"outputs/unseen/{universe['universe_id']}")
    command = [
        sys.executable,
        "run_validation.py",
        "--symbols", symbol_arg,
        "--horizons", "20",
        "--years", str(args.years),
        "--initial-train", str(args.initial_train),
        "--test-size", str(args.test_size),
        "--max-folds", str(args.max_folds),
        "--holdout-size", str(args.holdout_size),
        "--refit-every", str(args.refit_every),
        "--cost-bps", str(universe["config"]["cost_bps"]),
        "--slippage-bps", str(universe["config"]["slippage_bps"]),
        "--output-dir", str(output_dir),
    ]
    preflight = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_commit": current_commit(),
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_hash": manifest["manifest_hash"],
        "universe_id": universe["universe_id"],
        "universe_hash": universe["universe_hash"],
        "symbols": symbols,
        "symbol_markets": {symbol: symbol_market(symbol) for symbol in symbols},
        "pipeline": pipeline_fingerprint(),
        "command": command,
        "dry_run": bool(args.dry_run),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = output_dir / f"unseen_preflight_{universe['universe_hash'][:12]}.json"
    preflight_path.write_text(json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"preflight": str(preflight_path), "command": command}, ensure_ascii=False, indent=2))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
