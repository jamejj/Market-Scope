from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_oracle.candidate import run_candidate_forward_cycle
from market_oracle.forward import (
    CANDIDATE_MANIFEST_PATH,
    CANDIDATE_SNAPSHOT_PATH,
    FORWARD_LEDGER_PATH,
    FORWARD_UNIVERSE_PATH,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Candidate v1 forward scanner: refresh → full ML scan → ledger record.")
    parser.add_argument("--candidate", default=str(CANDIDATE_MANIFEST_PATH))
    parser.add_argument("--universe", default=str(FORWARD_UNIVERSE_PATH))
    parser.add_argument("--snapshot", default=str(CANDIDATE_SNAPSHOT_PATH))
    parser.add_argument("--ledger", default=str(FORWARD_LEDGER_PATH))
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--no-record", action="store_true", help="Save snapshot only; do not append to forward ledger.")
    parser.add_argument("--no-refresh-first", action="store_true", help="Skip pre-scan close/entry refresh. Not recommended.")
    parser.add_argument("--allow-before-close", action="store_true", help="Do not block same-day rows before the US close buffer.")
    args = parser.parse_args()

    snapshot, result = run_candidate_forward_cycle(
        manifest_path=Path(args.candidate),
        universe_path=Path(args.universe),
        snapshot_path=Path(args.snapshot),
        ledger_path=Path(args.ledger),
        years=args.years,
        record=not args.no_record,
        refresh_first=not args.no_refresh_first,
        require_closed_bar=not args.allow_before_close,
    )
    print(json.dumps({
        "snapshot": args.snapshot,
        "ledger": args.ledger,
        "status": snapshot.get("status"),
        "forward_universe": snapshot.get("forward_universe"),
        "records": len(snapshot.get("records") or []),
        "errors": snapshot.get("errors") or {},
        **result,
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
