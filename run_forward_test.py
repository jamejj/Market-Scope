from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_oracle.forward import (
    CANDIDATE_MANIFEST_PATH,
    FORWARD_LEDGER_PATH,
    record_snapshot_forward_signals,
    refresh_forward_ledger,
    forward_summary,
    load_candidate_manifest,
    load_forward_events,
    verify_frozen_hash,
)
from market_oracle.monitor import SNAPSHOT_PATH, load_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Append-only forward test ledger for MarketScope Candidate v1.")
    parser.add_argument("--snapshot", default=str(SNAPSHOT_PATH), help="Completed monitor snapshot, default data/signals.json.")
    parser.add_argument("--ledger", default=str(FORWARD_LEDGER_PATH), help="Append-only JSONL ledger path.")
    parser.add_argument("--candidate", default=str(CANDIDATE_MANIFEST_PATH), help="Frozen candidate manifest.")
    parser.add_argument("--years", type=int, default=3, help="History years used only to fill entry/exit prices.")
    parser.add_argument("--record-snapshot", action="store_true", help="Append new SIGNAL_OBSERVED rows from snapshot.")
    parser.add_argument("--refresh", action="store_true", help="Append ENTRY_FILLED/POSITION_CLOSED events when prices exist.")
    args = parser.parse_args()

    if not args.record_snapshot and not args.refresh:
        args.record_snapshot = True
        args.refresh = True

    candidate_path = Path(args.candidate)
    ledger_path = Path(args.ledger)
    manifest = load_candidate_manifest(candidate_path)
    if not verify_frozen_hash(manifest, "manifest_hash"):
        raise SystemExit(f"Manifest hash mismatch: {candidate_path}")

    added_signals = 0
    if args.record_snapshot:
        snapshot = load_snapshot(Path(args.snapshot))
        if snapshot is None:
            raise SystemExit(f"Nie znalazłem poprawnego snapshotu: {args.snapshot}")
        added_signals = record_snapshot_forward_signals(
            snapshot,
            path=ledger_path,
            manifest_path=candidate_path,
        )

    errors = {}
    if args.refresh:
        events, state, errors = refresh_forward_ledger(
            path=ledger_path,
            manifest_path=candidate_path,
            years=args.years,
        )
    else:
        events = load_forward_events(ledger_path)
        state = {}

    print(json.dumps({
        "candidate_id": manifest["candidate_id"],
        "ledger": str(ledger_path),
        "added_signals": added_signals,
        "summary": forward_summary(events),
        "state_items": len(state),
        "errors": errors,
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
