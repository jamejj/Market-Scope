from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_oracle.forward import (
    CANDIDATE_MANIFEST_PATH,
    CANDIDATE_SNAPSHOT_PATH,
    FORWARD_LEDGER_PATH,
    FORWARD_UNIVERSE_PATH,
    record_snapshot_forward_signals,
    refresh_forward_ledger,
    forward_summary,
    load_candidate_manifest,
    load_forward_events,
    reconstruct_forward_state,
    verify_frozen_hash,
)
from market_oracle.candidate import load_candidate_snapshot
from market_oracle.integrity import (
    INTEGRITY_EXIT_CODE,
    SnapshotIntegrityError,
    validate_candidate_snapshot_session,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Append-only forward test ledger for MarketScope Candidate v1.")
    parser.add_argument("--snapshot", default=str(CANDIDATE_SNAPSHOT_PATH), help="Completed Candidate v1 snapshot.")
    parser.add_argument("--ledger", default=str(FORWARD_LEDGER_PATH), help="Append-only JSONL ledger path.")
    parser.add_argument("--candidate", default=str(CANDIDATE_MANIFEST_PATH), help="Frozen candidate manifest.")
    parser.add_argument("--universe", default=str(FORWARD_UNIVERSE_PATH), help="Frozen forward universe manifest.")
    parser.add_argument("--years", type=int, default=3, help="History years used only to fill entry/exit prices.")
    parser.add_argument("--target-session-date", help="Closed market session bound to canonical proof writes (YYYY-MM-DD).")
    parser.add_argument("--record-snapshot", action="store_true", help="Append new SIGNAL_OBSERVED rows from snapshot.")
    parser.add_argument("--refresh", action="store_true", help="Append ENTRY_FILLED/POSITION_CLOSED events when prices exist.")
    args = parser.parse_args()

    if not args.record_snapshot and not args.refresh:
        args.record_snapshot = True
        args.refresh = True

    candidate_path = Path(args.candidate)
    ledger_path = Path(args.ledger)
    universe_path = Path(args.universe)
    manifest = load_candidate_manifest(candidate_path)
    if not verify_frozen_hash(manifest, "manifest_hash"):
        raise SystemExit(f"Manifest hash mismatch: {candidate_path}")

    try:
        snapshot = None
        if args.record_snapshot:
            snapshot = load_candidate_snapshot(Path(args.snapshot))
            if snapshot is None:
                raise SystemExit(f"Nie znalazłem poprawnego snapshotu: {args.snapshot}")
            validate_candidate_snapshot_session(
                snapshot,
                target_session_date=args.target_session_date,
                require_target=ledger_path.resolve() == FORWARD_LEDGER_PATH.resolve(),
            )

        errors = {}
        if args.refresh:
            events, state, errors = refresh_forward_ledger(
                path=ledger_path,
                manifest_path=candidate_path,
                years=args.years,
                target_session_date=args.target_session_date,
            )
        else:
            events = load_forward_events(ledger_path)
            state = {}

        added_signals = 0
        if args.record_snapshot:
            added_signals = record_snapshot_forward_signals(
                snapshot,
                path=ledger_path,
                manifest_path=candidate_path,
                universe_path=universe_path,
                target_session_date=args.target_session_date,
            )
            events = load_forward_events(ledger_path)
            state = reconstruct_forward_state(events)
    except SnapshotIntegrityError as exc:
        print(json.dumps({
            "status": "error",
            "failure_kind": exc.failure_kind,
            "exit_code": exc.exit_code,
            "integrity_errors": exc.errors,
        }, ensure_ascii=False))
        return INTEGRITY_EXIT_CODE

    print(json.dumps({
        "candidate_id": manifest["candidate_id"],
        "ledger": str(ledger_path),
        "added_signals": added_signals,
        "summary": forward_summary(events),
        "state_items": len(state),
        "errors": errors,
    }, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
