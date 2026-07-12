from __future__ import annotations

import json
import time
import fcntl
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .catalog import CATEGORIES, CRYPTO, ETF_CATEGORIES
from .engine import scan_market, scan_market_multi


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_PATH = DATA_DIR / "signals.json"
LOCK_PATH = DATA_DIR / "signals.lock"


def default_universe() -> list[str]:
    groups = [
        list(CATEGORIES["GPW — największe spółki"].values())[:14],
        list(CATEGORIES["GPW — średnie i mniejsze"].values())[:12],
        list(CATEGORIES["USA — technologia i półprzewodniki"].values())[:16],
        list(CATEGORIES["USA — banki i finanse"].values())[:6],
        list(CATEGORIES["USA — zdrowie i biotechnologia"].values())[:6],
        list(CATEGORIES["USA — przemysł i energia"].values())[:5],
        list(CATEGORIES["USA — handel, media i usługi"].values())[:6],
        list(CATEGORIES["USA — mniejsze i spekulacyjne"].values())[:8],
        list(CATEGORIES["Świat — spółki notowane w USA"].values())[:4],
        list(ETF_CATEGORIES["Szeroki rynek USA"].values())[:7],
        list(ETF_CATEGORIES["Sektory i technologia"].values())[:6],
        list(ETF_CATEGORIES["Świat i regiony"].values())[:4],
        list(ETF_CATEGORIES["Surowce"].values())[:4],
        list(ETF_CATEGORIES["Tematyczne i wzrostowe"].values())[:4],
        list(CRYPTO.values()),
    ]
    return list(dict.fromkeys(symbol for group in groups for symbol in group))


def _json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{key: _json_value(value) for key, value in row.items()} for row in frame.to_dict("records")]


def save_snapshot(payload: dict, path: Path = SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def run_signal_scan(
    symbols: list[str] | None = None,
    horizon: int = 20,
    horizons: tuple[int, ...] | None = (1, 5, 20),
    years: int = 8,
    path: Path = SNAPSHOT_PATH,
) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_PATH if path == SNAPSHOT_PATH else path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return load_snapshot(path) or {"status": "running", "completed": 0, "total": 0, "records": [], "errors": {}}

    universe = symbols or default_universe()
    started = datetime.now(timezone.utc)
    rows: list[dict] = []
    errors: dict[str, str] = {}
    payload = {
        "status": "running", "started_at": started.isoformat(), "updated_at": None,
        "horizon": horizon, "horizons": list(horizons or (horizon,)), "years": years, "completed": 0, "total": len(universe),
        "records": rows, "errors": errors,
    }
    save_snapshot(payload, path)

    try:
        for completed, symbol in enumerate(universe, start=1):
            if horizons:
                frame, failure = scan_market_multi([symbol], horizons=horizons, years=years)
            else:
                frame, failure = scan_market([symbol], horizon=horizon, years=years)
            if not frame.empty:
                rows.extend(_records(frame))
                rows.sort(key=lambda row: (row.get("Horyzont") or horizon, -(row.get("Score") or float("-inf"))))
            errors.update(failure)
            payload.update({"completed": completed, "records": rows, "errors": errors})
            save_snapshot(payload, path)

        payload.update({
            "status": "complete", "updated_at": datetime.now(timezone.utc).isoformat(),
            "completed": len(universe), "records": rows, "errors": errors,
        })
        save_snapshot(payload, path)
        return payload
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def snapshot_is_stale(snapshot: dict | None, max_age_hours: int = 12) -> bool:
    if not snapshot or snapshot.get("status") != "complete" or not snapshot.get("updated_at"):
        return True
    try:
        updated = datetime.fromisoformat(snapshot["updated_at"])
    except (TypeError, ValueError):
        return True
    return datetime.now(timezone.utc) - updated > timedelta(hours=max_age_hours)


def monitor_loop(interval_hours: int = 12, poll_seconds: int = 60) -> None:
    while True:
        try:
            snapshot = load_snapshot()
            if snapshot_is_stale(snapshot, interval_hours):
                run_signal_scan()
        except Exception as exc:
            snapshot = load_snapshot() or {}
            snapshot.update({"status": "error", "error": str(exc)})
            save_snapshot(snapshot)
        time.sleep(poll_seconds)
