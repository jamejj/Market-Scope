from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import argparse
import fcntl
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
from typing import Any, Callable

from .forward import DATA_DIR, ROOT, WARSAW, load_forward_cockpit


AUTOMATION_VERSION = 1
AUTOMATION_DIR = DATA_DIR / "forward_auto"
AUTOMATION_STATUS_PATH = AUTOMATION_DIR / "status.json"
AUTOMATION_LOCK_PATH = AUTOMATION_DIR / "candidate_v1.lock"
AUTOMATION_LOG_DIR = AUTOMATION_DIR / "logs"
LAUNCHD_LABEL = "com.jamejj.marketscope.candidate-forward"
LAUNCHD_PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
RUNNER_SCRIPT = ROOT / "run_forward_automation.py"
CANDIDATE_SCRIPT = ROOT / "run_candidate_forward.py"


def _default_python() -> Path:
    venv_python = ROOT / ".venv" / "bin" / "python"
    return venv_python if venv_python.exists() else Path(sys.executable)


@dataclass(frozen=True)
class AutomationConfig:
    """Operational schedule for Candidate v1 forward proof automation.

    The scanner itself stays unchanged. This config only decides *when* the
    already frozen proof flow may be called.
    """

    ready_hour: int = 22
    ready_minute: int = 35
    session_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    closed_dates: frozenset[str] = field(default_factory=frozenset)
    status_path: Path = AUTOMATION_STATUS_PATH
    lock_path: Path = AUTOMATION_LOCK_PATH
    log_dir: Path = AUTOMATION_LOG_DIR
    candidate_command: tuple[str, ...] = field(default_factory=lambda: (str(_default_python()), str(CANDIDATE_SCRIPT)))


class AutomationLocked(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return {}, None
    except json.JSONDecodeError as exc:
        return {}, f"Uszkodzony status automatu: {path}:{exc.lineno}"
    except OSError as exc:
        return {}, str(exc)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_safe_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_automation_status(
    *,
    config: AutomationConfig | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    config = config or AutomationConfig()
    stored, error = _read_json(config.status_path)
    cockpit = load_forward_cockpit()
    plan = build_automation_plan(cockpit=cockpit, now=now, config=config, stored_status=stored)
    return {
        "schema_version": AUTOMATION_VERSION,
        "status_file": str(config.status_path),
        "status_file_exists": config.status_path.exists(),
        "status_error": error,
        "stored": stored,
        "plan": plan,
        "launchd": launchd_status(),
        "commands": {
            "install": f"{_shell_path(_default_python())} {_shell_path(RUNNER_SCRIPT)} install",
            "uninstall": f"{_shell_path(_default_python())} {_shell_path(RUNNER_SCRIPT)} uninstall",
            "status": f"{_shell_path(_default_python())} {_shell_path(RUNNER_SCRIPT)} status",
            "run_now": f"{_shell_path(_default_python())} {_shell_path(RUNNER_SCRIPT)} run-now",
            "dry_run": f"{_shell_path(_default_python())} {_shell_path(RUNNER_SCRIPT)} dry-run",
        },
    }


def _shell_path(path: Path) -> str:
    text = str(path)
    if " " in text:
        return f'"{text}"'
    return text


def _as_warsaw(now: datetime | str | None) -> datetime:
    if now is None:
        return datetime.now(WARSAW)
    if isinstance(now, str):
        value = datetime.fromisoformat(now.replace("Z", "+00:00"))
    else:
        value = now
    if value.tzinfo is None:
        return value.replace(tzinfo=WARSAW)
    return value.astimezone(WARSAW)


def _date_text(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.astimezone(WARSAW).date().isoformat() if value.tzinfo else value.date().isoformat()
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        try:
            return date.fromisoformat(str(value)[:10]).isoformat()
        except ValueError:
            return None
    return parsed.astimezone(WARSAW).date().isoformat() if parsed.tzinfo else parsed.date().isoformat()


def _is_session_day(day: date, config: AutomationConfig) -> bool:
    return day.weekday() in set(config.session_weekdays) and day.isoformat() not in config.closed_dates


def _previous_session_day(day: date, config: AutomationConfig) -> date | None:
    candidate = day
    for _ in range(14):
        if _is_session_day(candidate, config):
            return candidate
        candidate -= timedelta(days=1)
    return None


def _next_session_day(day: date, config: AutomationConfig) -> date:
    candidate = day
    for _ in range(21):
        if _is_session_day(candidate, config):
            return candidate
        candidate += timedelta(days=1)
    raise RuntimeError("Nie udało się wyznaczyć kolejnej sesji.")


def _ready_time(day: date, config: AutomationConfig) -> datetime:
    return datetime.combine(day, time(config.ready_hour, config.ready_minute), tzinfo=WARSAW)


def target_session_date(now: datetime | str | None = None, config: AutomationConfig | None = None) -> date | None:
    config = config or AutomationConfig()
    local = _as_warsaw(now)
    today_ready = _ready_time(local.date(), config)
    latest_closed_candidate = local.date() if local >= today_ready else local.date() - timedelta(days=1)
    return _previous_session_day(latest_closed_candidate, config)


def next_planned_run(now: datetime | str | None = None, config: AutomationConfig | None = None) -> datetime:
    config = config or AutomationConfig()
    local = _as_warsaw(now)
    if _is_session_day(local.date(), config) and local < _ready_time(local.date(), config):
        return _ready_time(local.date(), config)
    return _ready_time(_next_session_day(local.date() + timedelta(days=1), config), config)


def build_automation_plan(
    *,
    cockpit: dict[str, Any] | None = None,
    now: datetime | str | None = None,
    config: AutomationConfig | None = None,
    stored_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or AutomationConfig()
    local = _as_warsaw(now)
    target = target_session_date(local, config)
    latest_audit = _date_text((cockpit or {}).get("latest_audit_date"))
    target_text = target.isoformat() if target else None
    stored_status = stored_status or {}
    last_status = stored_status.get("automation_status")
    should_run = False
    reason = "NO_ELIGIBLE_SESSION"
    warning = None

    if target is None:
        reason = "NO_RECENT_SESSION"
    elif latest_audit and latest_audit >= target_text:
        reason = "ALREADY_AUDITED"
    elif target == local.date():
        should_run = True
        reason = "SCHEDULED_SESSION_READY"
    else:
        should_run = True
        reason = "CATCH_UP_MISSED_SESSION"
        warning = f"Brakuje audytu dla ostatniej zamkniętej sesji: {target_text}."

    if last_status == "FAILED" and stored_status.get("target_session_date") == target_text:
        warning = f"Ostatni automat dla {target_text} zakończył się błędem."

    planned = next_planned_run(local, config)
    return {
        "now_local": local.isoformat(),
        "target_session_date": target_text,
        "latest_audit_date": latest_audit,
        "should_run": should_run,
        "reason": reason,
        "next_planned_run_local": planned.isoformat(),
        "missed_session_warning": warning,
        "ready_after_local": f"{config.ready_hour:02d}:{config.ready_minute:02d}",
    }


@contextmanager
def automation_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AutomationLocked(f"Forward automation is already running: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _log_paths(config: AutomationConfig, started_at: datetime) -> tuple[Path, Path]:
    stamp = started_at.astimezone(WARSAW).strftime("%Y%m%d_%H%M%S")
    config.log_dir.mkdir(parents=True, exist_ok=True)
    return (
        config.log_dir / f"{stamp}.stdout.log",
        config.log_dir / f"{stamp}.stderr.log",
    )


def _parse_runner_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.lstrip()
    if not text:
        return {"payload": None, "summary_text": ""}
    try:
        decoder = json.JSONDecoder()
        payload, end = decoder.raw_decode(text)
        return {"payload": payload, "summary_text": text[end:].strip()}
    except json.JSONDecodeError:
        return {"payload": None, "summary_text": text.strip()}


def execute_automation(
    *,
    config: AutomationConfig | None = None,
    now: datetime | str | None = None,
    dry_run: bool = False,
    runner: Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    config = config or AutomationConfig()
    stored, status_error = _read_json(config.status_path)
    cockpit = load_forward_cockpit()
    plan = build_automation_plan(cockpit=cockpit, now=now, config=config, stored_status=stored)
    base = {
        "schema_version": AUTOMATION_VERSION,
        "automation_version": AUTOMATION_VERSION,
        "plan": plan,
        "status_error": status_error,
        "candidate_command": list(config.candidate_command),
    }
    if dry_run:
        return {**base, "automation_status": "DRY_RUN", "message": "Nie uruchomiono skanera."}

    started_at = datetime.now(timezone.utc)
    stdout_path, stderr_path = _log_paths(config, started_at)
    if not plan["should_run"]:
        ended_at = datetime.now(timezone.utc)
        payload = {
            **base,
            "automation_status": "SKIPPED",
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": (ended_at - started_at).total_seconds(),
            "exit_code": 0,
            "stdout_log": None,
            "stderr_log": None,
        }
        _write_json(config.status_path, payload)
        return payload

    try:
        with automation_lock(config.lock_path):
            try:
                run = (runner or _default_runner)(config.candidate_command)
            except Exception as exc:
                run = subprocess.CompletedProcess(config.candidate_command, 1, "", str(exc))
    except AutomationLocked as exc:
        ended_at = datetime.now(timezone.utc)
        payload = {
            **base,
            "automation_status": "LOCKED",
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": (ended_at - started_at).total_seconds(),
            "exit_code": 75,
            "error": str(exc),
            "stdout_log": None,
            "stderr_log": None,
        }
        _write_json(config.status_path, payload)
        return payload

    stdout_path.write_text(run.stdout or "", encoding="utf-8")
    stderr_path.write_text(run.stderr or "", encoding="utf-8")
    parsed = _parse_runner_stdout(run.stdout or "")
    ended_at = datetime.now(timezone.utc)
    fresh_cockpit = load_forward_cockpit()
    fresh_plan = build_automation_plan(cockpit=fresh_cockpit, now=now, config=config, stored_status=stored)
    payload = {
        **base,
        "plan_after_run": fresh_plan,
        "automation_status": "OK" if run.returncode == 0 else "FAILED",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": (ended_at - started_at).total_seconds(),
        "target_session_date": plan.get("target_session_date"),
        "exit_code": int(run.returncode),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_tail": (run.stdout or "")[-4000:],
        "stderr_tail": (run.stderr or "")[-4000:],
        "runner_payload": parsed["payload"],
        "runner_summary_text": parsed["summary_text"],
    }
    _write_json(config.status_path, payload)
    return payload


def _default_runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def launchd_plist_payload(
    *,
    config: AutomationConfig | None = None,
    python_path: Path | None = None,
    plist_path: Path = LAUNCHD_PLIST_PATH,
) -> dict[str, Any]:
    config = config or AutomationConfig()
    python_path = python_path or _default_python()
    stdout = config.log_dir / "launchd.stdout.log"
    stderr = config.log_dir / "launchd.stderr.log"
    intervals = [
        {"Weekday": weekday, "Hour": config.ready_hour, "Minute": config.ready_minute}
        for weekday in range(1, 6)
    ]
    return {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [str(python_path), str(RUNNER_SCRIPT), "run-now"],
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "StartCalendarInterval": intervals,
        "StandardOutPath": str(stdout),
        "StandardErrorPath": str(stderr),
        "EnvironmentVariables": {
            "PATH": f"{ROOT / '.venv' / 'bin'}:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin",
            "PYTHONUNBUFFERED": "1",
        },
    }


def write_launchd_plist(path: Path = LAUNCHD_PLIST_PATH, *, config: AutomationConfig | None = None) -> Path:
    config = config or AutomationConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(launchd_plist_payload(config=config, plist_path=path), handle, sort_keys=False)
    return path


def install_launchd(path: Path = LAUNCHD_PLIST_PATH, *, config: AutomationConfig | None = None) -> dict[str, Any]:
    path = write_launchd_plist(path, config=config)
    unload = subprocess.run(["launchctl", "unload", str(path)], capture_output=True, text=True, check=False)
    load = subprocess.run(["launchctl", "load", "-w", str(path)], capture_output=True, text=True, check=False)
    return {
        "installed": load.returncode == 0,
        "plist": str(path),
        "unload_exit_code": unload.returncode,
        "load_exit_code": load.returncode,
        "load_stderr": load.stderr,
    }


def uninstall_launchd(path: Path = LAUNCHD_PLIST_PATH) -> dict[str, Any]:
    unload = subprocess.run(["launchctl", "unload", str(path)], capture_output=True, text=True, check=False)
    removed = False
    if path.exists():
        path.unlink()
        removed = True
    return {
        "installed": False,
        "plist": str(path),
        "removed": removed,
        "unload_exit_code": unload.returncode,
        "unload_stderr": unload.stderr,
    }


def launchd_status(path: Path = LAUNCHD_PLIST_PATH) -> dict[str, Any]:
    loaded = None
    error = None
    try:
        result = subprocess.run(["launchctl", "list", LAUNCHD_LABEL], capture_output=True, text=True, check=False)
        loaded = result.returncode == 0
        error = result.stderr.strip() or None
    except Exception as exc:
        error = str(exc)
    return {
        "label": LAUNCHD_LABEL,
        "plist": str(path),
        "plist_exists": path.exists(),
        "loaded": loaded,
        "error": error,
    }


def format_automation_summary(payload: dict[str, Any]) -> str:
    stored = payload.get("stored") or payload
    plan = payload.get("plan") or stored.get("plan") or {}
    status = stored.get("automation_status") or payload.get("automation_status") or "UNKNOWN"
    lines = [
        "",
        f"Candidate v1 automation: {status}",
        f"Target session: {stored.get('target_session_date') or plan.get('target_session_date') or '—'}",
        f"Plan reason: {plan.get('reason') or '—'}",
        f"Next planned run: {(stored.get('plan_after_run') or {}).get('next_planned_run_local') or plan.get('next_planned_run_local') or '—'}",
    ]
    if stored.get("exit_code") is not None:
        lines.append(f"Exit code: {stored.get('exit_code')}")
    runner = stored.get("runner_payload") or {}
    counts = runner.get("run_event_counts") or {}
    if counts:
        lines.append("Run events: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    if stored.get("runner_summary_text"):
        lines.append("")
        lines.append(str(stored["runner_summary_text"]).strip())
    warning = plan.get("missed_session_warning")
    if warning:
        lines.append(f"Warning: {warning}")
    if payload.get("status_error"):
        lines.append(f"Status file warning: {payload['status_error']}")
    return "\n".join(lines)


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safe macOS automation wrapper for MarketScope Candidate v1 forward proof flow.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-run", help="Show whether Candidate v1 would run now; do not mutate status or ledger.")
    sub.add_parser("run-now", help="Run only when the post-close guard and dedupe checks allow it.")
    sub.add_parser("status", help="Show automation status and latest plan.")
    sub.add_parser("install", help="Install launchd LaunchAgent for weekday post-close runs.")
    sub.add_parser("uninstall", help="Unload and remove the launchd LaunchAgent.")
    args = parser.parse_args(argv)

    if args.command == "dry-run":
        payload = execute_automation(dry_run=True)
        print(json.dumps(_safe_json(payload), ensure_ascii=False, indent=2))
        print(format_automation_summary(payload))
        return 0
    if args.command == "run-now":
        payload = execute_automation()
        print(json.dumps(_safe_json(payload), ensure_ascii=False, indent=2))
        print(format_automation_summary(payload))
        if payload.get("automation_status") in {"OK", "SKIPPED"}:
            return 0
        return int(payload.get("exit_code") or 1)
    if args.command == "status":
        payload = load_automation_status()
        print(json.dumps(_safe_json(payload), ensure_ascii=False, indent=2))
        print(format_automation_summary(payload))
        return 0
    if args.command == "install":
        payload = install_launchd()
        print(json.dumps(_safe_json(payload), ensure_ascii=False, indent=2))
        return 0 if payload.get("installed") else 1
    if args.command == "uninstall":
        payload = uninstall_launchd()
        print(json.dumps(_safe_json(payload), ensure_ascii=False, indent=2))
        return 0
    return 2
