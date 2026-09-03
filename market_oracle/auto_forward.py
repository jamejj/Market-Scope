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
EXPLICIT_NYSE_HOLIDAYS = frozenset({
    # Official NYSE/ICE holiday calendar for 2026-2028.
    # Early closes are not full-session closures and are intentionally excluded.
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19",
    "2028-07-04", "2028-09-04", "2028-11-23", "2028-12-25",
})


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
    holiday_calendar: str = "NYSE"
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


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    return current + timedelta(days=7 * (nth - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    next_month = date(year + int(month == 12), 1 if month == 12 else month + 1, 1)
    current = next_month - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _observed_fixed(year: int, month: int, day: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _easter_sunday(year: int) -> date:
    # Meeus/Jones/Butcher Gregorian algorithm.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def nyse_full_holidays(year: int) -> set[str]:
    holidays = {item for item in EXPLICIT_NYSE_HOLIDAYS if item.startswith(f"{year}-")}
    if holidays:
        return set(holidays)
    return {
        _observed_fixed(year, 1, 1).isoformat(),
        _nth_weekday(year, 1, 0, 3).isoformat(),   # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3).isoformat(),   # Washington's Birthday
        (_easter_sunday(year) - timedelta(days=2)).isoformat(),  # Good Friday
        _last_weekday(year, 5, 0).isoformat(),     # Memorial Day
        _observed_fixed(year, 6, 19).isoformat(),  # Juneteenth
        _observed_fixed(year, 7, 4).isoformat(),   # Independence Day
        _nth_weekday(year, 9, 0, 1).isoformat(),   # Labor Day
        _nth_weekday(year, 11, 3, 4).isoformat(),  # Thanksgiving
        _observed_fixed(year, 12, 25).isoformat(),
    }


def _is_session_day(day: date, config: AutomationConfig) -> bool:
    if day.weekday() not in set(config.session_weekdays):
        return False
    closed = set(config.closed_dates)
    if config.holiday_calendar.upper() == "NYSE":
        closed.update(nyse_full_holidays(day.year))
    return day.isoformat() not in closed


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


def eligible_session_dates(
    *,
    latest_audit_date: str | None,
    now: datetime | str | None = None,
    config: AutomationConfig | None = None,
    max_lookback_days: int = 45,
) -> list[str]:
    config = config or AutomationConfig()
    target = target_session_date(now, config)
    if target is None:
        return []
    if latest_audit_date:
        parsed_latest = date.fromisoformat(latest_audit_date)
        start = parsed_latest + timedelta(days=1)
    else:
        start = target
    earliest = target - timedelta(days=max_lookback_days)
    current = max(start, earliest)
    sessions: list[str] = []
    while current <= target:
        if _is_session_day(current, config):
            sessions.append(current.isoformat())
        current += timedelta(days=1)
    return sessions


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
    missing_sessions = eligible_session_dates(latest_audit_date=latest_audit, now=local, config=config)
    skipped_backfill = missing_sessions[:-1] if len(missing_sessions) > 1 else []
    stored_status = stored_status or {}
    last_status = stored_status.get("automation_status")
    should_run = False
    reason = "NO_ELIGIBLE_SESSION"
    warning = None

    if target is None:
        reason = "NO_RECENT_SESSION"
    elif not missing_sessions:
        reason = "ALREADY_AUDITED"
    elif latest_audit and latest_audit >= target_text:
        reason = "ALREADY_AUDITED"
    elif len(missing_sessions) > 1:
        should_run = True
        if target == local.date():
            reason = "SCHEDULED_SESSION_READY_WITH_GAP"
        else:
            reason = "CATCH_UP_LATEST_ONLY"
        warning = (
            f"Brakuje audytów dla {len(missing_sessions)} sesji: {', '.join(missing_sessions)}. "
            f"Wrapper wykona tylko najnowszą sesję {target_text}; wcześniejsze zostają jawnie oznaczone jako missed."
        )
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
        "missing_sessions": missing_sessions,
        "missed_sessions": skipped_backfill,
        "missed_sessions_count": len(skipped_backfill),
        "target_session_is_nyse_session": bool(target and _is_session_day(target, config)),
        "holiday_calendar": config.holiday_calendar,
        "should_run": should_run,
        "reason": reason,
        "next_planned_run_local": planned.isoformat(),
        "missed_session_warning": warning,
        "ready_after_local": f"{config.ready_hour:02d}:{config.ready_minute:02d}",
    }


def _last_successful_run(stored_status: dict[str, Any]) -> dict[str, Any] | None:
    current = stored_status.get("last_successful_run")
    if isinstance(current, dict):
        return current
    if stored_status.get("automation_status") == "OK":
        return {
            "target_session_date": stored_status.get("target_session_date"),
            "ended_at": stored_status.get("ended_at"),
            "exit_code": stored_status.get("exit_code"),
            "runner_summary_text": stored_status.get("runner_summary_text"),
        }
    return None


def _success_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_session_date": payload.get("target_session_date"),
        "ended_at": payload.get("ended_at"),
        "exit_code": payload.get("exit_code"),
        "runner_summary_text": payload.get("runner_summary_text"),
        "runner_payload_status": (payload.get("runner_payload") or {}).get("status"),
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


def _command_for_target_session(command: tuple[str, ...], target_session: str) -> tuple[str, ...]:
    """Bind the executed runner command to the post-lock automation target."""

    cleaned: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token == "--target-session-date":
            index += 1
            if index < len(command) and not command[index].startswith("--"):
                index += 1
            continue
        if token.startswith("--target-session-date="):
            index += 1
            continue
        cleaned.append(token)
        index += 1
    return (*cleaned, "--target-session-date", target_session)


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
        "last_successful_run": _last_successful_run(stored),
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
            "last_successful_run": _last_successful_run(stored),
        }
        _write_json(config.status_path, payload)
        return payload

    try:
        with automation_lock(config.lock_path):
            locked_stored, locked_status_error = _read_json(config.status_path)
            locked_cockpit = load_forward_cockpit()
            locked_plan = build_automation_plan(
                cockpit=locked_cockpit,
                now=now,
                config=config,
                stored_status=locked_stored,
            )
            if locked_status_error and not status_error:
                base["status_error"] = locked_status_error
            if not locked_plan["should_run"]:
                ended_at = datetime.now(timezone.utc)
                payload = {
                    **base,
                    "plan_after_lock": locked_plan,
                    "automation_status": "SKIPPED",
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": (ended_at - started_at).total_seconds(),
                    "exit_code": 0,
                    "stdout_log": None,
                    "stderr_log": None,
                    "race_recheck": True,
                    "last_successful_run": _last_successful_run(locked_stored),
                }
                _write_json(config.status_path, payload)
                return payload
            locked_target = str(locked_plan["target_session_date"])
            executed_command = _command_for_target_session(config.candidate_command, locked_target)
            try:
                run = (runner or _default_runner)(executed_command)
            except Exception as exc:
                run = subprocess.CompletedProcess(executed_command, 1, "", str(exc))
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
            "last_successful_run": _last_successful_run(stored),
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
        "candidate_command": list(executed_command),
        "plan_after_run": fresh_plan,
        "automation_status": "OK" if run.returncode == 0 else "FAILED",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": (ended_at - started_at).total_seconds(),
        "target_session_date": locked_target,
        "exit_code": int(run.returncode),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_tail": (run.stdout or "")[-4000:],
        "stderr_tail": (run.stderr or "")[-4000:],
        "runner_payload": parsed["payload"],
        "runner_summary_text": parsed["summary_text"],
    }
    runner_payload = parsed["payload"] if isinstance(parsed["payload"], dict) else {}
    if run.returncode != 0 and runner_payload.get("failure_kind"):
        payload["failure_kind"] = str(runner_payload["failure_kind"])
    payload["last_successful_run"] = _success_summary(payload) if run.returncode == 0 else _last_successful_run(stored)
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


def _launchd_domain() -> str:
    return f"gui/{os.getuid()}"


def _launchd_service() -> str:
    return f"{_launchd_domain()}/{LAUNCHD_LABEL}"


def _log_tail(path: Path, max_chars: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8")[-max_chars:]
    except FileNotFoundError:
        return ""
    except OSError as exc:
        return str(exc)


def _launchd_print_field(output: str, field: str) -> str | None:
    prefix = f"\t{field} = "
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def write_launchd_plist(path: Path = LAUNCHD_PLIST_PATH, *, config: AutomationConfig | None = None) -> Path:
    config = config or AutomationConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(launchd_plist_payload(config=config, plist_path=path), handle, sort_keys=False)
    return path


def install_launchd(path: Path = LAUNCHD_PLIST_PATH, *, config: AutomationConfig | None = None) -> dict[str, Any]:
    path = write_launchd_plist(path, config=config)
    domain = _launchd_domain()
    bootout = subprocess.run(["launchctl", "bootout", domain, str(path)], capture_output=True, text=True, check=False)
    bootstrap = subprocess.run(["launchctl", "bootstrap", domain, str(path)], capture_output=True, text=True, check=False)
    enable = subprocess.run(["launchctl", "enable", _launchd_service()], capture_output=True, text=True, check=False)
    status = launchd_status(path)
    return {
        "installed": bool(status.get("loaded")),
        "plist": str(path),
        "domain": domain,
        "bootout_exit_code": bootout.returncode,
        "bootstrap_exit_code": bootstrap.returncode,
        "enable_exit_code": enable.returncode,
        "bootstrap_stderr": bootstrap.stderr,
        "loaded": status.get("loaded"),
        "last_exit_code": status.get("last_exit_code"),
        "privacy_block_detected": status.get("privacy_block_detected"),
    }


def uninstall_launchd(path: Path = LAUNCHD_PLIST_PATH) -> dict[str, Any]:
    domain = _launchd_domain()
    bootout = subprocess.run(["launchctl", "bootout", domain, str(path)], capture_output=True, text=True, check=False)
    removed = False
    if path.exists():
        path.unlink()
        removed = True
    return {
        "installed": False,
        "plist": str(path),
        "domain": domain,
        "removed": removed,
        "bootout_exit_code": bootout.returncode,
        "bootout_stderr": bootout.stderr,
    }


def launchd_status(path: Path = LAUNCHD_PLIST_PATH) -> dict[str, Any]:
    domain = _launchd_domain()
    service = _launchd_service()
    print_stdout = ""
    print_stderr = ""
    try:
        result = subprocess.run(["launchctl", "print", service], capture_output=True, text=True, check=False)
        loaded = result.returncode == 0
        print_stdout = result.stdout or ""
        print_stderr = result.stderr or ""
    except Exception as exc:
        loaded = None
        print_stderr = str(exc)
    stdout_log = AUTOMATION_LOG_DIR / "launchd.stdout.log"
    stderr_log = AUTOMATION_LOG_DIR / "launchd.stderr.log"
    stderr_tail = _log_tail(stderr_log)
    stdout_tail = _log_tail(stdout_log)
    privacy_block = (
        "Operation not permitted" in stderr_tail
        and ("pyvenv.cfg" in stderr_tail or "/Documents/" in stderr_tail)
    )
    return {
        "label": LAUNCHD_LABEL,
        "domain": domain,
        "service": service,
        "plist": str(path),
        "plist_exists": path.exists(),
        "loaded": loaded,
        "state": _launchd_print_field(print_stdout, "state"),
        "runs": _launchd_print_field(print_stdout, "runs"),
        "last_exit_code": _launchd_print_field(print_stdout, "last exit code"),
        "error": print_stderr.strip() or None,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "privacy_block_detected": privacy_block,
        "privacy_hint": (
            "macOS privacy blocked the background LaunchAgent from reading files under Documents. "
            "Grant Full Disk Access to the launcher/Python path or move the repo outside Documents before relying on automation."
            if privacy_block else None
        ),
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
