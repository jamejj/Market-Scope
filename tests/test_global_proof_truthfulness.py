from __future__ import annotations

import ast
import html
import json
from pathlib import Path

import pytest

from market_oracle import auto_forward
from market_oracle.presentation import build_start_guidance


ROOT = Path(__file__).resolve().parents[1]


def _app_proof_state():
    source_path = ROOT / "app.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    wanted = {"short_datetime", "proof_state"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"html": html}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["proof_state"]


def _healthy_cockpit() -> dict:
    return {"healthy": True, "problems": []}


def _healthy_automation(**overrides) -> dict:
    payload = {
        "status_error": None,
        "stored": {"automation_status": "OK"},
        "plan": {"next_planned_run_local": "2026-10-05T22:35:00+02:00"},
        "launchd": {"loaded": True, "last_exit_code": 0},
    }
    payload.update(overrides)
    return payload


def _config(status_path: Path) -> auto_forward.AutomationConfig:
    return auto_forward.AutomationConfig(
        status_path=status_path,
        lock_path=status_path.with_suffix(".lock"),
        log_dir=status_path.parent / "logs",
    )


def test_read_json_missing_file_is_legitimate_empty_status(tmp_path):
    payload, error = auto_forward._read_json(tmp_path / "missing.json")

    assert payload == {}
    assert error is None


def test_read_json_valid_empty_object_is_legitimate_status(tmp_path):
    path = tmp_path / "status.json"
    path.write_text("{}", encoding="utf-8")

    payload, error = auto_forward._read_json(path)

    assert payload == {}
    assert error is None


def test_read_json_valid_stored_status_is_preserved(tmp_path):
    path = tmp_path / "status.json"
    stored = {"automation_status": "OK", "target_session_date": "2026-10-02"}
    path.write_text(json.dumps(stored), encoding="utf-8")

    payload, error = auto_forward._read_json(path)

    assert payload == stored
    assert error is None


def test_read_json_malformed_status_fails_closed(tmp_path):
    path = tmp_path / "status.json"
    path.write_text("{not-json", encoding="utf-8")

    payload, error = auto_forward._read_json(path)

    assert payload == {}
    assert error


def test_read_json_oserror_fails_closed(monkeypatch, tmp_path):
    path = tmp_path / "status.json"

    def fail_read_text(self, *args, **kwargs):
        raise OSError("status storage unavailable")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    payload, error = auto_forward._read_json(path)

    assert payload == {}
    assert error == "status storage unavailable"


def test_read_json_invalid_utf8_fails_closed(tmp_path):
    path = tmp_path / "status.json"
    path.write_bytes(b"\xff\xfe\xfa")

    payload, error = auto_forward._read_json(path)

    assert payload == {}
    assert error


@pytest.mark.parametrize(
    "root",
    [None, [], ["row"], "", "text", 0, 7, False, True],
    ids=[
        "null",
        "empty-list",
        "nonempty-list",
        "empty-string",
        "nonempty-string",
        "zero",
        "number",
        "false",
        "true",
    ],
)
def test_read_json_non_object_root_fails_closed(tmp_path, root):
    path = tmp_path / "status.json"
    path.write_text(json.dumps(root), encoding="utf-8")

    payload, error = auto_forward._read_json(path)

    assert payload == {}
    assert error


def test_load_status_keeps_plan_and_launchd_diagnostics_for_corrupt_file(monkeypatch, tmp_path):
    path = tmp_path / "status.json"
    path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(auto_forward, "load_forward_cockpit", lambda: {"healthy": True})
    monkeypatch.setattr(
        auto_forward,
        "build_automation_plan",
        lambda **kwargs: {"stored_seen": kwargs["stored_status"]},
    )
    monkeypatch.setattr(auto_forward, "launchd_status", lambda: {"loaded": True})

    status = auto_forward.load_automation_status(config=_config(path))

    assert status["status_error"]
    assert status["stored"] == {}
    assert status["plan"] == {"stored_seen": {}}
    assert status["launchd"] == {"loaded": True}


def test_invalid_utf8_status_remains_controlled_and_cannot_produce_global_ok(monkeypatch, tmp_path):
    path = tmp_path / "status.json"
    path.write_bytes(b"\xff\xfe\xfa")
    monkeypatch.setattr(auto_forward, "load_forward_cockpit", lambda: _healthy_cockpit())
    monkeypatch.setattr(
        auto_forward,
        "build_automation_plan",
        lambda **kwargs: {
            "stored_seen": kwargs["stored_status"],
            "next_planned_run_local": "2026-10-05T22:35:00+02:00",
        },
    )
    monkeypatch.setattr(
        auto_forward,
        "launchd_status",
        lambda: {"loaded": True, "last_exit_code": 0},
    )

    automation = auto_forward.load_automation_status(config=_config(path))
    state = _app_proof_state()(_healthy_cockpit(), automation, None, None)

    assert automation["stored"] == {}
    assert automation["status_error"]
    assert automation["plan"]["stored_seen"] == {}
    assert automation["launchd"]["loaded"] is True
    assert state["label"] == "Wymaga uwagi"
    assert state["klass"] == "bad"


def test_load_status_missing_file_remains_legal(monkeypatch, tmp_path):
    path = tmp_path / "missing.json"
    monkeypatch.setattr(auto_forward, "load_forward_cockpit", lambda: {"healthy": True})
    monkeypatch.setattr(auto_forward, "build_automation_plan", lambda **kwargs: {})
    monkeypatch.setattr(auto_forward, "launchd_status", lambda: {"loaded": True})

    status = auto_forward.load_automation_status(config=_config(path))

    assert status["status_file_exists"] is False
    assert status["status_error"] is None
    assert status["stored"] == {}


def test_proof_state_healthy_happy_path_remains_ok():
    state = _app_proof_state()(_healthy_cockpit(), _healthy_automation(), None, None)

    assert state["label"] == "OK"
    assert state["klass"] == ""


def test_proof_state_canonical_failed_status_requires_attention():
    automation = _healthy_automation(stored={"automation_status": "FAILED"})

    state = _app_proof_state()(_healthy_cockpit(), automation, None, None)

    assert state["label"] == "Wymaga uwagi"
    assert state["klass"] == "bad"


def test_corrupt_status_shared_proof_state_cannot_report_ok_to_topbar_or_start():
    automation = _healthy_automation(status_error="Uszkodzony status automatu: status.json:1")

    state = _app_proof_state()(_healthy_cockpit(), automation, None, None)

    assert state == {
        "label": "Wymaga uwagi",
        "klass": "bad",
        "detail": "Status automatyzacji jest uszkodzony lub niedostępny.",
    }
    assert state["label"] != "OK"


def test_corrupt_status_creates_start_guidance_proof_attention_card():
    automation = _healthy_automation(status_error="permission denied")
    state = _app_proof_state()(_healthy_cockpit(), automation, None, None)

    guidance = build_start_guidance(
        snapshot={},
        cockpit=_healthy_cockpit(),
        automation=automation,
        proof_state=state,
    )

    assert guidance["cards"][0]["id"] == "proof_attention"
    assert guidance["cards"][0]["status"] == "Wymaga uwagi"
    assert "uszkodzony lub niedostępny" in guidance["cards"][0]["body"]


def test_proof_state_healthy_ledger_without_loaded_launchd_is_ledger_ok():
    automation = _healthy_automation(launchd={"loaded": False, "last_exit_code": None})

    state = _app_proof_state()(_healthy_cockpit(), automation, None, None)

    assert state["label"] == "Ledger OK"
    assert state["klass"] == "warn"


def test_proof_state_cockpit_problem_requires_attention():
    cockpit = {"healthy": False, "problems": ["Brak audytu snapshotu"]}

    state = _app_proof_state()(cockpit, _healthy_automation(), None, None)

    assert state["label"] == "Wymaga uwagi"
    assert state["klass"] == "bad"
    assert state["detail"] == "Brak audytu snapshotu"
