from __future__ import annotations

import ast
import json
import multiprocessing
from pathlib import Path

import pytest

from market_oracle import watchlist


def _upsert_worker(path_text: str, symbol: str, start, results) -> None:
    try:
        start.wait(10)
        watchlist.upsert_watch_item({"symbol": symbol, "horizon": 20}, path=Path(path_text))
        results.put(None)
    except BaseException as exc:  # pragma: no cover - asserted in parent process
        results.put(f"{type(exc).__name__}:{exc}")


def _archive_worker(path_text: str, item_id: str, start, results) -> None:
    try:
        start.wait(10)
        changed = watchlist.archive_watch_item(item_id, path=Path(path_text))
        results.put(None if changed else "archive did not find item")
    except BaseException as exc:  # pragma: no cover - asserted in parent process
        results.put(f"{type(exc).__name__}:{exc}")


def _duplicate_upsert_worker(path_text: str, start, results) -> None:
    try:
        start.wait(10)
        _, created = watchlist.upsert_watch_item({"symbol": "SPY", "horizon": 20}, path=Path(path_text))
        results.put(created)
    except BaseException as exc:  # pragma: no cover - asserted in parent process
        results.put(f"{type(exc).__name__}:{exc}")


def _run_concurrently(process_specs: list[tuple]) -> list[str | None]:
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    processes = [context.Process(target=target, args=(*args, start, results)) for target, args in process_specs]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0
    return [results.get(timeout=2) for _ in processes]


def _item(**overrides) -> dict:
    item = {
        "id": "watch-1",
        "symbol": "SPY",
        "horizon": 20,
        "created_at": "2026-09-25T10:00:00+00:00",
        "status": "ACTIVE",
    }
    item.update(overrides)
    return item


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _assert_integrity_error(path: Path, code: str = "WATCHLIST_CORRUPT") -> None:
    error_type = getattr(watchlist, "WatchlistIntegrityError", None)
    assert error_type is not None, "strict Watchlist boundary must expose WatchlistIntegrityError"
    with pytest.raises(error_type) as caught:
        watchlist.load_watchlist(path)
    assert caught.value.code == code


def test_missing_watchlist_is_legitimately_empty(tmp_path):
    assert watchlist.load_watchlist(tmp_path / "missing.json") == []


def test_loads_canonical_schema_one_and_legacy_raw_list(tmp_path):
    canonical = tmp_path / "canonical.json"
    legacy = tmp_path / "legacy.json"
    item = _item()
    _write(canonical, {"schema_version": 1, "items": [item]})
    _write(legacy, [item])

    assert watchlist.load_watchlist(canonical) == [item]
    assert watchlist.load_watchlist(legacy) == [item]


@pytest.mark.parametrize(
    "raw",
    [
        "{not-json",
        "null",
        '"watchlist"',
        "42",
        '{"items": []}',
        '{"schema_version": 2, "items": []}',
        '{"schema_version": 1, "items": {}}',
        '[{"id": "ok"}, 7]',
    ],
)
def test_existing_malformed_or_structurally_invalid_watchlist_fails_closed(tmp_path, raw):
    path = tmp_path / "watchlist.json"
    path.write_text(raw, encoding="utf-8")

    _assert_integrity_error(path)


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_schema_version_requires_exact_integer_one(tmp_path, schema_version):
    path = tmp_path / "watchlist.json"
    _write(path, {"schema_version": schema_version, "items": [_item()]})

    _assert_integrity_error(path)


@pytest.mark.parametrize(
    "item",
    [
        _item(id=""),
        _item(symbol=""),
        _item(horizon=True),
        _item(horizon=0),
        _item(horizon="20"),
        _item(created_at=""),
        _item(status=""),
    ],
)
def test_minimum_historical_storage_contract_is_all_or_nothing(tmp_path, item):
    path = tmp_path / "watchlist.json"
    _write(path, [_item(), item])

    _assert_integrity_error(path)


def test_duplicate_persisted_ids_fail_closed(tmp_path):
    path = tmp_path / "watchlist.json"
    _write(path, [_item(), _item(symbol="QQQ")])

    _assert_integrity_error(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_nonfinite_json_numbers_fail_closed_including_numeric_overflow(tmp_path, constant):
    path = tmp_path / "watchlist.json"
    raw = json.dumps({"schema_version": 1, "items": [_item(extra={"value": "TOKEN"})]})
    path.write_text(raw.replace('"TOKEN"', constant), encoding="utf-8")

    _assert_integrity_error(path)


def test_read_error_is_typed_and_never_becomes_empty(monkeypatch, tmp_path):
    path = tmp_path / "watchlist.json"
    _write(path, {"schema_version": 1, "items": [_item()]})
    original = Path.read_text

    def fail_read(self, *args, **kwargs):
        if self == path:
            raise OSError("disk unavailable")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read)
    _assert_integrity_error(path, "WATCHLIST_READ_FAILED")


def test_invalid_utf8_is_typed_corruption(tmp_path):
    path = tmp_path / "watchlist.json"
    path.write_bytes(b"\xff\xfe\x00")

    _assert_integrity_error(path, "WATCHLIST_CORRUPT")


def test_safe_load_distinguishes_empty_from_corrupt(tmp_path):
    safe_load = getattr(watchlist, "safe_load_watchlist", None)
    assert safe_load is not None, "UI boundary must expose safe_load_watchlist"
    missing = tmp_path / "missing.json"
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{broken", encoding="utf-8")

    assert safe_load(missing) == ([], None)
    assert safe_load(corrupt) == (None, "WATCHLIST_CORRUPT")


def test_custom_watchlist_paths_have_deterministic_isolated_lock_paths(tmp_path):
    lock_path = getattr(watchlist, "_watchlist_lock_path", None)
    assert lock_path is not None
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert lock_path(first) == tmp_path / "first.json.lock"
    assert lock_path(first) == lock_path(first)
    assert lock_path(first) != lock_path(second)


def test_public_save_is_canonical_atomic_and_rejects_nonfinite_without_mutation(tmp_path):
    path = tmp_path / "watchlist.json"
    original = {"schema_version": 1, "items": [_item()]}
    _write(path, original)

    watchlist.save_watchlist([_item(symbol="QQQ")], path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "items": [_item(symbol="QQQ")],
    }
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))
    canonical_after_success = path.read_bytes()

    with pytest.raises(watchlist.WatchlistIntegrityError) as caught:
        watchlist.save_watchlist([_item(extra={"bad": float("nan")})], path)
    assert caught.value.code == "WATCHLIST_CORRUPT"
    assert path.read_bytes() == canonical_after_success
    assert watchlist.load_watchlist(path) == [_item(symbol="QQQ")]


def test_failed_load_blocks_upsert_and_archive_without_touching_bytes(tmp_path):
    path = tmp_path / "watchlist.json"
    path.write_text("{broken", encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(watchlist.WatchlistIntegrityError):
        watchlist.upsert_watch_item({"symbol": "QQQ", "horizon": 20}, path=path)
    assert path.read_bytes() == before

    with pytest.raises(watchlist.WatchlistIntegrityError):
        watchlist.archive_watch_item("watch-1", path=path)
    assert path.read_bytes() == before


def test_read_failure_while_mutation_owns_lock_causes_zero_write(monkeypatch, tmp_path):
    path = tmp_path / "watchlist.json"
    _write(path, {"schema_version": 1, "items": [_item()]})
    before = path.read_bytes()
    original = Path.read_text

    def fail_read(self, *args, **kwargs):
        if self == path:
            raise OSError("read unavailable")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(watchlist.WatchlistIntegrityError) as caught:
        watchlist.upsert_watch_item({"symbol": "QQQ", "horizon": 20}, path=path)
    assert caught.value.code == "WATCHLIST_READ_FAILED"
    assert path.read_bytes() == before


def test_mutations_do_not_reacquire_public_save_lock(monkeypatch, tmp_path):
    path = tmp_path / "watchlist.json"

    def forbidden_public_save(*args, **kwargs):
        raise AssertionError("mutation must call unlocked writer while owning the RMW lock")

    monkeypatch.setattr(watchlist, "save_watchlist", forbidden_public_save)
    saved, created = watchlist.upsert_watch_item({"symbol": "SPY", "horizon": 20}, path=path)
    assert created is True
    assert watchlist.archive_watch_item(saved["id"], path=path) is True


def test_lock_acquisition_failure_is_typed_and_does_not_mutate(monkeypatch, tmp_path):
    path = tmp_path / "watchlist.json"
    _write(path, {"schema_version": 1, "items": [_item()]})
    before = path.read_bytes()

    def fail_lock(*args, **kwargs):
        raise OSError("lock unavailable")

    assert hasattr(watchlist, "fcntl")
    monkeypatch.setattr(watchlist.fcntl, "flock", fail_lock)
    with pytest.raises(watchlist.WatchlistIntegrityError) as caught:
        watchlist.upsert_watch_item({"symbol": "QQQ", "horizon": 20}, path=path)
    assert caught.value.code == "WATCHLIST_LOCK_FAILED"
    assert path.read_bytes() == before


def test_lock_file_open_failure_is_typed_and_does_not_mutate(monkeypatch, tmp_path):
    path = tmp_path / "watchlist.json"
    _write(path, {"schema_version": 1, "items": [_item()]})
    before = path.read_bytes()
    lock_path = watchlist._watchlist_lock_path(path)
    original = Path.open

    def fail_lock_open(self, *args, **kwargs):
        if self == lock_path:
            raise OSError("lock file unavailable")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_lock_open)
    with pytest.raises(watchlist.WatchlistIntegrityError) as caught:
        watchlist.upsert_watch_item({"symbol": "QQQ", "horizon": 20}, path=path)
    assert caught.value.code == "WATCHLIST_LOCK_FAILED"
    assert path.read_bytes() == before


def test_temp_write_failure_is_typed_and_preserves_canonical_file(monkeypatch, tmp_path):
    path = tmp_path / "watchlist.json"
    _write(path, {"schema_version": 1, "items": [_item()]})
    before = path.read_bytes()

    def fail_temp(*args, **kwargs):
        raise OSError("temp unavailable")

    assert hasattr(watchlist, "tempfile")
    monkeypatch.setattr(watchlist.tempfile, "NamedTemporaryFile", fail_temp)
    with pytest.raises(watchlist.WatchlistIntegrityError) as caught:
        watchlist.save_watchlist([_item(symbol="QQQ")], path)
    assert caught.value.code == "WATCHLIST_WRITE_FAILED"
    assert path.read_bytes() == before


def test_replace_failure_is_typed_preserves_canonical_and_cleans_temp(monkeypatch, tmp_path):
    path = tmp_path / "watchlist.json"
    _write(path, {"schema_version": 1, "items": [_item()]})
    before = path.read_bytes()

    def fail_replace(*args, **kwargs):
        raise OSError("replace unavailable")

    assert hasattr(watchlist, "os")
    monkeypatch.setattr(watchlist.os, "replace", fail_replace)
    with pytest.raises(watchlist.WatchlistIntegrityError) as caught:
        watchlist.save_watchlist([_item(symbol="QQQ")], path)
    assert caught.value.code == "WATCHLIST_WRITE_FAILED"
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_fsync_failure_is_typed_preserves_canonical_and_cleans_temp(monkeypatch, tmp_path):
    path = tmp_path / "watchlist.json"
    _write(path, {"schema_version": 1, "items": [_item()]})
    before = path.read_bytes()

    def fail_fsync(*args, **kwargs):
        raise OSError("fsync unavailable")

    monkeypatch.setattr(watchlist.os, "fsync", fail_fsync)
    with pytest.raises(watchlist.WatchlistIntegrityError) as caught:
        watchlist.save_watchlist([_item(symbol="QQQ")], path)
    assert caught.value.code == "WATCHLIST_WRITE_FAILED"
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_concurrent_upserts_do_not_lose_updates(tmp_path):
    path = tmp_path / "watchlist.json"
    symbols = [f"SYM{index}" for index in range(12)]
    errors = _run_concurrently([(_upsert_worker, (str(path), symbol)) for symbol in symbols])

    assert errors == [None] * len(symbols)
    assert {item["symbol"] for item in watchlist.load_watchlist(path)} == set(symbols)


def test_concurrent_duplicate_upserts_create_exactly_one_active_item(tmp_path):
    path = tmp_path / "watchlist.json"
    results = _run_concurrently(
        [
            (_duplicate_upsert_worker, (str(path),)),
            (_duplicate_upsert_worker, (str(path),)),
        ]
    )

    assert sorted(results) == [False, True]
    items = watchlist.load_watchlist(path)
    assert len(items) == 1
    assert items[0]["symbol"] == "SPY"
    assert items[0]["horizon"] == 20
    assert items[0]["status"] == "ACTIVE"


def test_concurrent_upsert_and_archive_do_not_lose_either_mutation(tmp_path):
    path = tmp_path / "watchlist.json"
    target, _ = watchlist.upsert_watch_item({"symbol": "SPY", "horizon": 20}, path=path)
    new_symbols = [f"NEW{index}" for index in range(8)]
    specs = [(_archive_worker, (str(path), target["id"]))]
    specs.extend((_upsert_worker, (str(path), symbol)) for symbol in new_symbols)

    errors = _run_concurrently(specs)
    items = watchlist.load_watchlist(path)

    assert errors == [None] * len(specs)
    by_symbol = {item["symbol"]: item for item in items}
    assert by_symbol["SPY"]["status"] == "ARCHIVED"
    assert set(new_symbols).issubset(by_symbol)


def test_concurrent_archives_do_not_lose_updates(tmp_path):
    path = tmp_path / "watchlist.json"
    first, _ = watchlist.upsert_watch_item({"symbol": "SPY", "horizon": 20}, path=path)
    second, _ = watchlist.upsert_watch_item({"symbol": "QQQ", "horizon": 20}, path=path)
    errors = _run_concurrently(
        [
            (_archive_worker, (str(path), first["id"])),
            (_archive_worker, (str(path), second["id"])),
        ]
    )

    assert errors == [None, None]
    assert {item["status"] for item in watchlist.load_watchlist(path)} == {"ARCHIVED"}


def _app_function(name: str) -> tuple[str, ast.FunctionDef]:
    source_path = Path(__file__).resolve().parents[1] / "app.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return source, function


def _direct_called_names(function: ast.FunctionDef) -> set[str]:
    return {
        call.func.id
        for call in ast.walk(function)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


def test_watchlist_ui_uses_safe_load_and_never_turns_corruption_into_empty_state():
    for name in ("render_watchlist_capture", "render_watchlist"):
        _, function = _app_function(name)
        called = _direct_called_names(function)
        assert "safe_load_watchlist" in called
        assert "watchlist_failure_message" in called
        assert "load_watchlist" not in called

        safe_assignment_index = next(
            index for index, node in enumerate(function.body)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "safe_load_watchlist"
                for call in ast.walk(node)
            )
        )
        failure_guard = function.body[safe_assignment_index + 1]
        assert isinstance(failure_guard, ast.If)
        guard_calls = {
            call.func.attr
            for call in ast.walk(failure_guard)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        }
        assert "error" in guard_calls
        assert any(isinstance(node, ast.Return) for node in ast.walk(failure_guard))


@pytest.mark.parametrize(
    ("function_name", "mutation_name"),
    [
        ("render_watchlist_capture", "upsert_watch_item"),
        ("render_watchlist", "archive_watch_item"),
    ],
)
def test_watchlist_ui_mutation_failure_has_no_success_or_rerun(function_name, mutation_name):
    _, function = _app_function(function_name)
    action_try = next(
        node for node in ast.walk(function)
        if isinstance(node, ast.Try)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == mutation_name
            for call in ast.walk(node)
        )
    )
    handler_calls = {
        call.func.attr
        for handler in action_try.handlers
        for call in ast.walk(handler)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }
    else_calls = {
        call.func.attr
        for node in action_try.orelse
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }

    assert "error" in handler_calls
    assert not {"success", "toast", "rerun"}.intersection(handler_calls)
    assert "rerun" in else_calls
    assert {"success", "toast"}.intersection(else_calls)


def test_watchlist_archive_false_result_is_not_presented_as_success():
    source, function = _app_function("render_watchlist")
    action_try = next(
        node for node in ast.walk(function)
        if isinstance(node, ast.Try)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "archive_watch_item"
            for call in ast.walk(node)
        )
    )

    assert len(action_try.orelse) == 1
    result_guard = action_try.orelse[0]
    assert isinstance(result_guard, ast.If)
    assert ast.unparse(result_guard.test) == "not archived"
    failure_source = "\n".join(ast.get_source_segment(source, node) or "" for node in result_guard.body)
    failure_calls = {
        call.func.attr
        for node in result_guard.body
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }
    success_calls = {
        call.func.attr
        for node in result_guard.orelse
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }
    assert "error" in failure_calls
    assert "WATCHLIST_UNEXPECTED" not in failure_source
    assert "Nie znaleziono tej obserwacji do archiwizacji" in failure_source
    assert not {"success", "rerun"}.intersection(failure_calls)
    assert {"success", "rerun"}.issubset(success_calls)


def test_watchlist_failure_message_is_controlled_and_does_not_claim_empty():
    source, function = _app_function("watchlist_failure_message")
    module = ast.Module(body=[function], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, filename="app.py", mode="exec"), namespace)

    message = namespace["watchlist_failure_message"]("WATCHLIST_UNEXPECTED")

    assert "WATCHLIST_UNEXPECTED" in message
    assert "pusta" not in message.lower()
    assert "/tmp/private" not in message
    assert "Dane nie zostały zmienione" not in message
    assert "Nie udało się potwierdzić stanu Watchlisty" in message


@pytest.mark.parametrize(
    "error_code",
    [
        "WATCHLIST_CORRUPT",
        "WATCHLIST_READ_FAILED",
        "WATCHLIST_WRITE_FAILED",
        "WATCHLIST_LOCK_FAILED",
    ],
)
def test_controlled_watchlist_failure_message_keeps_no_mutation_guarantee(error_code):
    _, function = _app_function("watchlist_failure_message")
    module = ast.Module(body=[function], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, filename="app.py", mode="exec"), namespace)

    message = namespace["watchlist_failure_message"](error_code)

    assert "Dane nie zostały zmienione" in message
