from __future__ import annotations

import math
from typing import Any

from .product_verdict import has_complete_expected_return, product_forecast_verdict
from .signals import SignalVerdict


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pct(value: Any) -> str:
    number = _finite_float(value)
    return "—" if number is None else f"{number:.1%}"


def _signed_pct(value: Any) -> str:
    number = _finite_float(value)
    return "—" if number is None else f"{number:+.1%}"


def _decimal(value: Any) -> str:
    number = _finite_float(value)
    return "—" if number is None else f"{number:.3f}"


def _horizon_label(horizon: int, crypto: bool) -> str:
    unit = "dni" if crypto else "sesji"
    labels = {
        1: "1 dzień" if crypto else "1 sesja",
        5: "5 dni" if crypto else "5 sesji",
        20: f"20 {unit}",
        60: f"60 {unit}",
    }
    return labels.get(int(horizon), f"{horizon} {unit}")


def _forecast_items(forecasts: dict[Any, dict]) -> list[tuple[int, dict]]:
    items: list[tuple[int, dict]] = []
    for horizon, forecast in forecasts.items():
        try:
            items.append((int(horizon), forecast))
        except (TypeError, ValueError):
            continue
    return sorted(items, key=lambda item: item[0])


def _forecast_verdict(forecast: dict) -> SignalVerdict:
    return product_forecast_verdict(forecast, source="FULL_ANALYSIS")


def _quality_rank(quality: str) -> int:
    if quality == "WYSOKA":
        return 3
    if quality == "UMIARKOWANA":
        return 2
    if quality and not quality.startswith("NISKA"):
        return 1
    return 0


def _verdict_rank(verdict: SignalVerdict) -> int:
    if verdict.decision != 0:
        return 3
    if verdict.reason == "EXPECTED_RETURN_CONFLICT":
        return 1
    if verdict.reason == "LOW_QUALITY":
        return 0
    return 2


def _primary_forecast(forecasts: dict[Any, dict]) -> tuple[int, dict]:
    items = _forecast_items(forecasts)
    if not items:
        raise ValueError("Brak prognoz do zbudowania raportu analizy.")
    preferred = {20: 4, 5: 3, 60: 2, 1: 1}

    def probability_distance(forecast: dict) -> float:
        probability = _finite_float(forecast.get("probability_up"))
        return abs((0.5 if probability is None else probability) - 0.5)

    return max(
        items,
        key=lambda item: (
            int(has_complete_expected_return(item[1])),
            _verdict_rank(_forecast_verdict(item[1])),
            _quality_rank(str(item[1].get("quality") or "")),
            probability_distance(item[1]),
            abs(_finite_float(item[1].get("expected_return")) or 0.0),
            preferred.get(item[0], 0),
        ),
    )


def _report_forecast(forecasts: dict[Any, dict], selected_horizon: int | None) -> tuple[int, dict]:
    if selected_horizon is None:
        return _primary_forecast(forecasts)
    requested = int(selected_horizon)
    for horizon, forecast in _forecast_items(forecasts):
        if horizon == requested:
            return horizon, forecast
    available = ", ".join(str(horizon) for horizon, _ in _forecast_items(forecasts)) or "brak"
    raise ValueError(f"Horyzont {requested} nie jest dostępny w tej analizie. Dostępne: {available}.")


def _reason_label(reason: str) -> str:
    labels = {
        "LONG_CONFIRMED": "wzrost potwierdzony przez wspólną bramkę",
        "SHORT_CONFIRMED": "spadek potwierdzony przez wspólną bramkę",
        "LOW_QUALITY": "niska jakość walidacji",
        "EXPECTED_RETURN_CONFLICT": "konflikt P(wzrost) z oczekiwanym ruchem",
        "EXPECTED_RETURN_TOO_SMALL": "oczekiwany ruch za mały",
        "PROBABILITY_INSIDE_BAND": "prawdopodobieństwo wewnątrz pasma obserwacji",
        "INCOMPLETE_FORECAST": "brak wiarygodnej wartości oczekiwanego ruchu",
    }
    return labels.get(reason, reason.replace("_", " ").lower())


def _display_radar_action(value: Any) -> str:
    labels = {
        "RYZYKO / UNIKAJ": "Podwyższone ryzyko",
    }
    text = str(value or "—")
    return labels.get(text, text)


def _direction(verdict: SignalVerdict) -> tuple[str, str]:
    reason = _reason_label(verdict.reason)
    if verdict.decision == 1:
        return "Scenariusz wzrostowy spełnia warunki MarketScope", f"Werdykt techniczny: LONG — {reason}."
    if verdict.decision == -1:
        return "Scenariusz spadkowy spełnia warunki MarketScope", f"Werdykt techniczny: SHORT — {reason}."
    if verdict.reason == "LOW_QUALITY":
        return "Brak potwierdzonej przewagi", "Model został celowo ściągnięty w stronę 50%, bo walidacja nie pokazała stabilnej przewagi."
    if verdict.reason == "INCOMPLETE_FORECAST":
        return "Obserwuj", "Brak wiarygodnej wartości oczekiwanego ruchu."
    return "Obserwuj", f"Reguły MarketScope nie potwierdzają kierunku: {reason}."


def _trend_label(technical: dict) -> str:
    return_20d = _finite_float(technical.get("return_20d"))
    rsi_14 = _finite_float(technical.get("rsi_14"))
    points = sum([
        bool(return_20d is not None and return_20d > 0),
        bool(rsi_14 is not None and rsi_14 >= 50),
        bool(technical.get("above_sma_50")),
        bool(technical.get("above_sma_200")),
    ])
    if points >= 3:
        return "trend techniczny wspiera tezę"
    if points <= 1:
        return "trend techniczny jest słaby"
    return "trend techniczny jest mieszany"


def _freshness_value(value: Any) -> str:
    if value is None or value == "":
        return "—"
    text = str(value).replace("T", " ")
    return text[:16]


def _int_value(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(number, 0)


def _scan_stage_text(snapshot: dict, universe_size: int = 0) -> tuple[str, str]:
    """Human label for the two-step radar without changing scan logic."""
    universe_total = _int_value(snapshot.get("universe_total") or universe_size)
    fast_completed = _int_value(snapshot.get("fast_completed"))
    if not fast_completed and snapshot.get("status") == "complete" and universe_total:
        fast_completed = universe_total
    completed = _int_value(snapshot.get("completed"))
    total = _int_value(snapshot.get("total"))
    ml_completed = _int_value(snapshot.get("ml_completed"))
    ml_total = _int_value(snapshot.get("ml_total"))
    fast = f"FAST {fast_completed}/{universe_total}" if universe_total else "FAST —"
    if str(snapshot.get("status") or "") == "running":
        if universe_total and fast_completed >= universe_total:
            headline = f"{fast} · Deep ML w toku"
        else:
            headline = f"{fast} · skan w toku"
    elif str(snapshot.get("status") or "") == "complete":
        headline = f"{fast} · radar gotowy"
    else:
        headline = f"{fast} · status {snapshot.get('status') or 'offline'}"

    if ml_total:
        detail = f"Deep ML: {ml_completed}/{ml_total} kandydatów"
    elif total:
        detail = f"Pełny workflow: {completed}/{total} kroków"
    else:
        detail = "Dwustopniowy radar FAST → Deep ML"
    return headline, detail


def _radar_freshness(source_context: dict) -> tuple[str, str | None]:
    updated = _freshness_value(source_context.get("radar_updated_at"))
    if updated != "—":
        return updated, None
    if source_context.get("radar_status") == "running":
        started = _freshness_value(source_context.get("radar_started_at"))
        if started != "—":
            return f"skan w toku od {started}", "Pełna analiza została uruchomiona w trakcie odświeżania radaru; wartości mogą różnić się od ostatniego kompletnego snapshotu."
        return "skan w toku", "Pełna analiza została uruchomiona w trakcie odświeżania radaru; timestamp startu nie jest dostępny."
    return "snapshot niedostępny", "Raport uruchomiono poza zapisanym kompletnym snapshotem radaru albo snapshot nie ma timestampu."


def build_analysis_report(
    result: dict,
    profile: dict | None = None,
    source_context: dict | None = None,
    selected_horizon: int | None = None,
) -> dict:
    """Human-readable report layer for full instrument analysis.

    This function deliberately does not change models, thresholds or signal decisions.
    It only translates the already-computed analysis dictionary into product copy.
    """
    profile = profile or {}
    source_context = source_context or {}
    symbol = str(result.get("symbol") or "—")
    crypto = symbol.upper().endswith("-USD")
    forecasts = result.get("forecasts") or {}
    horizon, forecast = _report_forecast(forecasts, selected_horizon)
    selection_mode = "AUTO" if selected_horizon is None else "MANUAL"
    quality = str(forecast.get("quality") or "—")
    probability = _finite_float(forecast.get("probability_up"))
    verdict = _forecast_verdict(forecast)
    direction, direction_detail = _direction(verdict)
    technical = result.get("technical") or {}
    risk = result.get("risk") or {}
    trend = _trend_label(technical)
    horizon_text = _horizon_label(horizon, crypto)
    expected = _signed_pct(forecast.get("expected_return"))
    lower = _signed_pct(forecast.get("lower_return"))
    upper = _signed_pct(forecast.get("upper_return"))
    auc = _finite_float(forecast.get("auc"))
    brier = _finite_float(forecast.get("brier"))

    if verdict.decision == 1:
        headline = f"{symbol}: scenariusz wzrostowy spełnia warunki MarketScope na horyzoncie {horizon_text}."
    elif verdict.decision == -1:
        headline = f"{symbol}: scenariusz spadkowy spełnia warunki MarketScope na horyzoncie {horizon_text}."
    elif verdict.reason == "LOW_QUALITY":
        headline = f"{symbol}: brak potwierdzonej przewagi modelu — obserwuj, nie zakładaj edge."
    elif verdict.reason == "INCOMPLETE_FORECAST":
        headline = f"{symbol}: obserwuj — brak wiarygodnej wartości oczekiwanego ruchu na horyzoncie {horizon_text}."
    else:
        headline = f"{symbol}: obserwuj — reguły MarketScope nie potwierdzają kierunku na horyzoncie {horizon_text}."

    rsi = _finite_float(technical.get("rsi_14"))
    rsi_text = "—" if rsi is None else f"{rsi:.1f}"

    evidence = [
        f"Horyzont roboczy raportu: {horizon_text}; jakość walidacji: {quality}.",
        f"P(wzrost) { _pct(probability) }, oczekiwany ruch {expected}, zakres 90%: {lower} – {upper}.",
        f"Werdykt reguł MarketScope: {verdict.label} — {_reason_label(verdict.reason)}.",
        f"Technicznie: {trend}; zwrot 20 sesji/dni { _signed_pct(technical.get('return_20d')) }, RSI 14: {rsi_text}.",
    ]
    if auc is not None and brier is not None:
        evidence.append(f"Walidacja: AUC {auc:.3f}, Brier {brier:.3f}; to mówi o jakości kierunku i kalibracji, nie o gwarancji zysku.")

    counterpoints: list[str] = []
    if verdict.decision == 0 and not quality.startswith("NISKA"):
        counterpoints.append(f"Reguły MarketScope zwracają OBSERWUJ ({_reason_label(verdict.reason)}), więc raport nie wskazuje potwierdzonego kierunku mimo pojedynczych mocnych metryk.")
    if quality.startswith("NISKA"):
        counterpoints.append("Model sam oznacza jakość jako niską — brak przewagi jest ważniejszy niż pojedynczy ładny ruch ceny.")
    if _finite_float(forecast.get("lower_return")) is not None and float(forecast.get("lower_return")) < 0:
        counterpoints.append(f"Dolny zakres 90% nadal zakłada możliwy spadek ({lower}), więc ryzyko scenariusza przeciwnego jest realne.")
    if _finite_float(forecast.get("expected_return")) is not None and float(forecast.get("expected_return")) <= 0:
        counterpoints.append("Oczekiwany ruch nie jest dodatni — kierunek modelu nie wystarcza bez potencjału zwrotu.")
    if technical.get("above_sma_50") is False or technical.get("above_sma_200") is False:
        counterpoints.append("Cena nie jest jednocześnie nad kluczowymi średnimi 50/200, więc trend nie jest w pełni potwierdzony.")
    drawdown = _finite_float(risk.get("max_drawdown"))
    if drawdown is not None:
        counterpoints.append(f"Historyczny max drawdown wynosi {_signed_pct(drawdown)} — to przypomina, jak głębokie obsunięcia występowały wcześniej.")
    if not counterpoints:
        counterpoints.append("Brak dużego czerwonego alertu w danych raportu, ale to nadal analiza probabilistyczna, nie pewność ruchu.")

    horizon_detail = (
        "Ten horyzont ma teraz najwyższy priorytet raportu; pozostałe są poniżej."
        if selection_mode == "AUTO"
        else "Horyzont wybrany ręcznie; werdykt policzono dla istniejącej prognozy tego horyzontu."
    )
    cards = [
        ("Co system widzi?", direction, direction_detail),
        ("Wsparcie w walidacji", quality, f"AUC {auc:.3f} · Brier {brier:.3f}" if auc is not None and brier is not None else "Brak pełnych metryk walidacji"),
        ("Horyzont", horizon_text, horizon_detail),
        ("Oczekiwany ruch", expected, f"Zakres 90%: {lower} – {upper}"),
        ("Trend", trend, f"1d {_signed_pct(technical.get('return_1d'))} · 5d {_signed_pct(technical.get('return_5d'))} · 20d {_signed_pct(technical.get('return_20d'))}"),
        ("Max drawdown", _signed_pct(drawdown), "Historyczne obsunięcie; niżej w raporcie są pełne metryki ryzyka."),
    ]
    horizon_cards = []
    for h, f in _forecast_items(forecasts):
        horizon_verdict = _forecast_verdict(f)
        horizon_cards.append({
            "horizon": h,
            "label": _horizon_label(h, crypto),
            "verdict": horizon_verdict.label,
            "probability": _pct(f.get("probability_up")),
            "expected": _signed_pct(f.get("expected_return")),
            "lower": _signed_pct(f.get("lower_return")),
            "upper": _signed_pct(f.get("upper_return")),
            "quality": str(f.get("quality") or "—"),
            "auc": _decimal(f.get("auc")),
            "brier": _decimal(f.get("brier")),
        })

    full_data_as_of = result.get("last_date")
    if hasattr(full_data_as_of, "date"):
        full_data_as_of = str(full_data_as_of.date())
    radar_freshness, radar_note = _radar_freshness(source_context)
    freshness = {
        "radar": radar_freshness,
        "analysis": _freshness_value(full_data_as_of),
        "benchmark": str(result.get("benchmark") or "—"),
    }
    if radar_note:
        freshness["note"] = radar_note
    elif freshness["radar"] != "snapshot niedostępny" and freshness["analysis"] != "—" and not freshness["radar"].startswith(freshness["analysis"]):
        freshness["note"] = "Radar i pełna analiza mogą mieć minimalnie różne wartości, bo pełna analiza liczy aktualnie dostępne dane."
    else:
        freshness["note"] = "Radar i raport są spójne datowo albo raport został uruchomiony ręcznie poza snapshotem radaru."

    return {
        "symbol": symbol,
        "primary_horizon": horizon,
        "selection_mode": selection_mode,
        "requested_horizon": None if selected_horizon is None else int(selected_horizon),
        "verdict": {
            "label": verdict.label,
            "reason": verdict.reason,
            "decision": verdict.decision,
        },
        "headline": headline,
        "body": (
            f"Raport łączy istniejące prognozy ML, trend, momentum, ryzyko i świeżość danych. "
            f"To nie jest rekomendacja kupna ani sprzedaży — to szybka interpretacja dowodów dla {symbol}."
        ),
        "cards": cards,
        "evidence": evidence,
        "counterpoints": counterpoints,
        "horizon_cards": horizon_cards,
        "freshness": freshness,
    }


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _row_text(row: dict, key: str, default: str = "—") -> str:
    value = row.get(key)
    if value is None or value == "":
        return default
    return str(value)


def _row_number(row: dict, key: str) -> float | None:
    return _finite_float(row.get(key))


def _row_decimal(row: dict, key: str, digits: int = 2) -> str:
    number = _row_number(row, key)
    if number is None:
        return _row_text(row, key)
    return f"{number:,.{digits}f}"


def _horizon_short(row: dict) -> str:
    horizon = _row_number(row, "Horyzont")
    if horizon is None:
        return "—"
    return f"{int(horizon)}d"


def _row_signal_verdict(row: dict) -> SignalVerdict:
    return product_forecast_verdict(
        {
            "probability_up": row.get("P(wzrost)"),
            "expected_return": row.get("Oczekiwany ruch"),
            "quality": _row_text(row, "Jakość modelu", "NISKA — BRAK PRZEWAGI"),
            "auc": row.get("AUC walidacji"),
            "brier": row.get("Brier"),
        },
        source="START_GUIDANCE",
    )


def _row_score(row: dict) -> tuple:
    score_keys = ["Deep score", "Setup score", "Radar score", "Edge score", "Score"]
    scores = tuple(_row_number(row, key) or 0.0 for key in score_keys)
    probability = _row_number(row, "P(wzrost)")
    expected = _row_number(row, "Oczekiwany ruch")
    return (*scores, abs((0.5 if probability is None else probability) - 0.5), abs(expected or 0.0))


def _best_row(rows: list[dict]) -> dict | None:
    return max(rows, key=_row_score) if rows else None


def _risk_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in records:
        action = _row_text(row, "Akcja radaru", "").upper()
        grade = _row_text(row, "Setup grade", "").upper()
        thesis = _row_text(row, "Teza radaru", "").upper()
        if "RYZYKO" in action or "UNIKAJ" in action or grade.startswith("R") or "WYSOKIE RYZYKO" in thesis:
            rows.append(row)
    return rows


def _confirmed_ml_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in records:
        if _row_text(row, "Tryb analizy") != "ML":
            continue
        if _row_signal_verdict(row).decision != 0:
            rows.append(row)
    return rows


def _fast_rows(records: list[dict]) -> list[dict]:
    allowed_actions = {"FAST SHORTLIST", "MOMENTUM DO SPRAWDZENIA", "WATCHLIST", "PRIORYTET DO ANALIZY"}
    rows: list[dict] = []
    for row in records:
        action = _row_text(row, "Akcja radaru")
        if _row_text(row, "Tryb analizy") == "FAST" and action in allowed_actions:
            rows.append(row)
    return rows


def _card(
    *,
    card_id: str,
    priority: int,
    title: str,
    body: str,
    source: str,
    status: str,
    cta: str,
    action: str,
    symbol: str | None = None,
    tone: str = "info",
    meta: dict | None = None,
) -> dict:
    return {
        "id": card_id,
        "priority": priority,
        "title": title,
        "body": body,
        "source": source,
        "status": status,
        "cta": cta,
        "action": action,
        "symbol": symbol,
        "tone": tone,
        "meta": meta or {},
    }


def build_start_guidance(
    *,
    snapshot: dict | None,
    cockpit: dict | None,
    automation: dict | None,
    proof_state: dict | None,
    journal: dict | None = None,
    universe_size: int = 0,
    radar_stale: bool = False,
    max_cards: int = 5,
) -> dict:
    """Build a short home-screen action plan from existing MarketScope state.

    This is a presentation layer only: it does not change models, thresholds,
    Candidate v1, the forward ledger or scan artifacts. Cards tell the user what
    to inspect inside MarketScope, not what to buy or sell.
    """
    snapshot = snapshot or {}
    cockpit = cockpit or {}
    automation = automation or {}
    proof_state = proof_state or {}
    records = [row for row in _as_list(snapshot.get("records")) if isinstance(row, dict)]
    status = str(snapshot.get("status") or "offline")
    updated = _freshness_value(snapshot.get("updated_at"))
    started = _freshness_value(snapshot.get("started_at"))
    radar_freshness = updated if updated != "—" else (f"skan w toku od {started}" if status == "running" and started != "—" else "brak kompletnego snapshotu")
    warning = None
    cards: list[dict] = []
    used_symbols: set[str] = set()

    def add(card: dict) -> None:
        symbol = card.get("symbol")
        if symbol and symbol in used_symbols:
            return
        if symbol:
            used_symbols.add(symbol)
        cards.append(card)

    proof_label = str(proof_state.get("label") or "—")
    proof_detail = str(proof_state.get("detail") or "")
    if proof_state.get("klass") in {"bad", "warn"}:
        add(_card(
            card_id="proof_attention",
            priority=100,
            title="Sprawdź proof flow — system wymaga uwagi.",
            body=proof_detail or "Forward ledger albo automatyzacja zgłaszają problem diagnostyczny.",
            source="Proof / Forward",
            status=proof_label,
            cta="Pokaż szczegóły proof",
            action="show_forward_details",
            tone="danger" if proof_state.get("klass") == "bad" else "warn",
        ))

    if status == "running":
        stage_headline, stage_detail = _scan_stage_text(snapshot, universe_size)
        warning = "Radar jest w trakcie odświeżania — guidance może mieszać gotowe wiersze z częściowym skanem."
        add(_card(
            card_id="radar_running",
            priority=95,
            title="Radar jest właśnie aktualizowany — traktuj wyniki jako częściowe.",
            body=f"{stage_headline}. {stage_detail}. Skan rozpoczął się {started}; poczekaj na kompletne ML enrichment, jeśli chcesz pełny obraz.",
            source="Radar",
            status=stage_headline,
            cta="Pokaż bieżący snapshot",
            action="show_radar_snapshot",
            tone="warn",
        ))
    elif radar_stale:
        warning = "Ostatni radar jest stary albo niepełny — karty służą tylko jako orientacyjna lista pracy."
        add(_card(
            card_id="radar_stale",
            priority=92,
            title="Odśwież radar przed głębszą interpretacją rynku.",
            body=f"Ostatni kompletny snapshot: {updated}. MarketScope może już mieć świeższe ceny niż zapisany ranking.",
            source="Radar",
            status="snapshot wymaga odświeżenia",
            cta="Pokaż status radaru",
            action="show_radar_snapshot",
            tone="warn",
        ))
    elif status not in {"complete", "running"}:
        warning = "Brakuje kompletnego snapshotu radaru — guidance ogranicza się do forward proof i statusu operacyjnego."

    risk_leader = _best_row(_risk_rows(records))
    if risk_leader:
        symbol = _row_text(risk_leader, "Symbol")
        add(_card(
            card_id="risk_alert",
            priority=90,
            title=f"Przejrzyj ryzyko: {symbol} ma alert radaru.",
            body=f"{_row_text(risk_leader, 'Teza radaru')}. Horyzont {_horizon_short(risk_leader)}, ruch/impet {_signed_pct(risk_leader.get('Oczekiwany ruch'))}.",
            source="Radar FAST/ML",
            status=_display_radar_action(_row_text(risk_leader, "Akcja radaru")),
            cta=f"Uruchom pełną analizę: {symbol}",
            action="full_analysis",
            symbol=symbol,
            tone="danger",
            meta={"horizon": _horizon_short(risk_leader)},
        ))

    ml_leader = _best_row(_confirmed_ml_rows(records))
    if ml_leader:
        symbol = _row_text(ml_leader, "Symbol")
        verdict = _row_signal_verdict(ml_leader)
        add(_card(
            card_id="ml_candidate",
            priority=80,
            title=f"Przeanalizuj {symbol}: ML ma potwierdzony setup {_horizon_short(ml_leader)}.",
            body=(
                f"Reguły MarketScope: {verdict.label}; P(wzrost) {_pct(ml_leader.get('P(wzrost)'))}, "
                f"oczekiwany ruch {_signed_pct(ml_leader.get('Oczekiwany ruch'))}. To kandydat do analizy, nie polecenie transakcji."
            ),
            source="Deep ML",
            status=_row_text(ml_leader, "Jakość modelu"),
            cta=f"Uruchom pełną analizę: {symbol}",
            action="full_analysis",
            symbol=symbol,
            tone="success",
            meta={"horizon": _horizon_short(ml_leader)},
        ))

    open_positions = _as_list(cockpit.get("open_positions"))
    if open_positions:
        position = open_positions[0]
        symbol = _row_text(position, "Symbol")
        add(_card(
            card_id="forward_position",
            priority=70,
            title=f"Monitoruj aktywną hipotezę forward: {symbol}.",
            body=(
                f"Pozycja testowa jest otwarta od {_row_text(position, 'Data wejścia')} po {_row_decimal(position, 'Cena wejścia')}. "
                f"Do planowego rozliczenia zostało około {_row_text(position, 'Sesje do wyjścia')} sesji."
            ),
            source="Forward proof",
            status=f"portfel {((cockpit.get('portfolio') or {}).get('open', 0))}/{((cockpit.get('portfolio') or {}).get('slots', 5))}",
            cta="Pokaż szczegóły Forward",
            action="show_forward_details",
            symbol=symbol,
            tone="info",
        ))

    fast_leader = _best_row(_fast_rows(records))
    if fast_leader:
        symbol = _row_text(fast_leader, "Symbol")
        add(_card(
            card_id="fast_setup",
            priority=60,
            title=f"Zobacz, dlaczego {symbol} trafił na shortlistę FAST.",
            body=(
                f"{_row_text(fast_leader, 'Akcja radaru')} na horyzoncie {_horizon_short(fast_leader)}. "
                f"Teza: {_row_text(fast_leader, 'Teza radaru')}. FAST pomaga ustawić kolejność pracy, ale nie jest potwierdzeniem ML."
            ),
            source="FAST Radar",
            status="bez potwierdzenia ML",
            cta=f"Uruchom pełną analizę: {symbol}",
            action="full_analysis",
            symbol=symbol,
            tone="neutral",
            meta={"horizon": _horizon_short(fast_leader)},
        ))

    if not open_positions:
        add(_card(
            card_id="forward_empty",
            priority=40,
            title="Forward proof nie ma teraz otwartej pozycji.",
            body="To poprawny stan selektywnego systemu: czasem najlepszą decyzją badawczą jest brak nowej ekspozycji.",
            source="Forward proof",
            status=proof_label,
            cta="Pokaż Forward",
            action="show_forward_details",
            tone="info",
        ))

    if records:
        stage_headline, stage_detail = _scan_stage_text(snapshot, universe_size)
        add(_card(
            card_id="radar_overview",
            priority=30,
            title="Przejrzyj dzisiejszy radar po filtrach FAST/ML.",
            body=f"Snapshot zawiera {len(records)} wierszy/horyzontów dla universe {universe_size or snapshot.get('universe_total') or '—'} instrumentów. {stage_detail}. Użyj go jako mapy pracy, nie listy transakcji.",
            source="Radar",
            status=f"{stage_headline} · świeżość: {radar_freshness}",
            cta="Pokaż top snapshotu",
            action="show_radar_snapshot",
            tone="neutral",
        ))
    else:
        add(_card(
            card_id="radar_empty",
            priority=30,
            title="Radar nie ma jeszcze danych do prowadzenia użytkownika.",
            body="Uruchom albo poczekaj na skan. Na razie Start pokazuje głównie status proof flow i automatu.",
            source="Radar",
            status=status,
            cta="Pokaż status radaru",
            action="show_radar_snapshot",
            tone="warn",
        ))

    cards = sorted(cards, key=lambda card: card["priority"], reverse=True)[:max_cards]
    if len(cards) < 3:
        cards.append(_card(
            card_id="methodology_guardrail",
            priority=10,
            title="Czytaj guidance jako plan pracy w aplikacji.",
            body="MarketScope wskazuje, co sprawdzić i dlaczego. Nie zastępuje decyzji inwestora ani kontroli ryzyka.",
            source="Metodologia",
            status="research only",
            cta="Pokaż zasady interpretacji",
            action="show_methodology_hint",
            tone="info",
        ))

    return {
        "title": "Co dziś warto zrobić w MarketScope?",
        "subtitle": "Krótka lista pracy w aplikacji: co sprawdzić, dlaczego to ważne i z jakiego źródła pochodzi sygnał.",
        "freshness": radar_freshness,
        "warning": warning,
        "cards": cards[:max_cards],
        "stats": {
            "cards": len(cards[:max_cards]),
            "radar_records": len(records),
            "journal_total": (journal or {}).get("total"),
            "proof": proof_label,
        },
    }
