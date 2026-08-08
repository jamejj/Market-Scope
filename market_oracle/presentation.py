from __future__ import annotations

import math
from typing import Any

from .signals import DEFAULT_SIGNAL_THRESHOLD, SignalInputs, SignalVerdict, signal_verdict


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


def _forecast_inputs(forecast: dict) -> SignalInputs:
    probability = _finite_float(forecast.get("probability_up"))
    expected_return = _finite_float(forecast.get("expected_return"))
    return SignalInputs(
        probability=0.5 if probability is None else probability,
        expected_return=0.0 if expected_return is None else expected_return,
        quality=str(forecast.get("quality") or "NISKA — BRAK PRZEWAGI"),
        auc=_finite_float(forecast.get("auc")),
        brier=_finite_float(forecast.get("brier")),
        source="FULL_ANALYSIS",
    )


def _forecast_verdict(forecast: dict) -> SignalVerdict:
    return signal_verdict(_forecast_inputs(forecast), threshold=DEFAULT_SIGNAL_THRESHOLD)


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
    return max(
        items,
        key=lambda item: (
            _verdict_rank(_forecast_verdict(item[1])),
            _quality_rank(str(item[1].get("quality") or "")),
            abs((_finite_float(item[1].get("probability_up")) or 0.5) - 0.5),
            abs(_finite_float(item[1].get("expected_return")) or 0.0),
            preferred.get(item[0], 0),
        ),
    )


def _reason_label(reason: str) -> str:
    labels = {
        "LONG_CONFIRMED": "wzrost potwierdzony przez wspólną bramkę",
        "SHORT_CONFIRMED": "spadek potwierdzony przez wspólną bramkę",
        "LOW_QUALITY": "niska jakość walidacji",
        "EXPECTED_RETURN_CONFLICT": "konflikt P(wzrost) z oczekiwanym ruchem",
        "EXPECTED_RETURN_TOO_SMALL": "oczekiwany ruch za mały",
        "PROBABILITY_INSIDE_BAND": "prawdopodobieństwo wewnątrz pasma obserwacji",
    }
    return labels.get(reason, reason.replace("_", " ").lower())


def _direction(verdict: SignalVerdict) -> tuple[str, str]:
    reason = _reason_label(verdict.reason)
    if verdict.decision == 1:
        return "Potwierdzony kandydat wzrostowy", f"Wspólna bramka MarketScope zwraca LONG: {reason}."
    if verdict.decision == -1:
        return "Potwierdzone ryzyko spadku", f"Wspólna bramka MarketScope zwraca SHORT: {reason}."
    if verdict.reason == "LOW_QUALITY":
        return "Brak potwierdzonej przewagi", "Model został celowo ściągnięty w stronę 50%, bo walidacja nie pokazała stabilnej przewagi."
    return "Obserwuj", f"Wspólna bramka MarketScope nie potwierdza wejścia: {reason}."


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


def build_analysis_report(result: dict, profile: dict | None = None, source_context: dict | None = None) -> dict:
    """Human-readable report layer for full instrument analysis.

    This function deliberately does not change models, thresholds or signal decisions.
    It only translates the already-computed analysis dictionary into product copy.
    """
    profile = profile or {}
    source_context = source_context or {}
    symbol = str(result.get("symbol") or "—")
    crypto = symbol.upper().endswith("-USD")
    forecasts = result.get("forecasts") or {}
    horizon, forecast = _primary_forecast(forecasts)
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
        headline = f"{symbol}: potwierdzony kandydat wzrostowy na horyzoncie {horizon_text}."
    elif verdict.decision == -1:
        headline = f"{symbol}: potwierdzone ryzyko spadku na horyzoncie {horizon_text}."
    elif verdict.reason == "LOW_QUALITY":
        headline = f"{symbol}: brak potwierdzonej przewagi modelu — obserwuj, nie zakładaj edge."
    else:
        headline = f"{symbol}: obserwuj — bramka MarketScope nie potwierdza wejścia na horyzoncie {horizon_text}."

    rsi = _finite_float(technical.get("rsi_14"))
    rsi_text = "—" if rsi is None else f"{rsi:.1f}"

    evidence = [
        f"Horyzont roboczy raportu: {horizon_text}; jakość walidacji: {quality}.",
        f"P(wzrost) { _pct(probability) }, oczekiwany ruch {expected}, zakres 90%: {lower} – {upper}.",
        f"Decyzja bramki: {verdict.label} — {_reason_label(verdict.reason)}.",
        f"Technicznie: {trend}; zwrot 20 sesji/dni { _signed_pct(technical.get('return_20d')) }, RSI 14: {rsi_text}.",
    ]
    if auc is not None and brier is not None:
        evidence.append(f"Walidacja: AUC {auc:.3f}, Brier {brier:.3f}; to mówi o jakości kierunku i kalibracji, nie o gwarancji zysku.")

    counterpoints: list[str] = []
    if verdict.decision == 0 and not quality.startswith("NISKA"):
        counterpoints.append(f"Wspólna bramka MarketScope zwraca OBSERWUJ ({_reason_label(verdict.reason)}), więc raport nie promuje wejścia mimo pojedynczych mocnych metryk.")
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

    cards = [
        ("Co system widzi?", direction, direction_detail),
        ("Jak mocny dowód?", quality, f"AUC {auc:.3f} · Brier {brier:.3f}" if auc is not None and brier is not None else "Brak pełnych metryk walidacji"),
        ("Horyzont", horizon_text, f"Ten horyzont ma teraz najwyższy priorytet raportu; pozostałe są poniżej."),
        ("Oczekiwany ruch", expected, f"Zakres 90%: {lower} – {upper}"),
        ("Trend", trend, f"1d {_signed_pct(technical.get('return_1d'))} · 5d {_signed_pct(technical.get('return_5d'))} · 20d {_signed_pct(technical.get('return_20d'))}"),
        ("Max drawdown", _signed_pct(drawdown), "Historyczne obsunięcie; niżej w raporcie są pełne metryki ryzyka."),
    ]
    horizon_cards = []
    for h, f in _forecast_items(forecasts):
        horizon_cards.append({
            "label": _horizon_label(h, crypto),
            "probability": _pct(f.get("probability_up")),
            "expected": _signed_pct(f.get("expected_return")),
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
