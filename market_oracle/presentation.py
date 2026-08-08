from __future__ import annotations

import math
from typing import Any


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


def _is_confirmed(forecast: dict) -> bool:
    return not str(forecast.get("quality") or "").startswith("NISKA")


def _primary_forecast(forecasts: dict[Any, dict]) -> tuple[int, dict]:
    items = _forecast_items(forecasts)
    if not items:
        raise ValueError("Brak prognoz do zbudowania raportu analizy.")
    confirmed = [(horizon, forecast) for horizon, forecast in items if _is_confirmed(forecast)]
    if confirmed:
        return max(
            confirmed,
            key=lambda item: (
                str(item[1].get("quality") or "") == "WYSOKA",
                abs(float(item[1].get("probability_up") or 0.5) - 0.5),
                item[0],
            ),
        )
    preferred = {20: 4, 5: 3, 60: 2, 1: 1}
    return max(items, key=lambda item: (preferred.get(item[0], 0), abs(float(item[1].get("probability_up") or 0.5) - 0.5)))


def _direction(probability: float, quality: str) -> tuple[str, str]:
    if quality.startswith("NISKA"):
        return "Brak potwierdzonej przewagi", "Model został celowo ściągnięty w stronę 50%, bo walidacja nie pokazała stabilnej przewagi."
    if probability >= 0.62:
        return "Silny kandydat wzrostowy", "Prawdopodobieństwo i jakość modelu wspierają kierunek wzrostowy."
    if probability >= 0.54:
        return "Kandydat wzrostowy", "Kierunek jest dodatni, ale nadal wymaga kontroli ryzyka i kontekstu wykresu."
    if probability <= 0.38:
        return "Silne ryzyko spadku", "Model wskazuje podwyższone ryzyko kierunku spadkowego."
    if probability <= 0.46:
        return "Ryzyko spadku", "Kierunek jest negatywny, ale decyzja nadal zależy od kontekstu i ryzyka."
    return "Obserwuj", "Model jest blisko neutralnego 50/50; większe znaczenie ma trend, momentum i ryzyko."


def _trend_label(technical: dict) -> str:
    points = sum([
        bool((technical.get("return_20d") or 0) > 0),
        bool((technical.get("rsi_14") or 0) >= 50),
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
    probability = float(forecast.get("probability_up") or 0.5)
    direction, direction_detail = _direction(probability, quality)
    technical = result.get("technical") or {}
    risk = result.get("risk") or {}
    trend = _trend_label(technical)
    horizon_text = _horizon_label(horizon, crypto)
    expected = _signed_pct(forecast.get("expected_return"))
    lower = _signed_pct(forecast.get("lower_return"))
    upper = _signed_pct(forecast.get("upper_return"))
    auc = _finite_float(forecast.get("auc"))
    brier = _finite_float(forecast.get("brier"))

    headline = f"{symbol}: {direction.lower()} na horyzoncie {horizon_text}."
    if quality.startswith("NISKA"):
        headline = f"{symbol}: brak potwierdzonej przewagi modelu — obserwuj, nie zakładaj edge."

    evidence = [
        f"Horyzont roboczy raportu: {horizon_text}; jakość walidacji: {quality}.",
        f"P(wzrost) { _pct(probability) }, oczekiwany ruch {expected}, zakres 90%: {lower} – {upper}.",
        f"Technicznie: {trend}; zwrot 20 sesji/dni { _signed_pct(technical.get('return_20d')) }, RSI 14: {(_finite_float(technical.get('rsi_14')) or 0):.1f}.",
    ]
    if auc is not None and brier is not None:
        evidence.append(f"Walidacja: AUC {auc:.3f}, Brier {brier:.3f}; to mówi o jakości kierunku i kalibracji, nie o gwarancji zysku.")

    counterpoints: list[str] = []
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
            "auc": f"{float(f.get('auc') or 0.5):.3f}",
            "brier": f"{float(f.get('brier') or 0.25):.3f}",
        })

    full_data_as_of = result.get("last_date")
    if hasattr(full_data_as_of, "date"):
        full_data_as_of = str(full_data_as_of.date())
    freshness = {
        "radar": _freshness_value(source_context.get("radar_updated_at")),
        "analysis": _freshness_value(full_data_as_of),
        "benchmark": str(result.get("benchmark") or "—"),
    }
    if freshness["radar"] != "—" and freshness["analysis"] != "—" and not freshness["radar"].startswith(freshness["analysis"]):
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
