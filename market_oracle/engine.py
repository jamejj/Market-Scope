from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .data import download_history
from .features import build_features, supervised_frame
from .model import fit_forecast
from .risk import risk_metrics
from .signals import signal_decision, signal_inputs_from_forecast


def _benchmark_for(symbol: str) -> str | None:
    symbol = symbol.upper()
    if symbol.endswith("-USD"):
        return "BTC-USD" if symbol != "BTC-USD" else None
    if symbol.endswith(".WA") or symbol in {"^WIG20", "WIG20.WA"}:
        # Yahoo exposes almost no history for the raw WIG20 symbol; this liquid total-return ETF is the usable proxy.
        return "ETFBW20TR.WA"
    return "^GSPC"


def _technical_snapshot(data: pd.DataFrame) -> dict[str, float | bool]:
    close = data["Close"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float)
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    prev = close.shift(1)
    true_range = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    rolling_high_60 = close.rolling(60).max()
    rolling_low_60 = close.rolling(60).min()
    volume_20 = volume.rolling(20).mean()
    volume_base = float(volume_20.iloc[-1]) if np.isfinite(volume_20.iloc[-1]) else 0.0
    range_width = float(rolling_high_60.iloc[-1] - rolling_low_60.iloc[-1])
    range_position_60 = 0.5 if not np.isfinite(range_width) or range_width <= 0 else float((close.iloc[-1] - rolling_low_60.iloc[-1]) / range_width)
    return {
        "return_1d": float(close.pct_change(1).iloc[-1]),
        "return_5d": float(close.pct_change(5).iloc[-1]),
        "return_20d": float(close.pct_change(20).iloc[-1]),
        "return_60d": float(close.pct_change(60).iloc[-1]),
        "momentum_acceleration": float(close.pct_change(5).iloc[-1] - close.pct_change(20).iloc[-1] / 4),
        "rsi_14": float(rsi.iloc[-1]),
        "atr_pct": float(true_range.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1] / close.iloc[-1]),
        "relative_volume_20": float(volume.iloc[-1] / volume_base - 1) if volume_base > 0 else 0.0,
        "avg_dollar_volume_20": float((close * volume).rolling(20).mean().iloc[-1]),
        "drawdown_60": float(close.iloc[-1] / rolling_high_60.iloc[-1] - 1),
        "range_position_60": range_position_60,
        "above_sma_50": bool(close.iloc[-1] > close.rolling(50).mean().iloc[-1]),
        "above_sma_200": bool(close.iloc[-1] > close.rolling(200).mean().iloc[-1]),
        "near_20d_high": bool(close.iloc[-1] >= close.rolling(20).max().iloc[-1] * 0.98),
        "near_60d_high": bool(close.iloc[-1] >= close.rolling(60).max().iloc[-1] * 0.98),
    }


def analyze_asset(symbol: str, horizons: tuple[int, ...] = (1, 5, 20), years: int = 8) -> dict:
    data = download_history(symbol, years)
    benchmark_symbol = _benchmark_for(symbol)
    if benchmark_symbol:
        try:
            context = download_history(benchmark_symbol, years)
        except Exception:
            context = None
            benchmark_symbol = "brak"
    else:
        context = None
        benchmark_symbol = "brak — BTC jest benchmarkiem krypto"
    all_features = build_features(data, context).dropna()
    if all_features.empty:
        raise ValueError(f"Nie udało się zbudować cech dla {symbol}.")
    forecasts = {}
    for horizon in horizons:
        X, y, returns = supervised_frame(data, horizon, context)
        forecast = fit_forecast(X, y, returns, all_features.iloc[[-1]], horizon)
        forecasts[horizon] = asdict(forecast)
    return {
        "symbol": symbol.upper(), "last_date": data.index[-1], "last_price": float(data["Close"].iloc[-1]),
        "forecasts": forecasts, "risk": risk_metrics(data["Close"]), "history": data,
        "benchmark": benchmark_symbol, "technical": _technical_snapshot(data),
    }


def signal_label(probability: float, quality: str | None = None) -> str:
    if quality and quality.startswith("NISKA"):
        return "BRAK PRZEWAGI"
    if probability >= 0.62:
        return "SILNY WZROSTOWY"
    if probability >= 0.54:
        return "WZROSTOWY"
    if probability <= 0.38:
        return "SILNY SPADKOWY"
    if probability <= 0.46:
        return "SPADKOWY"
    return "NEUTRALNY"


def observation_label(forecast: dict) -> str:
    decision = signal_decision(signal_inputs_from_forecast(forecast), threshold=0.55)
    if decision == 0 and forecast["quality"].startswith("NISKA"):
        return "BRAK SYGNAŁU"
    probability = forecast["probability_up"]
    expected = forecast["expected_return"]
    if decision == 1 and probability >= 0.62:
        return "SILNY KANDYDAT WZROSTOWY"
    if decision == 1:
        return "KANDYDAT WZROSTOWY"
    if decision == -1 and probability <= 0.38:
        return "SILNE RYZYKO SPADKU"
    if decision == -1:
        return "RYZYKO SPADKU"
    return "OBSERWUJ"


def momentum_radar_label(symbol: str, technical: dict) -> str:
    """Fast discovery layer: flags unusual moves even when ML has no confirmed edge."""
    r1 = technical.get("return_1d", 0.0)
    r5 = technical.get("return_5d", 0.0)
    r20 = technical.get("return_20d", 0.0)
    rsi = technical.get("rsi_14", 50.0)
    crypto = symbol.endswith("-USD")
    hot_1d = 0.08 if crypto else 0.025
    hot_5d = 0.18 if crypto else 0.06
    hot_20d = 0.35 if crypto else 0.12
    panic_1d = -hot_1d
    panic_5d = -hot_5d
    panic_20d = -hot_20d

    if r1 <= panic_1d or r5 <= panic_5d or r20 <= panic_20d:
        return "PANIKA / RYZYKO"
    if r1 >= hot_1d or r5 >= hot_5d or r20 >= hot_20d:
        return "PEREŁKA MOMENTUM"
    if technical.get("near_20d_high") and r20 > (0.08 if crypto else 0.04) and 45 <= rsi <= 82:
        return "BREAKOUT WATCH"
    if r5 > (0.08 if crypto else 0.03) and r20 > (0.06 if crypto else 0.025):
        return "MOMENTUM WATCH"
    return "—"


def momentum_radar_score(symbol: str, technical: dict, risk: dict) -> float:
    """Ranking score for discovery; intentionally separate from validated ML score."""
    r1 = technical.get("return_1d", 0.0)
    r5 = technical.get("return_5d", 0.0)
    r20 = technical.get("return_20d", 0.0)
    r60 = technical.get("return_60d", 0.0)
    rsi = technical.get("rsi_14", 50.0)
    volatility = risk.get("annual_volatility", 0.0)
    trend_bonus = (
        (1.2 if technical.get("above_sma_50") else -0.4)
        + (0.8 if technical.get("above_sma_200") else 0.0)
        + (1.1 if technical.get("near_20d_high") else 0.0)
    )
    overheating_penalty = max(0.0, rsi - 82) * 0.08
    risk_penalty = min(3.0, float(volatility or 0.0)) * (0.8 if symbol.endswith("-USD") else 1.1)
    return float(r1 * 300 + r5 * 180 + r20 * 80 + r60 * 18 + trend_bonus - overheating_penalty - risk_penalty)


def risk_reward_metrics(forecast: dict, risk: dict, technical: dict) -> dict[str, float | str]:
    expected = float(forecast.get("expected_return") or 0.0)
    lower = float(forecast.get("lower_return") or 0.0)
    upper = float(forecast.get("upper_return") or 0.0)
    probability = float(forecast.get("probability_up") or 0.5)
    auc = float(forecast.get("auc") or 0.5)
    brier = float(forecast.get("brier") or 0.25)
    volatility = float(risk.get("annual_volatility") or 0.0)
    downside = max(abs(min(lower, 0.0)), volatility / 16, 0.005)
    upside = max(max(upper, expected, 0.0), 0.0)
    risk_reward = upside / downside if downside else 0.0
    model_quality = max(0.0, min(1.0, (auc - 0.50) / 0.12)) * max(0.0, min(1.0, (0.27 - brier) / 0.06))
    trend_quality = (
        (0.45 if technical.get("above_sma_50") else -0.15)
        + (0.35 if technical.get("above_sma_200") else 0.0)
        + (0.25 if technical.get("near_20d_high") else 0.0)
    )
    edge_score = (
        expected * 120
        + (probability - 0.5) * 90
        + min(risk_reward, 4.0) * 0.9
        + model_quality * 2.4
        + trend_quality
        - min(volatility, 3.0) * 0.35
    )
    if forecast.get("quality", "").startswith("NISKA"):
        action = "OBSERWUJ — BRAK EDGE ML"
    elif edge_score >= 4.5 and expected > 0:
        action = "PRIORYTET DO ANALIZY"
    elif edge_score >= 2.5 and expected > 0:
        action = "WATCHLIST"
    elif expected < 0 or probability < 0.45:
        action = "RYZYKO / UNIKAJ"
    else:
        action = "NEUTRALNIE"
    return {
        "risk_reward": float(risk_reward),
        "edge_score": float(edge_score),
        "radar_action": action,
    }


def _clip_score(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(0.0, min(100.0, value)))


def setup_intelligence(symbol: str, forecast: dict, risk: dict, technical: dict) -> dict[str, float | str]:
    """Explainable trader-facing scorecard for ranking setup quality."""
    crypto = symbol.upper().endswith("-USD")
    r1 = float(technical.get("return_1d") or 0.0)
    r5 = float(technical.get("return_5d") or 0.0)
    r20 = float(technical.get("return_20d") or 0.0)
    r60 = float(technical.get("return_60d") or 0.0)
    rsi = float(technical.get("rsi_14") or 50.0)
    acceleration = float(technical.get("momentum_acceleration") or 0.0)
    relative_volume = float(technical.get("relative_volume_20") or 0.0)
    range_position = float(technical.get("range_position_60") or 0.5)
    annual_volatility = float(risk.get("annual_volatility") or 0.0)
    max_drawdown = abs(float(risk.get("max_drawdown") or 0.0))
    atr_pct = float(technical.get("atr_pct") or 0.0)
    dollar_volume = max(float(technical.get("avg_dollar_volume_20") or 0.0), 1.0)
    expected = float(forecast.get("expected_return") or 0.0)
    probability = float(forecast.get("probability_up") or 0.5)
    auc = float(forecast.get("auc") or 0.5)
    brier = float(forecast.get("brier") or 0.25)
    risk_reward = float(risk_reward_metrics(forecast, risk, technical)["risk_reward"])

    # Crypto has naturally wider moves, so its momentum thresholds are deliberately higher.
    hot_1d = 0.08 if crypto else 0.025
    hot_5d = 0.18 if crypto else 0.06
    hot_20d = 0.35 if crypto else 0.12
    overheating = max(0.0, (rsi - 78) / 22)

    momentum_score = _clip_score(
        45
        + (r1 / hot_1d) * 10
        + (r5 / hot_5d) * 18
        + (r20 / hot_20d) * 18
        + (r60 / max(hot_20d * 1.8, 0.01)) * 8
        + min(relative_volume, 2.5) * 5
        + (acceleration / hot_5d) * 10
        - overheating * 18
    )
    trend_score = _clip_score(
        30
        + (22 if technical.get("above_sma_50") else -8)
        + (20 if technical.get("above_sma_200") else -4)
        + (12 if technical.get("near_20d_high") else 0)
        + (10 if technical.get("near_60d_high") else 0)
        + (range_position - 0.5) * 26
        + max(min(r20, hot_20d), -hot_20d) / hot_20d * 10
    )
    risk_control = _clip_score(
        86
        - min(annual_volatility, 3.0) * (30 if crypto else 55)
        - min(max_drawdown, 0.90) * 32
        - min(atr_pct, 0.18) * (110 if crypto else 170)
        - max(0.0, rsi - 82) * 0.65
    )
    liquidity_score = _clip_score((np.log10(dollar_volume) - (5.2 if crypto else 5.6)) * 25)
    auc_quality = max(0.0, min(1.0, (auc - 0.50) / 0.12))
    brier_quality = max(0.0, min(1.0, (0.27 - brier) / 0.06))
    ml_score = _clip_score(
        auc_quality * brier_quality * 62
        + max(0.0, probability - 0.5) * 120
        + max(0.0, expected) * 180
    )
    rr_score = _clip_score(min(risk_reward, 4.0) / 4 * 100)
    setup_score = _clip_score(
        momentum_score * 0.23
        + trend_score * 0.20
        + risk_control * 0.18
        + ml_score * 0.22
        + liquidity_score * 0.07
        + rr_score * 0.10
    )

    if setup_score >= 72 and ml_score >= 45 and risk_control >= 45:
        grade = "A — czysty setup"
    elif setup_score >= 62 and risk_control >= 35:
        grade = "B — watchlist"
    elif momentum_score >= 72 and risk_control >= 30:
        grade = "M — momentum do sprawdzenia"
    elif risk_control < 28:
        grade = "R — ryzyko dominuje"
    else:
        grade = "C — obserwuj"

    reasons: list[str] = []
    if momentum_score >= 70:
        reasons.append("silne momentum")
    elif r5 > 0 and r20 > 0:
        reasons.append("dodatni impet")
    if trend_score >= 70:
        reasons.append("trend 50/200 wspiera ruch")
    elif technical.get("near_20d_high"):
        reasons.append("blisko wybicia 20d")
    if ml_score >= 55:
        reasons.append("ML potwierdza edge")
    elif forecast.get("quality", "").startswith("NISKA"):
        reasons.append("ML bez przewagi")
    if relative_volume >= 0.35:
        reasons.append("wolumen powyżej normy")
    if risk_control < 35:
        reasons.append("wysokie ryzyko/zmienność")
    if not reasons:
        reasons.append("brak dominującego czynnika")

    return {
        "setup_score": setup_score,
        "setup_grade": grade,
        "momentum_score": momentum_score,
        "trend_score": trend_score,
        "risk_control": risk_control,
        "liquidity_score": liquidity_score,
        "ml_score": ml_score,
        "thesis": " · ".join(reasons[:4]),
    }


def _fast_forecast_proxy(symbol: str, horizon: int, technical: dict, risk: dict) -> dict:
    """Cheap directional proxy used only for first-pass ranking, not as a validated ML forecast."""
    crypto = symbol.upper().endswith("-USD")
    periods = 365 if crypto else 252
    horizon_risk = max(float(risk.get("annual_volatility") or 0.0) * np.sqrt(max(horizon, 1) / periods), 0.006)
    trend_bias = (
        (0.012 if technical.get("above_sma_50") else -0.004)
        + (0.010 if technical.get("above_sma_200") else 0.0)
        + (0.008 if technical.get("near_20d_high") else 0.0)
    )
    if horizon <= 1:
        raw_expected = (
            float(technical.get("return_1d") or 0.0) * 0.28
            + float(technical.get("return_5d") or 0.0) * 0.06
            + trend_bias * 0.20
        )
    elif horizon <= 5:
        raw_expected = (
            float(technical.get("return_5d") or 0.0) * 0.24
            + float(technical.get("return_20d") or 0.0) * 0.07
            + trend_bias * 0.45
        )
    else:
        raw_expected = (
            float(technical.get("return_20d") or 0.0) * 0.18
            + float(technical.get("return_60d") or 0.0) * 0.05
            + trend_bias * 0.75
        )
    cap = 0.22 if crypto else 0.10
    expected = float(max(-cap, min(cap, raw_expected)))
    strength = max(-8.0, min(8.0, expected / horizon_risk))
    probability = float(max(0.25, min(0.75, 1 / (1 + np.exp(-strength * 0.55)))))
    interval = max(horizon_risk * 1.75, abs(expected) * 1.7, 0.012)
    return {
        "quality": "NISKA — FAST RADAR BEZ ML",
        "probability_up": probability,
        "expected_return": expected,
        "lower_return": expected - interval,
        "upper_return": expected + interval,
        "auc": 0.50,
        "brier": 0.25,
    }


def _fast_action(momentum_label: str, intelligence: dict, expected: float) -> str:
    setup_score = float(intelligence.get("setup_score") or 0.0)
    risk_control = float(intelligence.get("risk_control") or 0.0)
    if momentum_label == "PANIKA / RYZYKO" or risk_control < 28 or expected < -0.025:
        return "RYZYKO / UNIKAJ"
    if setup_score >= 64:
        return "FAST SHORTLIST"
    if momentum_label in {"PEREŁKA MOMENTUM", "BREAKOUT WATCH", "MOMENTUM WATCH"}:
        return "MOMENTUM DO SPRAWDZENIA"
    return "OBSERWUJ"


def _asset_class(symbol: str) -> str:
    if symbol.endswith("-USD"):
        return "Krypto"
    if symbol.endswith(".WA"):
        return "GPW"
    if "." in symbol:
        return "ETF / Europa"
    if symbol.startswith("^"):
        return "Indeks"
    return "USA / ETF"


def _setup_label(technical: dict, horizon: int) -> str:
    rsi = technical["rsi_14"]
    if technical.get("near_20d_high") and technical["return_20d"] > 0 and 45 <= rsi <= 78:
        return "Breakout / momentum"
    if technical["above_sma_50"] and technical["above_sma_200"] and technical["return_20d"] > 0:
        return "Trend continuation"
    if technical["return_5d"] > 0 and technical["return_20d"] < 0 and horizon <= 5:
        return "Odbicie krótkoterminowe"
    if rsi >= 78:
        return "Mocne momentum, ryzyko przegrzania"
    if rsi <= 32:
        return "Wyprzedanie / wysokie ryzyko"
    return "Neutralny setup"


def _row_from_result(symbol: str, result: dict, horizon: int) -> dict:
    f, r = result["forecasts"][horizon], result["risk"]
    technical = result["technical"]
    confidence = abs(f["probability_up"] - 0.5) * 2
    quality = max(0.0, min(1.0, (f["auc"] - 0.45) / 0.20))
    risk_penalty = min(1.5, r["annual_volatility"]) * (8 if symbol.endswith("-USD") else 10)
    momentum_bonus = (
        max(-0.08, min(0.12, technical["return_20d"])) * 25
        + (0.75 if technical["above_sma_50"] else -0.35)
        + (0.55 if technical["near_20d_high"] else 0.0)
    )
    score = (f["probability_up"] - 0.5) * 210 * quality + f["expected_return"] * 90 + momentum_bonus - risk_penalty
    rr = risk_reward_metrics(f, r, technical)
    intelligence = setup_intelligence(symbol, f, r, technical)
    return {
        "Symbol": symbol, "Klasa": _asset_class(symbol), "Horyzont": horizon,
        "Data": str(result["last_date"].date()), "Setup": _setup_label(technical, horizon), "Cena": result["last_price"],
        "Radar momentum": momentum_radar_label(symbol, technical),
        "Radar score": momentum_radar_score(symbol, technical, r),
        "Risk/reward": rr["risk_reward"], "Edge score": rr["edge_score"], "Akcja radaru": rr["radar_action"],
        "Tryb analizy": "ML", "Deep score": intelligence["setup_score"] * 0.75 + max(rr["edge_score"], 0) * 4,
        "Setup score": intelligence["setup_score"], "Setup grade": intelligence["setup_grade"],
        "Momentum score": intelligence["momentum_score"], "Trend score": intelligence["trend_score"],
        "Risk control": intelligence["risk_control"], "Liquidity score": intelligence["liquidity_score"],
        "Model edge": intelligence["ml_score"], "Teza radaru": intelligence["thesis"],
        "Ocena": observation_label(f), "Sygnał": signal_label(f["probability_up"], f["quality"]),
        "P(wzrost)": f["probability_up"], "Oczekiwany ruch": f["expected_return"],
        "Dolna granica 90%": f["lower_return"], "Górna granica 90%": f["upper_return"],
        "Zwrot 1d": technical["return_1d"], "Zwrot 5d": technical["return_5d"], "Zwrot 20d": technical["return_20d"],
        "RSI 14": technical["rsi_14"], "AUC walidacji": f["auc"], "Brier": f["brier"],
        "Jakość modelu": f["quality"], "Pewność": confidence * quality,
        "Zmienność roczna": r["annual_volatility"], "Max drawdown": r["max_drawdown"], "Score": score,
    }


def _fast_row_from_result(symbol: str, result: dict, horizon: int) -> dict:
    r = result["risk"]
    technical = result["technical"]
    f = _fast_forecast_proxy(symbol, horizon, technical, r)
    rr = risk_reward_metrics(f, r, technical)
    intelligence = setup_intelligence(symbol, f, r, technical)
    momentum_label = momentum_radar_label(symbol, technical)
    radar_score = momentum_radar_score(symbol, technical, r)
    action = _fast_action(momentum_label, intelligence, f["expected_return"])
    deep_score = intelligence["setup_score"] * 0.72 + _clip_score(50 + radar_score * 6) * 0.28
    return {
        "Symbol": symbol, "Klasa": _asset_class(symbol), "Horyzont": horizon,
        "Data": str(result["last_date"].date()), "Setup": _setup_label(technical, horizon), "Cena": result["last_price"],
        "Radar momentum": momentum_label, "Radar score": radar_score,
        "Risk/reward": rr["risk_reward"], "Edge score": rr["edge_score"], "Akcja radaru": action,
        "Tryb analizy": "FAST", "Deep score": deep_score,
        "Setup score": intelligence["setup_score"], "Setup grade": intelligence["setup_grade"],
        "Momentum score": intelligence["momentum_score"], "Trend score": intelligence["trend_score"],
        "Risk control": intelligence["risk_control"], "Liquidity score": intelligence["liquidity_score"],
        "Model edge": intelligence["ml_score"], "Teza radaru": intelligence["thesis"],
        "Ocena": "OBSERWUJ", "Sygnał": "FAST RADAR",
        "P(wzrost)": f["probability_up"], "Oczekiwany ruch": f["expected_return"],
        "Dolna granica 90%": f["lower_return"], "Górna granica 90%": f["upper_return"],
        "Zwrot 1d": technical["return_1d"], "Zwrot 5d": technical["return_5d"], "Zwrot 20d": technical["return_20d"],
        "RSI 14": technical["rsi_14"], "AUC walidacji": f["auc"], "Brier": f["brier"],
        "Jakość modelu": "FAST — BEZ ML", "Pewność": 0.0,
        "Zmienność roczna": r["annual_volatility"], "Max drawdown": r["max_drawdown"], "Score": deep_score,
    }


def analyze_fast_asset(symbol: str, horizons: tuple[int, ...] = (1, 5, 20), years: int = 2) -> dict:
    data = download_history(symbol, years)
    return {
        "symbol": symbol.upper(), "last_date": data.index[-1], "last_price": float(data["Close"].iloc[-1]),
        "forecasts": {}, "risk": risk_metrics(data["Close"]), "history": data,
        "benchmark": "fast radar — bez benchmarku ML", "technical": _technical_snapshot(data),
    }


def scan_market_fast(symbols: list[str], horizons: tuple[int, ...] = (1, 5, 20), years: int = 2) -> tuple[pd.DataFrame, dict[str, str]]:
    rows, errors = [], {}
    clean_symbols = dict.fromkeys(s.strip().upper() for s in symbols if s.strip())
    for symbol in clean_symbols:
        try:
            result = analyze_fast_asset(symbol, horizons=horizons, years=years)
            for horizon in horizons:
                rows.append(_fast_row_from_result(symbol, result, horizon))
        except Exception as exc:
            errors[symbol] = str(exc)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["Horyzont", "Deep score"], ascending=[True, False]).reset_index(drop=True)
    return frame, errors


def scan_market(symbols: list[str], horizon: int = 5, years: int = 8) -> tuple[pd.DataFrame, dict[str, str]]:
    rows, errors = [], {}
    for symbol in dict.fromkeys(s.strip().upper() for s in symbols if s.strip()):
        try:
            result = analyze_asset(symbol, horizons=(horizon,), years=years)
            rows.append(_row_from_result(symbol, result, horizon))
        except Exception as exc:
            errors[symbol] = str(exc)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("Score", ascending=False).reset_index(drop=True)
    return frame, errors


def scan_market_multi(symbols: list[str], horizons: tuple[int, ...] = (1, 5, 20), years: int = 8) -> tuple[pd.DataFrame, dict[str, str]]:
    rows, errors = [], {}
    clean_symbols = dict.fromkeys(s.strip().upper() for s in symbols if s.strip())
    for symbol in clean_symbols:
        try:
            result = analyze_asset(symbol, horizons=horizons, years=years)
            for horizon in horizons:
                rows.append(_row_from_result(symbol, result, horizon))
        except Exception as exc:
            errors[symbol] = str(exc)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["Horyzont", "Score"], ascending=[True, False]).reset_index(drop=True)
    return frame, errors
