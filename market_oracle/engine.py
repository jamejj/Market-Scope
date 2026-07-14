from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .data import download_history
from .features import build_features, supervised_frame
from .model import fit_forecast
from .risk import risk_metrics


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
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return {
        "return_1d": float(close.pct_change(1).iloc[-1]),
        "return_5d": float(close.pct_change(5).iloc[-1]),
        "return_20d": float(close.pct_change(20).iloc[-1]),
        "return_60d": float(close.pct_change(60).iloc[-1]),
        "rsi_14": float(rsi.iloc[-1]),
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
    if forecast["quality"].startswith("NISKA"):
        return "BRAK SYGNAŁU"
    probability = forecast["probability_up"]
    expected = forecast["expected_return"]
    if probability >= 0.62 and expected > 0:
        return "SILNY KANDYDAT WZROSTOWY"
    if probability >= 0.55 and expected > 0:
        return "KANDYDAT WZROSTOWY"
    if probability <= 0.38 and expected < 0:
        return "SILNE RYZYKO SPADKU"
    if probability <= 0.45 and expected < 0:
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
    return {
        "Symbol": symbol, "Klasa": _asset_class(symbol), "Horyzont": horizon,
        "Data": str(result["last_date"].date()), "Setup": _setup_label(technical, horizon), "Cena": result["last_price"],
        "Radar momentum": momentum_radar_label(symbol, technical),
        "Radar score": momentum_radar_score(symbol, technical, r),
        "Risk/reward": rr["risk_reward"], "Edge score": rr["edge_score"], "Akcja radaru": rr["radar_action"],
        "Ocena": observation_label(f), "Sygnał": signal_label(f["probability_up"], f["quality"]),
        "P(wzrost)": f["probability_up"], "Oczekiwany ruch": f["expected_return"],
        "Dolna granica 90%": f["lower_return"], "Górna granica 90%": f["upper_return"],
        "Zwrot 1d": technical["return_1d"], "Zwrot 5d": technical["return_5d"], "Zwrot 20d": technical["return_20d"],
        "RSI 14": technical["rsi_14"], "AUC walidacji": f["auc"], "Brier": f["brier"],
        "Jakość modelu": f["quality"], "Pewność": confidence * quality,
        "Zmienność roczna": r["annual_volatility"], "Max drawdown": r["max_drawdown"], "Score": score,
    }


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
