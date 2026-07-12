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
    return {
        "Symbol": symbol, "Klasa": _asset_class(symbol), "Horyzont": horizon,
        "Setup": _setup_label(technical, horizon), "Cena": result["last_price"],
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
