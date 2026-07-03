from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd


class MarketDataError(RuntimeError):
    pass


def download_history(symbol: str, years: int = 8) -> pd.DataFrame:
    """Download adjusted daily OHLCV data. yfinance is research-only data."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise MarketDataError("Brak yfinance. Uruchom: pip install -r requirements.txt") from exc

    end = datetime.now(timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=int(years * 365.25) + 40)
    try:
        frame = yf.download(
            symbol.strip().upper(), start=start.date(), end=end.date(),
            auto_adjust=True, progress=False, threads=False,
        )
    except Exception as exc:
        raise MarketDataError(f"Nie udało się pobrać {symbol}: {exc}") from exc
    if frame.empty:
        raise MarketDataError(f"Brak danych dla symbolu {symbol}.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in frame]
    if missing:
        raise MarketDataError(f"Brak kolumn dla {symbol}: {', '.join(missing)}")
    frame = frame[required].copy().dropna(subset=["Close"])
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if len(frame) < 320:
        raise MarketDataError(f"Za mało historii dla {symbol}: {len(frame)} sesji (minimum 320).")
    return frame


def download_profile(symbol: str) -> dict:
    """Best-effort company/fund metadata; missing fields are returned as None."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol.strip().upper()).info or {}
    except Exception:
        return {}
    fields = (
        "longName", "shortName", "quoteType", "sector", "industry", "country", "currency",
        "exchange", "marketCap", "trailingPE", "forwardPE", "priceToBook", "dividendYield",
        "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "totalAssets", "category",
    )
    return {field: info.get(field) for field in fields}
