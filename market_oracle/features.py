from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = data["Close"].shift(1)
    tr = pd.concat(
        [(data["High"] - data["Low"]), (data["High"] - prev).abs(), (data["Low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_features(data: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create stationary, past-only technical features."""
    c = data["Close"].astype(float)
    v = data["Volume"].astype(float)
    logret = np.log(c).diff()
    out = pd.DataFrame(index=data.index)

    for n in (1, 2, 5, 10, 20, 60):
        out[f"ret_{n}"] = np.log(c / c.shift(n))
    for n in (5, 10, 20, 60):
        out[f"vol_{n}"] = logret.rolling(n).std() * np.sqrt(252)
    for n in (10, 20, 50, 100, 200):
        ma = c.rolling(n).mean()
        out[f"ma_dist_{n}"] = c / ma - 1
    out["rsi_14"] = (_rsi(c, 14) - 50) / 50

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / c
    out["macd"] = macd
    out["macd_signal"] = macd - macd.ewm(span=9, adjust=False).mean()

    mean20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bollinger_z"] = (c - mean20) / std20.replace(0, np.nan)
    out["atr_pct"] = _atr(data, 14) / c
    low14, high14 = data["Low"].rolling(14).min(), data["High"].rolling(14).max()
    out["stochastic"] = (c - low14) / (high14 - low14).replace(0, np.nan) - 0.5

    logv = np.log1p(v)
    out["volume_z20"] = (logv - logv.rolling(20).mean()) / logv.rolling(20).std()
    out["volume_change"] = logv.diff(5)
    signed_volume = np.sign(c.diff()).fillna(0) * v
    obv = signed_volume.cumsum()
    out["obv_trend"] = obv.diff(20) / v.rolling(20).sum().replace(0, np.nan)
    out["range_pct"] = (data["High"] - data["Low"]) / c
    out["gap"] = data["Open"] / c.shift(1) - 1

    # Context makes a stock forecast relative to its broad market instead of treating it in isolation.
    if context is not None and not context.empty:
        market_close = context["Close"].reindex(out.index).ffill()
        market_ret = np.log(market_close).diff()
        for n in (1, 5, 20, 60):
            out[f"market_ret_{n}"] = np.log(market_close / market_close.shift(n))
            out[f"relative_strength_{n}"] = out[f"ret_{n}"] - out[f"market_ret_{n}"]
        out["market_vol_20"] = market_ret.rolling(20).std() * np.sqrt(252)
        covariance = logret.rolling(60).cov(market_ret)
        out["market_beta_60"] = covariance / market_ret.rolling(60).var().replace(0, np.nan)
        out["market_corr_60"] = logret.rolling(60).corr(market_ret)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def supervised_frame(data: pd.DataFrame, horizon: int, context: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = build_features(data, context)
    forward_return = data["Close"].shift(-horizon) / data["Close"] - 1
    target = (forward_return > 0).astype(float)
    valid = features.notna().all(axis=1) & forward_return.notna()
    return features.loc[valid], target.loc[valid].astype(int), forward_return.loc[valid]
