from __future__ import annotations

import numpy as np
import pandas as pd


def periods_per_year(index: pd.Index) -> int:
    """Use 365 for seven-day markets such as crypto, otherwise trading-year 252."""
    dates = pd.DatetimeIndex(index)
    weekend_share = float((dates.dayofweek >= 5).mean()) if len(dates) else 0.0
    return 365 if weekend_share > 0.10 else 252


def risk_metrics(close: pd.Series) -> dict[str, float]:
    ret = close.pct_change().dropna()
    annualizer = periods_per_year(close.index)
    recent = ret.tail(annualizer * 3)
    wealth = (1 + ret).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    downside = recent[recent < 0].std() * np.sqrt(annualizer)
    var95 = recent.quantile(0.05)
    cvar95 = recent[recent <= var95].mean()
    ann_return = recent.mean() * annualizer
    ann_vol = recent.std() * np.sqrt(annualizer)
    return {
        "annual_return": float(ann_return), "annual_volatility": float(ann_vol),
        "downside_volatility": float(downside), "max_drawdown": float(drawdown.min()),
        "var_95_daily": float(var95), "cvar_95_daily": float(cvar95),
        "sharpe_zero_rf": float(ann_return / ann_vol) if ann_vol else 0.0,
        "periods_per_year": float(annualizer),
    }
