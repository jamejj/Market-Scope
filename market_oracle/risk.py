from __future__ import annotations

import numpy as np
import pandas as pd


def risk_metrics(close: pd.Series) -> dict[str, float]:
    ret = close.pct_change().dropna()
    recent = ret.tail(252 * 3)
    wealth = (1 + ret).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    downside = recent[recent < 0].std() * np.sqrt(252)
    var95 = recent.quantile(0.05)
    cvar95 = recent[recent <= var95].mean()
    ann_return = recent.mean() * 252
    ann_vol = recent.std() * np.sqrt(252)
    return {
        "annual_return": float(ann_return), "annual_volatility": float(ann_vol),
        "downside_volatility": float(downside), "max_drawdown": float(drawdown.min()),
        "var_95_daily": float(var95), "cvar_95_daily": float(cvar95),
        "sharpe_zero_rf": float(ann_return / ann_vol) if ann_vol else 0.0,
    }
