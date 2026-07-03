from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from .features import supervised_frame


def walk_forward_backtest(data: pd.DataFrame, horizon: int = 5, threshold: float = 0.56, cost_bps: float = 10) -> tuple[pd.DataFrame, dict]:
    """Expanding walk-forward test; predictions are always made out of sample."""
    X, y, forward = supervised_frame(data, horizon)
    start = max(300, int(len(X) * 0.55))
    records = []
    model = None
    for i in range(start, len(X)):
        if model is None or (i - start) % 20 == 0:
            train_end = i - horizon
            if train_end < 200:
                continue
            model = make_pipeline(RobustScaler(), LogisticRegression(C=0.35, max_iter=1500, class_weight="balanced"))
            model.fit(X.iloc[:train_end], y.iloc[:train_end])
        probability = float(model.predict_proba(X.iloc[[i]])[0, 1])
        position = 1 if probability >= threshold else (-1 if probability <= 1 - threshold else 0)
        gross = position * float(forward.iloc[i])
        net = gross - (abs(position) * cost_bps / 10_000)
        records.append({"Date": X.index[i], "Probability": probability, "Position": position, "Return": net})
    result = pd.DataFrame(records).set_index("Date")
    if result.empty:
        raise ValueError("Za mało danych do backtestu.")
    # Horizon trades overlap; divide exposure to avoid pretending each trade has full independent capital.
    result["Strategy"] = result["Return"] / horizon
    result["Equity"] = (1 + result["Strategy"]).cumprod()
    benchmark = data["Close"].reindex(result.index).pct_change().fillna(0)
    result["BuyHold"] = (1 + benchmark).cumprod()
    daily = result["Strategy"]
    metrics = {
        "total_return": float(result["Equity"].iloc[-1] - 1),
        "annual_return": float(result["Equity"].iloc[-1] ** (252 / len(result)) - 1),
        "annual_volatility": float(daily.std() * np.sqrt(252)),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() else 0.0,
        "max_drawdown": float((result["Equity"] / result["Equity"].cummax() - 1).min()),
        "trades": int((result["Position"] != 0).sum()),
        "hit_rate": float((result.loc[result["Position"] != 0, "Return"] > 0).mean()),
    }
    return result, metrics
