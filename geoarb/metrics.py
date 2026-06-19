"""
metrics.py
==========

Performance and risk metrics for the arbitrage simulator output.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def performance_summary(equity: pd.DataFrame, trades: pd.DataFrame,
                        starting_capital: float,
                        periods_per_year: int = 252) -> dict:
    """Compute headline performance statistics from the equity curve & trades."""
    eq = equity["equity"].to_numpy()
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]

    total_return = eq[-1] / starting_capital - 1.0
    n_days = len(eq)
    years = n_days / periods_per_year
    cagr = (eq[-1] / starting_capital) ** (1 / years) - 1.0 if years > 0 else np.nan

    vol = rets.std(ddof=1) * np.sqrt(periods_per_year) if len(rets) > 1 else np.nan
    mean_ann = rets.mean() * periods_per_year if len(rets) else np.nan
    sharpe = mean_ann / vol if vol and vol > 0 else np.nan

    # max drawdown
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1.0
    max_dd = dd.min() if len(dd) else np.nan

    # trade stats
    if trades is not None and not trades.empty:
        wins = trades["pnl"] > 0
        win_rate = wins.mean()
        avg_win = trades.loc[wins, "pnl"].mean() if wins.any() else 0.0
        avg_loss = trades.loc[~wins, "pnl"].mean() if (~wins).any() else 0.0
        n_trades = len(trades)
        gross_profit = trades.loc[wins, "pnl"].sum()
        gross_loss = -trades.loc[~wins, "pnl"].sum()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        total_pnl = trades["pnl"].sum()
    else:
        win_rate = avg_win = avg_loss = n_trades = profit_factor = total_pnl = np.nan

    return {
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "ann_vol_pct": vol * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "n_trades": n_trades,
        "win_rate_pct": win_rate * 100 if np.isfinite(win_rate) else np.nan,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "profit_factor": profit_factor,
        "total_trade_pnl_usd": total_pnl,
        "final_equity_usd": eq[-1],
    }


def cointegration_check(prices: pd.DataFrame, code_a: str, code_b: str) -> dict:
    """Quick Engle-Granger style check that the spread is stationary.

    Returns the estimated AR(1) coefficient of the spread and an approximate
    half-life of mean reversion. (Uses OLS; no statsmodels dependency.)
    """
    s = (prices[code_a] - prices[code_b]).to_numpy()
    s = s - s.mean()
    y, x = s[1:], s[:-1]
    beta = float(np.dot(x, y) / np.dot(x, x))     # AR(1) coefficient
    half_life = np.log(0.5) / np.log(abs(beta)) if 0 < abs(beta) < 1 else np.inf
    return {
        "pair": f"{code_a}-{code_b}",
        "ar1_coef": beta,
        "spread_std": float(np.std(s, ddof=1)),
        "half_life_days": float(half_life),
        "stationary_like": bool(abs(beta) < 0.99),
    }
