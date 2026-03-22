from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _to_numpy_1d(r):
    array = np.asarray(r, dtype=float).reshape(-1)
    return array[~np.isnan(array)]


def annualized_return(r):
    values = _to_numpy_1d(r)
    if values.size == 0:
        return np.nan
    return np.exp(values.mean() * TRADING_DAYS) - 1


def annualized_vol(r, ddof=1):
    values = _to_numpy_1d(r)
    if values.size < 2:
        return np.nan
    return values.std(ddof=ddof) * np.sqrt(TRADING_DAYS)


def sharpe(r, rf_annual=0.0, ddof=1):
    values = _to_numpy_1d(r)
    if values.size < 2:
        return np.nan
    rf_daily_log = np.log1p(rf_annual) / TRADING_DAYS
    excess = values - rf_daily_log
    vol = excess.std(ddof=ddof)
    if vol <= 0 or np.isnan(vol):
        return np.nan
    return excess.mean() / vol * np.sqrt(TRADING_DAYS)


def sortino(r, rf_annual=0.0):
    values = _to_numpy_1d(r)
    if values.size < 2:
        return np.nan
    rf_daily_log = np.log1p(rf_annual) / TRADING_DAYS
    downside = values[values < rf_daily_log] - rf_daily_log
    downside_vol = np.sqrt(np.mean(downside**2)) if downside.size else 0.0
    if downside_vol <= 0:
        return np.nan
    return (values.mean() - rf_daily_log) / downside_vol * np.sqrt(TRADING_DAYS)


def equity_curve(r):
    values = _to_numpy_1d(r)
    if values.size == 0:
        return np.array([])
    return np.exp(np.cumsum(values))


def cumulative_return(r):
    equity = equity_curve(r)
    if equity.size == 0:
        return np.array([])
    return equity - 1.0


def total_return(r):
    values = _to_numpy_1d(r)
    if values.size == 0:
        return np.nan
    return np.exp(values.sum()) - 1


def drawdown(r):
    equity = equity_curve(r)
    if equity.size == 0:
        return np.array([])
    running_max = np.maximum.accumulate(equity)
    return equity / running_max - 1.0


def max_drawdown(r):
    dd = drawdown(r)
    if dd.size == 0:
        return np.nan
    return dd.min()


def calmar_ratio(r):
    mdd = max_drawdown(r)
    if mdd >= 0 or np.isnan(mdd):
        return np.nan
    return annualized_return(r) / abs(mdd)


def rolling_sharpe(series: pd.Series, window: int = 21) -> pd.Series:
    values = series.dropna()
    return values.rolling(window).apply(lambda x: sharpe(x), raw=False)


def rolling_volatility(series: pd.Series, window: int = 21) -> pd.Series:
    values = series.dropna()
    return values.rolling(window).std() * np.sqrt(TRADING_DAYS)


def performance_summary(r):
    return {
        "Total Return": total_return(r),
        "Ann Return": annualized_return(r),
        "Ann Vol": annualized_vol(r),
        "Sharpe": sharpe(r),
        "Sortino": sortino(r),
        "Calmar": calmar_ratio(r),
        "Max DD": max_drawdown(r),
    }
