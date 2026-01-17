import numpy as np

TRADING_DAYS = 252

def _to_numpy_1d(r):
    r = np.asarray(r, dtype=float).reshape(-1)
    r = r[~np.isnan(r)]
    return r

def annualized_return(r):
    r = _to_numpy_1d(r)
    if r.size == 0:
        return np.nan
    return np.exp(r.mean() * TRADING_DAYS) - 1

def annualized_vol(r, ddof=1):
    r = _to_numpy_1d(r)
    if r.size < 2:
        return np.nan
    return r.std(ddof=ddof) * np.sqrt(TRADING_DAYS)

def sharpe(r, rf_annual=0.0, ddof=1):
    r = _to_numpy_1d(r)
    if r.size < 2:
        return np.nan
    rf_daily_log = np.log1p(rf_annual) / TRADING_DAYS  # consistent w/ log returns
    excess = r - rf_daily_log
    vol = excess.std(ddof=ddof)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return excess.mean() / vol * np.sqrt(TRADING_DAYS)

def equity_curve(r):
    r = _to_numpy_1d(r)
    if r.size == 0:
        return np.array([])
    return np.exp(np.cumsum(r))

def cumulative_return(r):
    eq = equity_curve(r)
    if eq.size == 0:
        return np.array([])
    return eq - 1

def total_return(r):
    r = _to_numpy_1d(r)
    if r.size == 0:
        return np.nan
    return np.exp(r.sum()) - 1

def drawdown(r):
    eq = equity_curve(r)
    if eq.size == 0:
        return np.array([])
    running_max = np.maximum.accumulate(eq)
    return eq / running_max - 1

def max_drawdown(r):
    dd = drawdown(r)
    if dd.size == 0:
        return np.nan
    return dd.min()

def performance_summary(r):
    return {
        "Total Return": total_return(r),
        "Ann Return": annualized_return(r),
        "Ann Vol": annualized_vol(r),
        "Sharpe": sharpe(r),
        "Max DD": max_drawdown(r),
    }
