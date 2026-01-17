import pandas as pd
import numpy as np
from reconstruct_cov import rebuild_covariance, make_psd
from optimizer import optimize_weights
from KPIs import *

### NOTE: ALL IN LOG RETURNS ### 

# ============================================================
# Initial Set up - raw daily data (log)
# ============================================================

data = pd.read_csv("data/raw/merged_data.csv", index_col="timestamp", parse_dates=True)

# select daily log returns
ret_cols = [c for c in data.columns if c.endswith("_log")]
log_ret_daily = data[ret_cols].dropna()

# Get correct order of the symbols - ensure sorted like qnn
symbols_all = sorted({c.split('_')[0] for c in ret_cols})

# weekly rebalance calendar
weekly_ends = log_ret_daily.resample("W-FRI").last().index



# ============================================================
# Rebuild ALL covariance matrices - from predictions
# ============================================================
cov_preds = np.load("qnn_cov_pca_hybrid_rxrz_predictions.npz") # update

# Shape: (T, n_assets*(n_assets+1)//2)
Y_cov = cov_preds["Y_pred_test"]

n_assets = len(symbols_all)

Sigma_psd_all = []

for t in range(Y_cov.shape[0]):
    C_raw = rebuild_covariance(Y_cov[t], n_assets)
    Sigma_psd = make_psd(C_raw)
    Sigma_psd_all.append(Sigma_psd)

Sigma_psd_all = np.stack(Sigma_psd_all)



# ============================================================
# Sanity checks on covariance matrix
# ============================================================

t = 0
assert np.allclose(Sigma_psd_all[t], Sigma_psd_all[t].T, atol=1e-8)
assert np.min(np.linalg.eigvalsh(Sigma_psd_all[t])) >= 0
print("Covariance matrix is symmetric and PSD.")

print("Len Symbols: ", len(symbols_all))


# ============================================================
# Simple log returns from predictions
# ============================================================
data = np.load("qnn_returns_angles_hybrid_rxrz_predictions.npz") # update

# extract test
mu_log = data["Y_pred_test"]

# simple returns
mu = np.expm1(mu_log)


# ============================================================
# Align weekly data 
# ============================================================

log_ret_weekly = log_ret_daily.resample("W-FRI").sum()

# number of weekly predictions
T = mu.shape[0]  

# dates corresponding to weekly rebalancing
weekly_dates = log_ret_weekly.index[-T:]

# Safety check
assert len(weekly_dates) == mu.shape[0] == Sigma_psd_all.shape[0]

print("Weekly prediction window:")
print(weekly_dates[0], "->", weekly_dates[-1])


# wrap predictions in labelled container

# Expected WEEKLY log returns
mu_log_df = pd.DataFrame(
    mu_log,
    index=weekly_dates,
    columns=symbols_all
)

# Expected WEEKLY simple returns
mu_df = pd.DataFrame(
    mu,
    index=weekly_dates,
    columns=symbols_all
)

# Covariance matrices indexed by week
Sigma_weekly_daily_cov = {
    date: Sigma_psd_all[i]
    for i, date in enumerate(weekly_dates)
}

print("Alignment complete:")
print("mu_df shape:", mu_df.shape)
print("Sigma_weekly entries:", len(Sigma_weekly_daily_cov))


# ============================================================
# BACKTEST: weekly rebalance, DAILY PnL (NORMAL RETURNS)
# ============================================================

# Prepare normal returns
ret_daily = np.expm1(log_ret_daily)   # normal daily returns
ret_daily = ret_daily[symbols_all]    # enforce column order

# Prepare dictionaries to store results
portfolio_simple_returns = []
portfolio_dates = []
weights_history = {}


# Weekly rebalance loop
for i, rebalance_date in enumerate(weekly_dates):

    # mu_df is WEEKLY expected SIMPLE returns
    mu_t = mu_df.loc[rebalance_date].values

    # Covariance of DAILY returns inside the coming week
    Sigma_t = Sigma_weekly_daily_cov[rebalance_date]

    # Calculate weights for the week
    weights = optimize_weights(mu_t, Sigma_t, u=0.1)
    weights_history[rebalance_date] = weights

    # Determine DAILY holding period for this week
    if i < len(weekly_dates) - 1:
        start = rebalance_date
        end = weekly_dates[i + 1]
        daily_slice = ret_daily.loc[start:end]
    else:
        daily_slice = ret_daily.loc[rebalance_date:]

    # Apply weights to DAILY NORMAL RETURNS
    for date, row in daily_slice.iterrows():
        daily_portfolio_ret = np.dot(weights, row.values)
        portfolio_simple_returns.append(daily_portfolio_ret)
        portfolio_dates.append(date)


# Wrap up results as a pandas Series
portfolio_simple_returns = pd.Series(
    portfolio_simple_returns,
    index=pd.DatetimeIndex(portfolio_dates),
    name="portfolio_simple_return"
)


# KPI functions expect LOG RETURNS internally
portfolio_log_returns = np.log1p(portfolio_simple_returns)

# Performance summary (In normal returns)
summary = performance_summary(portfolio_log_returns.values)

print("\n===== BACKTEST PERFORMANCE =====")
for k, v in summary.items():
    print(f"{k:>15}: {v:.4f}")
