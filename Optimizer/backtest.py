import pandas as pd
import numpy as np
from reconstruct_cov import rebuild_covariance, make_psd
from optimizer import optimize_weights
from KPIs import *
import matplotlib.pyplot as plt
import os



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
cov_preds = np.load("Optimizer/prediction data/qnn_cov_angles_hybrid_rxrz_predictions.npz") 

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
data = np.load("Optimizer/prediction data/qnn_returns_angles_hybrid_rxrz_predictions.npz") # update

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
ret_daily = np.expm1(log_ret_daily)
# Rename columns: 'AMZN_log' -> 'AMZN'
ret_daily.columns = [c.replace("_log", "") for c in ret_daily.columns]
# Enforce column order
ret_daily = ret_daily[symbols_all]


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

# ============================================================
# EQUAL-WEIGHT BENCHMARK (daily, normal returns)
# ============================================================


# Define test window explicitly from QNN portfolio
test_start = weekly_dates[0]
test_end = portfolio_simple_returns.index[-1]

# Restrict daily returns to test window
ret_daily_test = ret_daily.loc[test_start:test_end]

# Equal-weight vector
equal_weights = np.ones(n_assets) / n_assets

# Apply equal weights to DAILY normal returns (test window only)
benchmark_simple_returns = ret_daily_test.dot(equal_weights)
benchmark_simple_returns.name = "equal_weight_simple_return"

# Align QNN portfolio explicitly (safety)
portfolio_simple_returns = portfolio_simple_returns.loc[
    benchmark_simple_returns.index
]

# Convert both to LOG returns for KPIs
portfolio_log_returns = np.log1p(portfolio_simple_returns)
benchmark_log_returns = np.log1p(benchmark_simple_returns)

# ============================================================
# PERFORMANCE COMPARISON (TEST WINDOW ONLY)
# ============================================================

print("\n===== PERFORMANCE COMPARISON (TEST WINDOW ONLY) =====")

print("\n--- QNN Portfolio ---")
qnn_summary = performance_summary(portfolio_log_returns.values)
for k, v in qnn_summary.items():
    print(f"{k:>15}: {v:.4f}")

print("\n--- Equal-Weight Benchmark ---")
bench_summary = performance_summary(benchmark_log_returns.values)
for k, v in bench_summary.items():
    print(f"{k:>15}: {v:.4f}")

# ------------------------------------------------------------
# Define output path
# ------------------------------------------------------------
output_dir = "Optimizer/backtest results"
os.makedirs(output_dir, exist_ok=True)

plot_path = os.path.join(output_dir, "cumulative_performance.png")

# ------------------------------------------------------------
# Build equity curves (NORMAL RETURNS)
# ------------------------------------------------------------
qnn_equity = (1 + portfolio_simple_returns).cumprod()
benchmark_equity = (1 + benchmark_simple_returns).cumprod()

# ------------------------------------------------------------
# Plot and SAVE comparison
# ------------------------------------------------------------
comparison_plot_path = os.path.join(output_dir, "cumulative_performance_comparison.png")

plt.figure(figsize=(10, 5))
plt.plot(qnn_equity, label="QNN Portfolio", linewidth=2)
plt.plot(benchmark_equity, label="Equal-Weight Benchmark", linestyle="--")

plt.title("Cumulative Portfolio Performance: QNN vs Equal-Weight")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(comparison_plot_path, dpi=200)
plt.close()

print(f"Saved performance comparison plot to: {comparison_plot_path}")


# ============================================================
# Inspect portfolio weights over time
# ============================================================

# Convert weight history dict -> DataFrame
weights_df = pd.DataFrame.from_dict(
    weights_history,
    orient="index",
    columns=symbols_all
).sort_index()

# Ensure datetime index is tz-naive (important for slicing / plotting)
weights_df.index = weights_df.index.tz_localize(None)

# Preview weights table
print("\n=== Weights table (head) ===")
print(weights_df.head())

# Inspect one rebalance allocation
date = weights_df.index[0]
print(f"\n=== Weights on {date} ===")
print(weights_df.loc[date].sort_values(ascending=False))

# Sanity checks
print("\n=== Sanity checks ===")
print("Weights sum (first 5 periods):")
print(weights_df.sum(axis=1).head())

print("\nMax weight per period (summary):")
print(weights_df.max(axis=1).describe())

# ------------------------------------------------------------
# Plot weight evolution for top assets (CLEAN + ZOOMED OUT)
# ------------------------------------------------------------
top_assets = (
    weights_df.mean()
    .sort_values(ascending=False)
    .head(5)
    .index
)

weights_plot_path = os.path.join(output_dir, "top_asset_weights.png")

plt.figure(figsize=(12, 5))  # wider & flatter = zoomed out feel

for asset in top_assets:
    plt.plot(
        weights_df[asset],
        label=asset,
        linewidth=1.5
    )

# Monthly ticks (like cumulative performance plots)
ax = plt.gca()
ax.xaxis.set_major_locator(
    plt.matplotlib.dates.MonthLocator(interval=2)
)
ax.xaxis.set_major_formatter(
    plt.matplotlib.dates.DateFormatter("%b %Y")
)

# Smaller, cleaner legend
plt.legend(
    loc="upper center",
    ncol=len(top_assets),
    frameon=False,
    fontsize=9
)

plt.title("Top 5 Asset Weights Over Time")
plt.ylabel("Weight")
plt.xlabel("Rebalance Date")

# No grid (reduces clutter)
plt.grid(False)

plt.tight_layout()
plt.savefig(weights_plot_path, dpi=200)
plt.close()

print(f"Saved weights evolution plot to: {weights_plot_path}")



# ------------------------------------------------------------
# Plot equity curve with weekly rebalancing markers (with zoom)
# ------------------------------------------------------------
comparison_plot_path = os.path.join(
    output_dir, "cumulative_performance_2025_with_rebalance.png"
)

# --- Make indices tz-naive (IMPORTANT FIX)
qnn_equity_plot = qnn_equity.copy()
benchmark_equity_plot = benchmark_equity.copy()

qnn_equity_plot.index = qnn_equity_plot.index.tz_localize(None)
benchmark_equity_plot.index = benchmark_equity_plot.index.tz_localize(None)

rebalance_dates_plot = pd.DatetimeIndex(weekly_dates).tz_localize(None)

# Define zoom window (tz-naive)
zoom_start = pd.Timestamp("2025-01-01")
zoom_end   = pd.Timestamp("2025-03-31")

plt.figure(figsize=(10, 5))

# Plot equity curves (restricted to 2025)
plt.plot(
    qnn_equity_plot.loc[zoom_start:zoom_end],
    label="QNN Portfolio",
    linewidth=2
)

plt.plot(
    benchmark_equity_plot.loc[zoom_start:zoom_end],
    label="Equal-Weight Benchmark",
    linestyle="--"
)

# Add vertical lines for weekly rebalancing
for d in rebalance_dates_plot:
    if zoom_start <= d <= zoom_end:
        plt.axvline(d, color="gray", alpha=0.7, linewidth=0.9, linestyle = "-")

plt.title("Cumulative Portfolio Performance (2025 Q1, Weekly Rebalance)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(False)
plt.tight_layout()

plt.savefig(comparison_plot_path, dpi=200)
plt.close()

print(f"Saved zoomed 2025 rebalance plot to: {comparison_plot_path}")



