import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
merged_path = os.path.join(base_dir, 'data', 'raw', 'merged_data.csv')
assets_path = os.path.join(base_dir, 'source', 'assets.csv')

merged_df = pd.read_csv(merged_path, index_col='timestamp', parse_dates=True)
assets_df = pd.read_csv(assets_path)
tickers = sorted(assets_df['ticker'].tolist())

print(f"Loaded {len(tickers)} tickers: {tickers}")
print(f"Merged data shape: {merged_df.shape}")
print(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")

# Create 4 subplots in 2x2 grid: RSI, Volatility, Log Return Densities, Volume Change
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

# Plot RSI for all tickers
for ticker in tickers:
    col = f"{ticker}_rsi"
    if col in merged_df.columns:
        ax1.plot(merged_df.index, merged_df[col], label=ticker, alpha=0.7, linewidth=1)

ax1.set_title("RSI (Relative Strength Index) - All Tickers", fontsize=14, fontweight='bold')
ax1.set_ylabel("RSI")
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=70, color='r', linestyle='--', alpha=0.3, label='Overbought (70)')
ax1.axhline(y=30, color='g', linestyle='--', alpha=0.3, label='Oversold (30)')

# Plot Volatility for all tickers
for ticker in tickers:
    col = f"{ticker}_vola"
    if col in merged_df.columns:
        ax2.plot(merged_df.index, merged_df[col], label=ticker, alpha=0.7, linewidth=1)

ax2.set_title("Volatility (Rolling Std) - All Tickers", fontsize=14, fontweight='bold')
ax2.set_ylabel("Volatility")
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot Log Return densities for all tickers (histograms + KDE when available)
log_cols = [f"{ticker}_log" for ticker in tickers if f"{ticker}_log" in merged_df.columns]
if not log_cols:
    ax3.text(0.5, 0.5, 'No log-return columns found', ha='center', va='center')
else:
    colors = plt.cm.tab20.colors
    for i, col in enumerate(log_cols):
        series = merged_df[col].dropna()
        if series.empty:
            continue
        color = colors[i % len(colors)]
        # transparent histogram for density
        ax3.hist(series.values, bins=100, density=True, alpha=0.25, color=color)
        # KDE if scipy available
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(series.values)
            xs = np.linspace(series.min(), series.max(), 200)
            ax3.plot(xs, kde(xs), color=color, linewidth=1.2, label=col.split('_')[0])
        except Exception:
            pass

    ax3.set_title("Log Return Densities - All Tickers", fontsize=14, fontweight='bold')
    ax3.set_xlabel('Log Return')
    ax3.set_ylabel('Density')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

# Plot Volume Change for all tickers
for ticker in tickers:
    col = f"{ticker}_volume_change"
    if col in merged_df.columns:
        ax4.plot(merged_df.index, merged_df[col], label=ticker, alpha=0.7, linewidth=1)

ax4.set_title("Volume Change - All Tickers", fontsize=14, fontweight='bold')
ax4.set_ylabel("Volume Change")
ax4.set_xlabel("Date")
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(base_dir, 'analysis', 'plots', 'sanity_check_density.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Plot saved to: {out_path}")
plt.show()
