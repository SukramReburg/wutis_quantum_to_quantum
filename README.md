# wutis_quantum_to_quantum

WUTIS semester project for WS 25/26.

The repository combines:

- market-data ingestion and feature engineering
- Qiskit + PyTorch quantum neural network training and tuning
- Markowitz-style portfolio optimization and backtesting
- centralized analysis outputs for data, model, and portfolio investigation

## Project Goal

The project forecasts:

- next-period returns
- next-period covariance structure

Those forecasts are then fed into a long-only optimizer and evaluated in a weekly rebalancing backtest.

## Repository Layout

```text
wutis_quantum_to_quantum/
├── README.md
├── docs/
│   └── WORKFLOWS.md
├── install.sh
├── requirements.txt
├── config/
│   ├── config.yaml
│   ├── data_config.yaml
│   └── model_config.yaml
├── source/
│   └── assets.csv
├── data/
│   ├── fetch.py
│   ├── preprocess.py
│   ├── datasets.py
│   ├── datasets_weekly.py
│   ├── data_pipeline.py
│   ├── raw/
│   └── processed/
├── qnn/
│   ├── config.py
│   ├── runtime.py
│   ├── encode.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── artifacts.py
│   ├── trainer.py
│   ├── study.py
│   ├── train.py
│   ├── tuning.py
│   └── results/
├── analysis/
│   ├── common.py
│   ├── data_report.py
│   ├── model_report.py
│   ├── run_analysis.py
│   ├── cov_plot.py
│   ├── test_plot.py
│   ├── plots/
│   └── reports/
└── optimizer/
    ├── bundle.py
    ├── portfolio.py
    ├── engine.py
    ├── reporting.py
    ├── backtest.py
    ├── reconstruct_cov.py
    └── KPIs.py
```

## Environment Setup

The project is designed around a local virtual environment.

Create and populate the venv with:

```bash
bash install.sh
```

`install.sh` now:

1. creates `.venv/`
2. upgrades `pip`, `setuptools`, and `wheel`
3. installs `requirements.txt`
4. verifies the core imports used by the pipeline

Activate it with:

```bash
source .venv/bin/activate
```

The install step verifies:

- `qiskit`
- `qiskit-aer`
- `qiskit-machine-learning`
- `torch`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `PyYAML`

## Additional Documentation

For a command-oriented guide to the whole project, see:

- [docs/WORKFLOWS.md](docs/WORKFLOWS.md)

That document covers:

- full end-to-end workflows
- data-only, train-only, tuning-only, optimizer-only, and analysis-only pipelines
- important classes and methods behind each stage
- main output artifacts and where to find them
- overnight training, resource allocation, and run monitoring

## Configuration

All config files live under `config/`.

### `config/config.yaml`

Required for Alpaca downloads.

```yaml
alpaca_api:
  secret_key: "your_secret_key"
  api_key: "your_api_key"
  base_url: "https://paper-api.alpaca.markets"
```

### `config/data_config.yaml`

Controls:

- asset universe
- indicator generation
- daily vs weekly dataset settings
- data directories

Important keys:

- `assets`
- `start_year`
- `records_number_threshold`
- `indicators`
- `train_size`
- `lookback_window`
- `cov_window`
- `use_past_cov_in_features`
- `use_past_ret_in_features`
- `paths.raw`
- `paths.processed`
- `paths.scalers`
- `paths.plots`

### `config/model_config.yaml`

Controls:

- QNN runtime profile and simulator options
- noise-model options
- resource settings
- model architecture
- training loss / scheduler / early stopping
- Optuna search spaces
- optimizer objective and constraints
- analysis output roots

Key paths:

- `paths.results`: saved predictions, metrics, summaries, and model states
- `paths.plots`: centralized plot root, now `analysis/plots`
- `paths.reports`: centralized JSON/CSV report root, now `analysis/reports`
- `paths.optuna`: study tables and best-parameter payloads

## End-to-End Pipeline

### 1. Define the asset universe

Edit:

```text
source/assets.csv
```

### 2. Fetch raw market data

```bash
python data/fetch.py
```

Output:

- `data/raw/tickers/*.csv`

### 3. Preprocess and engineer features

```bash
python data/preprocess.py
```

Output:

- `data/raw/merged_data.csv`

### 4. Build supervised datasets

Daily:

```bash
python data/datasets.py
```

Weekly:

```bash
python data/datasets_weekly.py
```

Or orchestrate all steps:

```bash
python data/data_pipeline.py --mode both
```

Outputs:

- `data/processed/qnn_datasets_daily.npz`
- `data/processed/qnn_datasets_weekly.npz`
- `data/processed/scalers/*.joblib`

## QNN Pipeline

### Train

Train both default experiments:

```bash
python qnn/train.py
```

Train one mode only:

```bash
python qnn/train.py --mode returns
python qnn/train.py --mode cov
```

Override runtime profile:

```bash
python qnn/train.py --mode returns --runtime-profile fast_exact
```

What training writes:

- predictions, metrics, summaries, model weights under `qnn/results/`
- plots under `analysis/plots/models/<mode>/...`

The trainer is config-driven through `config/model_config.yaml`.

Default loss behavior:

- `huber` is the default for returns and covariance
- `mse`, `mae`, `smooth_l1`/`huber`, and `log_cosh` are supported

Default simulator behavior:

- runtime profile is `realism`
- Aer-style finite-shot execution is preferred when available
- `balanced` and `fast_exact` remain available for development and ablation

### Tune

Run Optuna:

```bash
python qnn/tuning.py --mode returns
python qnn/tuning.py --mode cov
```

Outputs:

- study tables and best params under `qnn/results/optuna/<study_name>/`
- tuning plots under `analysis/plots/tuning/<study_name>/latest/`

### Predict

The current workflow treats training output as the prediction stage. A completed training run writes:

- `predictions.npz`
- `metrics.npz`
- `summary.json`

under:

- `qnn/results/latest/returns/`
- `qnn/results/latest/cov/`

Each prediction artifact includes metadata such as:

- `asset_symbols`
- `sample_dates`
- `target_frequency`
- `runtime_profile`
- `loss_name`
- `best_epoch`

## Optimizer And Backtest Pipeline

Run the weekly backtest against the latest prediction bundles:

```bash
python optimizer/backtest.py
```

Or point to specific prediction artifacts:

```bash
python optimizer/backtest.py \
  --returns-pred qnn/results/latest/returns/predictions.npz \
  --cov-pred qnn/results/latest/cov/predictions.npz
```

The backtest now:

- reads prediction metadata directly from `.npz` bundles
- rebuilds covariance matrices from predicted upper-triangle vectors
- applies a mean-variance optimizer that actually uses both `mu` and `Sigma`
- uses half-open holding windows `(t, t+1]` to avoid double-counting rebalance dates

Outputs:

- plots under `analysis/plots/optimizer/latest/`
- summary files under `analysis/reports/optimizer/latest/`

## Analysis Pipeline

All plots are now centralized under `analysis/plots/`. The analysis folder is the single home for visual diagnostics.

### Market-data investigation

```bash
python analysis/data_report.py
```

Outputs include:

- covariance heatmap
- correlation heatmap
- return distribution histogram
- cumulative return paths
- rolling volatility chart
- feature availability plot

### Model-performance investigation

```bash
python analysis/model_report.py --mode both
```

Outputs include:

- learning curves
- RMSE / MAE curves
- gradient and update norms
- quantum-output diagnostics
- theta heatmap
- residual histogram
- predicted-vs-true scatter
- rolling error
- uncertainty plot
- per-asset error and sign accuracy
- covariance sample heatmaps and Frobenius error time series

### Run all analysis reports

```bash
python analysis/run_analysis.py
```

Backward-compatible wrappers still exist:

```bash
python analysis/cov_plot.py
python analysis/test_plot.py
```

## Plot And Report Locations

### Plots

- data EDA: `analysis/plots/data/latest/`
- model diagnostics: `analysis/plots/models/<mode>/latest/`
- tuning diagnostics: `analysis/plots/tuning/<study_name>/latest/`
- optimizer/backtest diagnostics: `analysis/plots/optimizer/latest/`

### Reports

- data summaries: `analysis/reports/data/latest/`
- model summaries: `analysis/reports/models/<mode>/latest/`
- optimizer summaries: `analysis/reports/optimizer/latest/`

### Non-plot Artifacts

- datasets: `data/processed/`
- scalers: `data/processed/scalers/`
- trained QNN outputs: `qnn/results/`
- Optuna tables / best params: `qnn/results/optuna/`

## Code Structure Notes

The codebase is still script-oriented, but the main runtime logic is now split into reusable services:

- `qnn/trainer.py`: QNN training loop, normalization, early stopping, evaluation, and artifact writing
- `qnn/study.py`: Optuna orchestration from config-defined search spaces
- `optimizer/engine.py`: prediction loading, weekly alignment, and portfolio simulation
- `optimizer/reporting.py`: optimizer diagnostics and plot generation
- `analysis/data_report.py`: data quality and exploratory analysis
- `analysis/model_report.py`: prediction and model diagnostics

Main functions and classes are intentionally short and commented so the flow is easier to follow than the previous script-only implementation.

## Current Assumptions

- weekly datasets remain the default for training and backtesting
- `config/config.yaml`, `config/data_config.yaml`, and `config/model_config.yaml` are the only supported config locations
- commands should be run from the repository root

## Tests

The repository now includes:

- data pipeline contract tests
- QNN / optimizer stack tests

Run them with your environment active:

```bash
python -m unittest discover tests
```
