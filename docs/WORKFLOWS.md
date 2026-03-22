# Project Workflows And Functionalities

This document is the operational guide for the repository. It complements the main
README and focuses on:

- which pipelines exist
- how to run the project from raw data to analysis outputs
- which artifacts each stage produces
- which classes and methods are important when extending the code

## 1. Prerequisites

Run everything from the repository root.

Create the virtual environment and install dependencies:

```bash
bash install.sh
source .venv/bin/activate
```

Required config files:

- `config/config.yaml`: Alpaca API credentials
- `config/data_config.yaml`: data-engineering settings and paths
- `config/model_config.yaml`: QNN, tuning, optimizer, and analysis settings

Before fetching data, make sure the asset universe is defined in:

```text
source/assets.csv
```

## 2. Full End-To-End Workflow

The usual weekly workflow is:

1. fetch raw OHLCV market data
2. preprocess and merge all assets into one feature table
3. build daily and/or weekly supervised datasets
4. train QNN models for returns and covariance
5. optionally tune hyperparameters with Optuna
6. run the optimizer/backtest on the latest prediction bundles
7. generate analysis plots and reports

Recommended command sequence:

```bash
bash install.sh
source .venv/bin/activate

python data/data_pipeline.py --mode both
python qnn/train.py --mode both
python optimizer/backtest.py
python analysis/run_analysis.py
```

Main outputs after a full run:

- datasets: `data/processed/`
- scalers: `data/processed/scalers/`
- model outputs: `qnn/results/`
- plots: `analysis/plots/`
- reports: `analysis/reports/`

## 3. Pipeline Variants

### 3.1 Data-only pipeline

Use this when you want to refresh market data and regenerate model-ready datasets.

```bash
python data/data_pipeline.py --mode both
```

Or run the steps separately:

```bash
python data/fetch.py
python data/preprocess.py
python data/datasets.py
python data/datasets_weekly.py
```

Use `--mode daily`, `--mode weekly`, or `--mode both` with:

```bash
python data/data_pipeline.py --mode weekly
```

### 3.2 Train-only pipeline

Use this when datasets already exist and you want to train fresh models.

```bash
python qnn/train.py --mode both
```

Single-target runs:

```bash
python qnn/train.py --mode returns
python qnn/train.py --mode cov
```

Runtime override example:

```bash
python qnn/train.py --mode returns --runtime-profile fast_exact
```

### 3.3 Tune-only pipeline

Use this to search over the config-defined Optuna search space.

```bash
python qnn/tuning.py --mode returns
python qnn/tuning.py --mode cov
```

Custom study example:

```bash
python qnn/tuning.py --mode returns --study-name returns_realism --n-trials 30
```

### 3.4 Optimizer-only pipeline

Use this when predictions already exist and you only want portfolio simulation.

```bash
python optimizer/backtest.py
```

Or point to explicit prediction bundles:

```bash
python optimizer/backtest.py \
  --returns-pred qnn/results/latest/returns/predictions.npz \
  --cov-pred qnn/results/latest/cov/predictions.npz
```

### 3.5 Analysis-only pipeline

Generate market-data, model, and optimizer diagnostics without retraining.

```bash
python analysis/run_analysis.py
```

Stage-specific analysis:

```bash
python analysis/data_report.py
python analysis/model_report.py --mode both
python optimizer/backtest.py
```

### 3.6 Monitoring long training runs

`python qnn/train.py --mode both` trains the `returns` model first and then the
`cov` model. The trainer now prints:

- a run header
- one line per epoch
- an early-stopping message when triggered
- a completion line with final metrics and artifact location

Useful checks while training:

```bash
tail -f logs/train_latest.log
find qnn/results/latest -type f | tail
find analysis/plots/models -type f | tail
```

On macOS, CPU activity can be checked with:

```bash
top -o cpu -stats pid,command,cpu,threads,time
```

## 4. Stage-By-Stage Details

### 4.1 Data aggregation and preprocessing

Important files:

- `data/fetch.py`
- `data/preprocess.py`
- `data/datasets.py`
- `data/datasets_weekly.py`
- `data/data_pipeline.py`

Important methods:

- `data.fetch.compute_fetch_end_datetime()`
  Computes the fetch end timestamp safely with `timedelta`, including month boundaries.
- `data.fetch.load_alpaca_client()`
  Loads the Alpaca client from `config/config.yaml`.
- `data.fetch.fetch_and_save_data()`
  Downloads raw ticker CSVs into a temporary run directory and atomically replaces `data/raw/tickers/`.
- `data.preprocess.merge_dataframes()`
  Merges per-ticker frames into one aligned dataset.
- `data.preprocess.preprocess_and_save_data()`
  Computes indicators, sorts assets deterministically, and writes `data/raw/merged_data.csv`.
- `data.datasets.save_dataset_bundle_from_config()`
  Builds and saves the daily dataset bundle.
- `data.datasets_weekly.save_dataset_bundle_from_config()`
  Builds and saves the weekly dataset bundle.

Key artifacts:

- `data/raw/tickers/*.csv`
- `data/raw/merged_data.csv`
- `data/processed/qnn_datasets_daily.npz`
- `data/processed/qnn_datasets_weekly.npz`
- `data/processed/scalers/*.joblib`

### 4.2 QNN configuration and training

Important files:

- `qnn/config.py`
- `qnn/runtime.py`
- `qnn/model.py`
- `qnn/losses.py`
- `qnn/trainer.py`
- `qnn/train.py`

Important methods and classes:

- `qnn.config.load_experiment_config(mode=...)`
  Loads `config/model_config.yaml`, applies per-mode overrides, and returns a typed experiment config.
- `qnn.runtime.QuantumRuntimeFactory`
  Selects the runtime profile and simulator configuration.
- `qnn.model.QNNBuilder`
  Builds the quantum model specification and circuit stack.
- `qnn.losses.LossFactory`
  Creates the configured regression loss such as `huber`, `mse`, `mae`, or `log_cosh`.
- `qnn.trainer.QNNTrainer.load_dataset()`
  Loads the processed `.npz` bundle, metadata, and split arrays for the selected mode.
- `qnn.trainer.QNNTrainer.train()`
  Runs the full training loop, evaluation, artifact writing, and optional plotting.
- `qnn.trainer.train_qnn_from_npz()`
  Convenience entrypoint for training from dataset artifacts.

Training behavior:

- target normalization can be enabled in config
- early stopping and scheduler behavior are config-driven
- repeated validation passes can estimate prediction uncertainty
- training writes both archived and `latest` outputs
- training prints per-epoch progress to the terminal so long runs are observable

Prediction behavior:

- training currently produces the prediction bundles
- the main output file is `predictions.npz`
- prediction bundles contain metadata such as `asset_symbols`, `sample_dates`, `runtime_profile`, `loss_name`, and `best_epoch`

Key artifacts:

- `qnn/results/latest/returns/predictions.npz`
- `qnn/results/latest/cov/predictions.npz`
- `qnn/results/latest/<mode>/metrics.npz`
- `qnn/results/latest/<mode>/summary.json`
- `qnn/results/latest/<mode>/model_state.pth`

### 4.3 Hyperparameter tuning

Important files:

- `qnn/study.py`
- `qnn/tuning.py`

Important methods and classes:

- `qnn.study.QNNStudyRunner`
  Central Optuna orchestrator.
- `qnn.study.QNNStudyRunner.create_objective()`
  Builds the training objective from config-defined search spaces.
- `qnn.study.QNNStudyRunner.run()`
  Creates or resumes a study, runs trials, and saves study artifacts.

Tuning outputs:

- `qnn/results/optuna/<study_name>/trials.csv`
- `qnn/results/optuna/<study_name>/best_params.json`
- `analysis/plots/tuning/<study_name>/latest/`

### 4.4 Portfolio optimization and backtesting

Important files:

- `optimizer/bundle.py`
- `optimizer/portfolio.py`
- `optimizer/engine.py`
- `optimizer/reporting.py`
- `optimizer/backtest.py`

Important methods and classes:

- `optimizer.bundle.PredictionBundle.load()`
  Loads prediction artifacts and validates shape, asset order, dates, and frequency.
- `optimizer.portfolio.PortfolioOptimizer.optimize()`
  Solves the configured portfolio objective.
- `optimizer.engine.BacktestEngine.build_holding_windows()`
  Defines half-open holding windows `(t, t+1]` to avoid double-counting rebalance dates.
- `optimizer.engine.BacktestEngine.run()`
  Loads predictions, aligns them with market data, computes weights, simulates portfolio returns, and produces diagnostics.
- `optimizer.reporting.BacktestReporter.save()`
  Writes optimizer plots and reports into `analysis/`.

Backtest outputs:

- `analysis/plots/optimizer/latest/`
- `analysis/reports/optimizer/latest/summary.json`
- `analysis/reports/optimizer/latest/weights.csv`
- `analysis/reports/optimizer/latest/diagnostics.csv`

### 4.5 Analysis and visualization

Important files:

- `analysis/common.py`
- `analysis/data_report.py`
- `analysis/model_report.py`
- `analysis/run_analysis.py`

Important methods and classes:

- `analysis.common.AnalysisPathManager`
  Central resolver for analysis plot and report locations.
- `analysis.data_report.MarketDataInvestigator.save_report()`
  Generates data-quality and exploratory visualizations from `merged_data.csv`.
- `analysis.model_report.ModelPerformanceInvestigator.save_report()`
  Generates QNN prediction diagnostics from the latest model outputs.
- `qnn.plotting.QNNPlotter.save_all()`
  Writes learning curves, residual plots, per-asset diagnostics, and uncertainty plots for trained models.

Typical analysis plots:

- covariance and correlation heatmaps
- return distribution histograms
- cumulative return paths
- rolling volatility
- learning curves
- per-loss curves
- gradient and parameter diagnostics
- predicted-vs-true plots
- rolling error diagnostics
- covariance reconstruction error
- backtest equity, drawdown, turnover, and risk diagnostics

## 5. Recommended Workflows

### Research workflow

Use this for experimentation:

```bash
python data/data_pipeline.py --mode weekly
python qnn/tuning.py --mode returns --n-trials 20
python qnn/tuning.py --mode cov --n-trials 20
python qnn/train.py --mode both
python optimizer/backtest.py
python analysis/run_analysis.py
```

### Fast development workflow

Use this when iterating on code and you do not want the heaviest simulator profile.

```bash
python qnn/train.py --mode returns --runtime-profile fast_exact --no-plots
python qnn/train.py --mode cov --runtime-profile fast_exact --no-plots
python analysis/model_report.py --mode both
```

### Rebuild analysis after existing runs

Use this when data, predictions, or backtest outputs already exist.

```bash
python optimizer/backtest.py
python analysis/run_analysis.py
```

### Overnight training workflow

Use this when you want the laptop to stay awake, consume most CPU resources, and
write a log file you can inspect later.

1. Update `config/model_config.yaml` with a CPU-heavy setup:

```yaml
runtime:
  device: CPU
  max_parallel_threads: 10
  max_parallel_experiments: 1
  max_parallel_shots: 0

resources:
  prefer_cuda: false
  torch_threads: 10
  dataloader_workers: 2
  pin_memory: false
```

2. Start training with `caffeinate` and capture the output:

```bash
mkdir -p logs
caffeinate -dimsu env MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 \
  .venv/bin/python qnn/train.py --mode both 2>&1 | tee logs/train_latest.log
```

Notes:

- `caffeinate` prevents the Mac from sleeping
- `OMP_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` help CPU-heavy math libraries use the available cores
- `logs/train_latest.log` lets you inspect progress after the run starts
- on a laptop, running multiple training processes at once is usually worse than one well-threaded process

### Device availability check

Before trying to force GPU usage, check what the environment actually exposes:

```bash
.venv/bin/python - <<'PY'
import torch
from qiskit_aer import AerSimulator

print("torch_cuda", torch.cuda.is_available())
print(
    "torch_mps_available",
    getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
)
print("aer_available_devices", list(AerSimulator().available_devices()))
PY
```

Interpretation:

- if Aer reports only `['CPU']`, the quantum part of training is CPU-only
- if `torch.cuda` is `False`, the classical PyTorch head is also CPU-only
- on many Macs, Qiskit Aer exposes CPU only, so changing GPU config values does not create GPU acceleration

## 6. Output Map

Use this as a quick reference:

- raw fetched data: `data/raw/tickers/`
- merged market table: `data/raw/merged_data.csv`
- processed model datasets: `data/processed/`
- QNN saved runs: `qnn/results/`
- tuning studies: `qnn/results/optuna/`
- analysis plots: `analysis/plots/`
- analysis reports: `analysis/reports/`

## 7. Common Notes

- Weekly datasets are the default training and backtest frequency.
- Training is also the current prediction step; there is no separate `predict.py` yet.
- If you want only one stage, run that stage directly instead of the full pipeline.
- If a stage cannot find its input artifacts, regenerate the upstream stage first.
- Resource knobs for training live in `config/model_config.yaml` under `runtime` and `resources`.
