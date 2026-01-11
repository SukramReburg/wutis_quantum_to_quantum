# wutis_quantum_on_quantum

**WUTIS Semester Project WS 25/26**
Markowitz Portfolio Optimization using Quantum Neuronal Networks and Quantum Annealer.

---
## How to Use

### Install packages 
To start with the project, run `install.sh` script. Python version used: 3.11.14

### Fetch and Preprocess Data
Run the following scripts in sequence to fetch and preprocess the data:

1. **`fetch.py`**  
    Fetch historical market data from external APIs.
2. **`preprocess.py`**  
    Add indicators defined in `indicators.py` and merge data into `raw/merged_data.csv `.
3. **`datasets.py`**  
    Create datasets for covariance and returns prediction and save in .npz under `processed/qnn_datasets.npz`.

### QNN Training and Tuning 

1. **`encode.py`**  
    Encode the datasets into unitary QNN (data) layer. 
2. **`model.py`**  
    QNN model and quantum circuit definition
3. **`train.py`** 
    Train the model with predefined parameters
4. **`tuning.py`**  
    Tune model's hyperparameters with Optuna

### Alpaca API Configuration
Alpaca API is needed to fetch stock data from the market. 
To use the Alpaca API for historical data, define a `config.yaml` file in the `config/` directory as follows:

```yaml
alpaca_api: 
  secret_key: 'your_secret_key'
  api_key: 'your_api_key'
  base_url: 'https://paper-api.alpaca.markets' # Example URL
```

### Project Structure: 
```
wutis-quantum/
├── analysis/             # Data visualizations, plots,...
├── config/               # Configuration files
├── data/                 # Data methods
├── qnn/                  # QNN training, tuning
├── source/               # Project sources: presentation, documentations etc.
└── README.md             # Project documentation
```

## Metrics Logging and Reports

This part records training diagnostics so you can interpret QNN behavior and compare runs. The logger lives in `qnn/qnn_metrics_logger.py` and is used directly by `qnn/train.py`, so metrics are collected from the exact forward pass used for the loss.

How it works:
- At the start of each epoch, accumulators reset.
- For every batch, it logs gradient norms after `loss.backward()` and QNN output stats from the same batch.
- At the end of each epoch, it aggregates per-epoch values, records train/val loss, computes parameter update norms (current vs. previous epoch), and stores the full theta trajectory.
- After training, it runs a one-time sensitivity check by perturbing inputs and measuring prediction change.

Metrics recorded (grouped):
- Loss: train/val MSE per epoch.
- Optimization: quantum/classical gradient norms, update norms, and the update balance ratio.
- Model signals: QNN output mean/variance/min/max; theta trajectory per epoch.
- Stability: sensitivity score from noisy input perturbations.

Outputs (when `save_artifacts=True` and `save_debug_metrics` is not disabled):
- Metrics files: `qnn/results/metrics/latest/_tmp_metrics_latest_debug.npz` (latest) and `qnn/results/metrics/archive/_tmp_metrics_<run_tag>.npz` (archived).
- Reports: `qnn/results/reports/latest/_tmp_metrics_latest_debug.pdf` and `qnn/results/reports/archive/_tmp_metrics_<run_tag>.pdf` generated automatically.
- Run comparison log: `qnn/results/metrics/metrics_runs.jsonl` (one JSON line per run).

Note: QNN output stats are collected via an extra `model.quantum(xb)` forward under `no_grad`. This is fine for deterministic backends, but if you switch to stochastic/finite-shot backends, update logging to capture outputs from the training forward pass to preserve strict reproducibility.
