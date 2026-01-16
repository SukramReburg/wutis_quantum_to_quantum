# Optuna Tuning Plan (Overnight)

This document defines the locked hyperparameter search space and run settings
for the next tuning pass.

## Run Settings
- Mode: `returns`
- Trials: `50`
- Epochs per trial: `8`
- Storage: `sqlite:///qnn_optuna.db`
- Best params export: `qnn/results/optuna/optuna_best_params_<study>_<timestamp>.json`

## Search Space (Locked)

| Parameter | Range / Choices | Reasoning |
| --- | --- | --- |
| `n_qubits` | 2–8 | Explore higher-capacity circuits while still capping the search for runtime. |
| `n_layers` | 2–8 | Broader depth search to capture more expressive models. |
| `feature_mode` | `["angles", "pca"]` | Compare the two current encodings without expanding the space further. |
| `use_dense_head` | `[True]` | Required for cov under low qubits; avoids invalid output sizes for small `n_qubits`. |
| `circuit_type` | `["rxrz"]` | Faster, stable baseline; avoid the slower `zz_feature` for overnight runs. |
| `learning_rate` | `1e-4`–`2e-3` (log) | Narrowed to stable Adam learning rates for this model family. |
| `batch_size` | `[16, 32]` | Balances noise vs. speed without hitting memory limits. |
| `entanglement` | `["ring"]` | Default layout; keeps the search space tight. |

## Notes
- These ranges are tuned for **returns** with a max of **5 qubits**.
- If switching to **cov** while keeping the 5‑qubit cap, `use_dense_head` must remain `True`.
