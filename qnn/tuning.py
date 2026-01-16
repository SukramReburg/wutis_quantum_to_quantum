""" 
Includes Optuna hyperparameter tuning over:
- n_qubits, n_layers, feature_mode, use_dense_head,
  circuit_type, learning_rate, batch_size
"""
import json
import os
from datetime import datetime
from typing import Optional
import optuna
import yaml
try:
    from qnn.train import train_qnn_from_npz
except ModuleNotFoundError:
    from train import train_qnn_from_npz

# ======================================================================
# OPTUNA HYPERPARAMETER TUNING
# ======================================================================
def create_optuna_objective(
    config_path: str,
    mode: str,
    npz_name: str = "qnn_datasets.npz",
    n_epochs: int = 10,
):
    """
    Returns an Optuna objective function that tunes:
      - n_qubits, n_layers, feature_mode, use_dense_head,
        circuit_type, learning_rate, batch_size
    """
    def objective(trial: optuna.trial.Trial) -> float:
        # Hyperparameter search space
        n_qubits = trial.suggest_int("n_qubits", 2, 8)
        n_layers = trial.suggest_int("n_layers", 2, 8)
        feature_mode = trial.suggest_categorical("feature_mode", ["angles", "pca"])
        use_dense_head = trial.suggest_categorical("use_dense_head", [True])
        circuit_type = trial.suggest_categorical("circuit_type", ["rxrz"])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        entanglement = trial.suggest_categorical("entanglement", ["ring"])

        print(
            f"\n[Optuna trial {trial.number}] "
            f"n_qubits={n_qubits}, n_layers={n_layers}, "
            f"feature_mode={feature_mode}, use_dense_head={use_dense_head}, "
            f"circuit_type={circuit_type}, lr={learning_rate}, batch={batch_size}"
        )

        result = train_qnn_from_npz(
            config_path=config_path,
            mode=mode,
            n_qubits=n_qubits,
            n_layers=n_layers,
            feature_mode=feature_mode,
            use_dense_head=use_dense_head,
            npz_name=npz_name,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            circuit_type=circuit_type,
            entanglement=entanglement,
            save_artifacts=False,   # do NOT spam files during tuning
        )

        return result["final_mse"]

    return objective


def run_optuna_study(
    config_path: str,
    mode: str,
    npz_name: str = "qnn_datasets.npz",
    n_trials: int = 30,
    n_epochs: int = 10,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    best_params_path: Optional[str] = None,
):
    """
    Run Optuna study and return it.
    - storage: e.g. 'sqlite:///qnn_optuna.db' if you want persistence.
    """
    objective = create_optuna_objective(
        config_path=config_path,
        mode=mode,
        npz_name=npz_name,
        n_epochs=n_epochs,
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials)

    print("\n=== Optuna best result ===")
    print("Best value (MSE):", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    if best_params_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_config_path = os.path.join(base_dir, "config", "model_config.yaml")
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        results_path = os.path.join(base_dir, model_config["paths"]["results"])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = study_name if study_name else mode
        filename = f"optuna_best_params_{tag}_{stamp}.json"
        best_params_path = os.path.join(results_path, "optuna", filename)

    os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
    payload = {
        "study_name": study.study_name,
        "mode": mode,
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": n_trials,
        "n_epochs": n_epochs,
        "storage": storage,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Wrote Optuna best params to {best_params_path}")

    return study

if __name__ == "__main__":
    # RUN OPTUNA TUNING FOR RETURNS (overnight)
    print("===== Running Optuna tuning for RETURNS (overnight) =====")
    _ = run_optuna_study(
        config_path="config/data_config.yaml",
        mode="returns",
        npz_name="qnn_datasets.npz",
        n_trials=50,
        n_epochs=8,
        study_name="qnn_returns_overnight",
        storage="sqlite:///qnn_optuna.db",
    )
