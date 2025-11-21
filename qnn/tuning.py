""" 
Includes Optuna hyperparameter tuning over:
- n_qubits, n_layers, feature_mode, use_dense_head,
  circuit_type, learning_rate, batch_size
"""
from typing import Optional
import optuna
from qnn.train import train_qnn_from_npz

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
        n_qubits = trial.suggest_int("n_qubits", 3, 7)
        n_layers = trial.suggest_int("n_layers", 1, 4)
        feature_mode = trial.suggest_categorical("feature_mode", ["angles", "pca"])
        use_dense_head = trial.suggest_categorical("use_dense_head", [True, False])
        circuit_type = trial.suggest_categorical("circuit_type", ["rxrz", "zz_feature"])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        entanglement = trial.suggest_categorical("entanglement", ["ring", "linear"])

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

    return study

if __name__ == "__main__":
    # Example 3: RUN OPTUNA TUNING FOR RETURNS (quick small study)
    # Comment out if you don't want to tune every run.
    print("\n===== Running Optuna tuning for RETURNS (demo) =====")
    _ = run_optuna_study(
        config_path="config/data_config.yaml",
        mode="returns",
        npz_name="qnn_datasets.npz",
        n_trials=10,          # increase to 50–100 for real tuning
        n_epochs=10,          # smaller epochs for tuning
        study_name="qnn_returns_demo",
        storage="sqlite:///qnn_optuna.db",         # or e.g. "sqlite:///qnn_optuna.db"
    )