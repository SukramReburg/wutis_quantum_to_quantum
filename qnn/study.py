from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data.common import project_base_dir, resolve_path

try:
    from .config import deep_merge, load_experiment_config, load_project_config, set_nested
    from .trainer import QNNTrainer
except ImportError:
    from qnn.config import deep_merge, load_experiment_config, load_project_config, set_nested
    from qnn.trainer import QNNTrainer

try:
    import optuna
    from optuna.importance import get_param_importances
except ModuleNotFoundError:
    optuna = None
    get_param_importances = None


def _require_optuna() -> None:
    if optuna is None:
        raise RuntimeError("Optuna is required to run QNN hyperparameter tuning.")


class QNNStudyRunner:
    def __init__(self, config_path: str = "config/model_config.yaml", base_dir: str | None = None):
        self.config_path = config_path
        self.base_dir = project_base_dir(__file__, base_dir)
        self.project_config = load_project_config(config_path=config_path, base_dir=self.base_dir)

    def _search_space(self, mode: str) -> dict[str, Any]:
        tuning = self.project_config.get("tuning", {})
        search_space = deep_merge(dict(tuning.get("search_space", {})), {})
        mode_space = tuning.get("mode_search_space", {}).get(mode, {})
        return deep_merge(search_space, mode_space)

    def _suggest(self, trial, name: str, spec: dict[str, Any]) -> Any:
        kind = spec["type"]
        if kind == "int":
            return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        if kind == "float":
            return trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step"),
                log=spec.get("log", False),
            )
        if kind == "categorical":
            return trial.suggest_categorical(name, list(spec["choices"]))
        if kind == "bool":
            return trial.suggest_categorical(name, [True, False])
        raise ValueError(f"Unsupported search space type '{kind}' for '{name}'.")

    def create_objective(self, mode: str, runtime_profile: str | None = None):
        search_space = self._search_space(mode)
        tuning_config = self.project_config.get("tuning", {})

        def objective(trial) -> float:
            overrides: dict[str, Any] = {
                "training": {
                    "save_trial_artifacts": tuning_config.get("save_trial_artifacts", False),
                },
                "plots": {"enabled": tuning_config.get("save_trial_artifacts", False)},
                "metrics": {
                    "save_model": tuning_config.get("save_trial_artifacts", False),
                    "save_predictions": tuning_config.get("save_trial_artifacts", False),
                    "save_metrics": tuning_config.get("save_trial_artifacts", False),
                    "save_summary": tuning_config.get("save_trial_artifacts", False),
                },
            }
            if runtime_profile is not None:
                overrides["runtime"] = {"profile": runtime_profile}
            for dotted_name, spec in search_space.items():
                set_nested(overrides, dotted_name, self._suggest(trial, dotted_name, spec))

            experiment = load_experiment_config(
                mode=mode,
                config_path=self.config_path,
                base_dir=self.base_dir,
                overrides=overrides,
            )
            result = QNNTrainer(experiment).train(
                no_plots=not tuning_config.get("save_trial_artifacts", False),
                run_tag=f"{mode}_trial_{trial.number:04d}",
                save_artifacts=tuning_config.get("save_trial_artifacts", False),
            )

            metric_name = tuning_config.get("metric", "rmse")
            value = result.summary["test_metrics"][metric_name]
            trial.set_user_attr("run_tag", result.artifacts.run_tag)
            trial.set_user_attr("best_epoch", result.summary["best_epoch"])
            return float(value)

        return objective

    def _study_output_dir(self, study_name: str) -> str:
        optuna_dir = resolve_path(self.base_dir, self.project_config["paths"]["optuna"])
        path = os.path.join(optuna_dir, study_name)
        os.makedirs(path, exist_ok=True)
        return path

    def _save_study_artifacts(self, study, output_dir: str) -> None:
        dataframe = study.trials_dataframe()
        dataframe.to_csv(os.path.join(output_dir, "trials.csv"), index=False)

        summary_path = os.path.join(output_dir, "best_params.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "study_name": study.study_name,
                    "best_value": float(study.best_value),
                    "best_params": study.best_params,
                    "n_trials": len(study.trials),
                },
                handle,
                indent=2,
                sort_keys=True,
            )

        plt.figure(figsize=(8, 4.5))
        values = [trial.value for trial in study.trials if trial.value is not None]
        plt.plot(values, marker="o")
        plt.xlabel("Trial")
        plt.ylabel(study.direction.name.lower())
        plt.title("Optuna Trial History")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "history.png"), dpi=180)
        plt.close()

        if get_param_importances is not None:
            importances = get_param_importances(study)
            if importances:
                labels = list(importances.keys())
                values = [importances[label] for label in labels]
                plt.figure(figsize=(8, 4.5))
                plt.barh(labels, values)
                plt.xlabel("Importance")
                plt.title("Parameter Importance")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "param_importance.png"), dpi=180)
                plt.close()

    def run(
        self,
        mode: str,
        study_name: str | None = None,
        n_trials: int | None = None,
        runtime_profile: str | None = None,
    ):
        _require_optuna()
        tuning = self.project_config.get("tuning", {})
        if study_name is None:
            study_name = tuning.get("study_name") or f"qnn_{mode}_study"

        objective = self.create_objective(mode, runtime_profile=runtime_profile)
        storage = tuning.get("storage")
        study = optuna.create_study(
            direction=tuning.get("direction", "minimize"),
            study_name=study_name,
            storage=storage,
            load_if_exists=storage is not None,
        )
        if runtime_profile is not None:
            study.set_user_attr("runtime_profile_override", runtime_profile)
        study.optimize(
            objective,
            n_trials=n_trials or tuning.get("n_trials", 20),
            n_jobs=tuning.get("n_jobs", 1),
            timeout=tuning.get("timeout_seconds"),
        )

        output_dir = self._study_output_dir(study_name)
        self._save_study_artifacts(study, output_dir)
        return study
