from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml

from data.common import project_base_dir, resolve_path


@dataclass
class AnalysisOutputLocation:
    plots_dir: str
    reports_dir: str


def analysis_roots(model_config: dict[str, Any], data_config: dict[str, Any]) -> tuple[str, str]:
    """Resolve the canonical analysis plot/report roots from loaded configs."""

    plots_root = model_config.get("paths", {}).get(
        "plots",
        data_config.get("paths", {}).get("plots", "analysis/plots"),
    )
    reports_root = model_config.get("paths", {}).get("reports", "analysis/reports")
    return plots_root, reports_root


def load_analysis_configs(
    base_dir: str | None = None,
    model_config_path: str = "config/model_config.yaml",
    data_config_path: str = "config/data_config.yaml",
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    project_dir = project_base_dir(__file__, base_dir)
    with open(resolve_path(project_dir, model_config_path), "r", encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)
    with open(resolve_path(project_dir, data_config_path), "r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)
    return project_dir, model_config, data_config


class AnalysisPathManager:
    """Resolves stage-specific plot and report folders under analysis/."""

    def __init__(
        self,
        base_dir: str,
        plots_root: str = "analysis/plots",
        reports_root: str = "analysis/reports",
    ):
        self.base_dir = os.path.abspath(base_dir)
        self.plots_root = resolve_path(self.base_dir, plots_root)
        self.reports_root = resolve_path(self.base_dir, reports_root)

    @classmethod
    def from_configs(
        cls,
        base_dir: str | None = None,
        model_config_path: str = "config/model_config.yaml",
        data_config_path: str = "config/data_config.yaml",
    ) -> "AnalysisPathManager":
        project_dir, model_config, data_config = load_analysis_configs(
            base_dir=base_dir,
            model_config_path=model_config_path,
            data_config_path=data_config_path,
        )
        plots_root, reports_root = analysis_roots(model_config, data_config)
        return cls(project_dir, plots_root=plots_root, reports_root=reports_root)

    def stage(self, *parts: str, run_tag: str | None = None) -> AnalysisOutputLocation:
        if run_tag is None:
            plots_dir = os.path.join(self.plots_root, *parts, "latest")
            reports_dir = os.path.join(self.reports_root, *parts, "latest")
        else:
            plots_dir = os.path.join(self.plots_root, *parts, "runs", run_tag)
            reports_dir = os.path.join(self.reports_root, *parts, "runs", run_tag)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        return AnalysisOutputLocation(plots_dir=plots_dir, reports_dir=reports_dir)

    def data(self, run_tag: str | None = None) -> AnalysisOutputLocation:
        return self.stage("data", run_tag=run_tag)

    def model(self, mode: str, run_tag: str | None = None) -> AnalysisOutputLocation:
        return self.stage("models", mode, run_tag=run_tag)

    def tuning(self, study_name: str) -> AnalysisOutputLocation:
        return self.stage("tuning", study_name)

    def optimizer(self, run_tag: str | None = None) -> AnalysisOutputLocation:
        return self.stage("optimizer", run_tag=run_tag)


def save_json(path: str, payload: dict[str, Any]) -> None:
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
