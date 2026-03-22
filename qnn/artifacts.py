from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from data.common import resolve_path

from .config import QNNExperimentConfig


@dataclass
class ArtifactLocation:
    root: str
    predictions_path: str
    model_path: str
    metrics_path: str
    summary_path: str
    plots_dir: str


@dataclass
class RunArtifacts:
    run_tag: str
    archive: ArtifactLocation
    latest: ArtifactLocation


class ArtifactManager:
    def __init__(self, config: QNNExperimentConfig):
        self.config = config

    def _location(self, root: str) -> ArtifactLocation:
        return ArtifactLocation(
            root=root,
            predictions_path=os.path.join(root, "predictions.npz"),
            model_path=os.path.join(root, "model_state.pth"),
            metrics_path=os.path.join(root, "metrics.npz"),
            summary_path=os.path.join(root, "summary.json"),
            plots_dir=os.path.join(root, "plots"),
        )

    def prepare(self, mode: str, run_tag: str | None = None) -> RunArtifacts:
        results_root = resolve_path(self.config.base_dir, self.config.paths.results)
        if run_tag is None:
            run_tag = f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        archive_root = os.path.join(results_root, self.config.metrics.runs_dirname, run_tag)
        latest_root = os.path.join(results_root, self.config.metrics.latest_dirname, mode)

        archive = self._location(archive_root)
        latest = self._location(latest_root)
        for location in (archive, latest):
            os.makedirs(location.root, exist_ok=True)
            os.makedirs(location.plots_dir, exist_ok=True)
        return RunArtifacts(run_tag=run_tag, archive=archive, latest=latest)

    @staticmethod
    def _save_npz(path: str, payload: dict[str, Any]) -> None:
        serialised: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, str):
                serialised[key] = np.asarray(value, dtype=str)
            else:
                serialised[key] = value
        np.savez_compressed(path, **serialised)

    @staticmethod
    def _save_json(path: str, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def save_predictions(self, artifacts: RunArtifacts, payload: dict[str, Any]) -> None:
        for location in (artifacts.archive, artifacts.latest):
            self._save_npz(location.predictions_path, payload)

    def save_metrics(self, artifacts: RunArtifacts, payload: dict[str, Any]) -> None:
        for location in (artifacts.archive, artifacts.latest):
            self._save_npz(location.metrics_path, payload)

    def save_summary(self, artifacts: RunArtifacts, payload: dict[str, Any]) -> None:
        for location in (artifacts.archive, artifacts.latest):
            self._save_json(location.summary_path, payload)

    def save_model_state(self, artifacts: RunArtifacts, state_dict) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is required to save model weights.") from exc

        for location in (artifacts.archive, artifacts.latest):
            torch.save(state_dict, location.model_path)
