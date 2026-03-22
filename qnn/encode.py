from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA


def compress_features_to_angles(X: np.ndarray, n_qubits: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n_samples, _ = X.shape
    angles = np.zeros((n_samples, n_qubits), dtype=np.float32)
    for idx in range(n_samples):
        chunks = np.array_split(X[idx], n_qubits)
        angles[idx] = np.asarray([chunk.mean() for chunk in chunks], dtype=np.float32)
    return angles


@dataclass
class FeatureEncoder:
    n_qubits: int
    feature_mode: str = "angles"
    transformer: Optional[PCA] = None

    def fit_transform(self, X_train_raw: np.ndarray) -> np.ndarray:
        X_train_raw = np.asarray(X_train_raw, dtype=np.float32)
        if self.feature_mode == "angles":
            self.transformer = None
            return compress_features_to_angles(X_train_raw, self.n_qubits)
        if self.feature_mode == "pca":
            self.transformer = PCA(n_components=self.n_qubits)
            return self.transformer.fit_transform(X_train_raw).astype(np.float32)
        raise ValueError("feature_mode must be 'angles' or 'pca'.")

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        X_raw = np.asarray(X_raw, dtype=np.float32)
        if self.feature_mode == "angles":
            return compress_features_to_angles(X_raw, self.n_qubits)
        if self.feature_mode == "pca":
            if self.transformer is None:
                raise ValueError("PCA encoder must be fitted before calling transform().")
            return self.transformer.transform(X_raw).astype(np.float32)
        raise ValueError("feature_mode must be 'angles' or 'pca'.")

    def metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "feature_mode": self.feature_mode,
            "n_qubits": self.n_qubits,
        }
        if self.transformer is not None:
            payload["explained_variance_ratio"] = self.transformer.explained_variance_ratio_.astype(
                np.float32
            )
        return payload


def transform_features(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    n_qubits: int,
    feature_mode: str = "angles",
):
    encoder = FeatureEncoder(n_qubits=n_qubits, feature_mode=feature_mode)
    X_train = encoder.fit_transform(X_train_raw)
    X_test = encoder.transform(X_test_raw)
    return X_train, X_test, encoder.transformer
