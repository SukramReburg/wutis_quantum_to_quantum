import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Optional

# ======================================================================
# FEATURE PREPROCESSING
# ======================================================================

def compress_features_to_angles(X: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Compress high-dimensional features into n_qubits angles
    by chunk-averaging along the feature dimension.
    """
    X = np.asarray(X, dtype=np.float32)
    n_samples, _ = X.shape
    angles = np.zeros((n_samples, n_qubits), dtype=np.float32)
    for i in range(n_samples):
        chunks = np.array_split(X[i], n_qubits)
        angles[i] = np.array([chunk.mean() for chunk in chunks], dtype=np.float32)
    return angles


def transform_features(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    n_qubits: int,
    feature_mode: str = "angles",
) -> Tuple[np.ndarray, np.ndarray, Optional[PCA]]:
    """
    Transform raw features into size (n_samples, n_qubits).

    feature_mode:
        - "angles": chunk-mean angle encoding
        - "pca": PCA to n_qubits components
    """
    X_train_raw = np.asarray(X_train_raw, dtype=np.float32)
    X_test_raw = np.asarray(X_test_raw, dtype=np.float32)

    if feature_mode == "angles":
        X_train = compress_features_to_angles(X_train_raw, n_qubits)
        X_test = compress_features_to_angles(X_test_raw, n_qubits)
        transformer = None
    elif feature_mode == "pca":
        pca = PCA(n_components=n_qubits)
        X_train = pca.fit_transform(X_train_raw).astype(np.float32)
        X_test = pca.transform(X_test_raw).astype(np.float32)
        transformer = pca
    else:
        raise ValueError("feature_mode must be 'angles' or 'pca'")

    return X_train, X_test, transformer