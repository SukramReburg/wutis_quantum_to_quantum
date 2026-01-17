import numpy as np

def rebuild_covariance(vec, n_assets):
    iu = np.triu_indices(n_assets)
    C = np.zeros((n_assets, n_assets))
    C[iu] = vec
    C[(iu[1], iu[0])] = vec
    return C


def make_psd(C, eps=1e-6):
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

