"""Scalar magnitude and magnitude weights from distance matrix."""

import numpy as np


def similarity_matrix(D: np.ndarray) -> np.ndarray:
    return np.exp(-D)


def scalar_magnitude(D: np.ndarray) -> float:
    Z = similarity_matrix(D)
    ones = np.ones(Z.shape[0])
    weights = np.linalg.solve(Z, ones)
    return float(np.sum(weights))


def magnitude_weights(D: np.ndarray) -> np.ndarray:
    Z = similarity_matrix(D)
    ones = np.ones(Z.shape[0])
    return np.linalg.solve(Z, ones)
