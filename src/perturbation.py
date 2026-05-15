"""Perturbation theory for magnitude of equilateral probe configurations."""

import numpy as np


SQRT_LN2 = np.sqrt(np.log(2))
D_EQUILATERAL = SQRT_LN2
A = np.exp(-D_EQUILATERAL)  # ≈ 0.4349


def eigenvalues_equilateral(n: int = 3) -> tuple:
    a = A
    lambda1 = 1 + (n - 1) * a
    lambda2 = 1 - a
    return lambda1, lambda2


def condition_number(n: int = 3) -> float:
    l1, l2 = eigenvalues_equilateral(n)
    return l1 / l2


def perturbation_coefficient(n: int = 3) -> float:
    a = A
    return 2 * a / (1 + (n - 1) * a) ** 2


def worst_case_bound_coefficient(n: int = 3) -> float:
    a = A
    return n * (1 / (1 - a)) ** 2


def z_inverse_analytic(n: int = 3) -> np.ndarray:
    a = A
    I = np.eye(n)
    J = np.ones((n, n))
    return (1 / (1 - a)) * I - (a / ((1 - a) * (1 + (n - 1) * a))) * J


def z_inverse_ones(n: int = 3) -> np.ndarray:
    a = A
    return (1 / (1 + (n - 1) * a)) * np.ones(n)


def perimeter_change(baseline_distances: np.ndarray, drift_distances: np.ndarray) -> float:
    return float(np.sum(drift_distances - baseline_distances))


def first_order_prediction(delta_perimeter: float) -> float:
    coeff = perturbation_coefficient()
    return coeff * delta_perimeter


def second_order_shape_term(delta_d: np.ndarray) -> float:
    d = D_EQUILATERAL
    a = A
    mean_delta = np.mean(delta_d)
    shape_deviations = delta_d - mean_delta
    coeff = np.exp(-2 * d) / (1 + 2 * np.exp(-d)) ** 3
    return float(coeff * np.sum(shape_deviations ** 2))


def mode_decomposition(delta_d: np.ndarray) -> dict:
    mean_delta = np.mean(delta_d)
    breathing = np.full_like(delta_d, mean_delta)
    shearing = delta_d - breathing
    total_var = np.var(delta_d, ddof=0) * len(delta_d)
    shearing_var = np.sum(shearing ** 2)
    breathing_var = np.sum(breathing ** 2)
    # Paper reports norm ratio ||shearing|| / ||breathing|| (= coefficient of variation)
    shearing_frac = np.sqrt(shearing_var / breathing_var) if breathing_var > 0 else 0.0
    return {
        "mean_delta": float(mean_delta),
        "breathing": breathing.tolist(),
        "shearing": shearing.tolist(),
        "shearing_variance_fraction": float(shearing_frac),
        "breathing_perimeter": float(np.sum(breathing)),
        "shearing_perimeter": float(np.sum(shearing)),
    }


def coefficient_ratio() -> float:
    a = A
    return (1 - a) ** 2 / (1 + 2 * a) ** 2
