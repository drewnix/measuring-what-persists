"""Bootstrap resampling for magnitude and distance CIs."""

import numpy as np
from typing import Dict, List, Tuple
from .distances import empirical_distribution, sqrt_jsd, SQRT_LN2
from .magnitude import scalar_magnitude


def bootstrap_distance(
    samples_a: List[str],
    samples_b: List[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    k = len(samples_a)
    arr_a = np.array(samples_a)
    arr_b = np.array(samples_b)

    point_est = sqrt_jsd(empirical_distribution(samples_a), empirical_distribution(samples_b))

    boot_dists = []
    for _ in range(n_resamples):
        idx_a = rng.choice(k, size=k, replace=True)
        idx_b = rng.choice(k, size=k, replace=True)
        d_a = empirical_distribution(arr_a[idx_a].tolist())
        d_b = empirical_distribution(arr_b[idx_b].tolist())
        boot_dists.append(sqrt_jsd(d_a, d_b))

    boot_arr = np.array(boot_dists)
    return {
        "point_estimate": float(point_est),
        "bootstrap_mean": float(np.mean(boot_arr)),
        "ci_lower": float(np.percentile(boot_arr, 2.5)),
        "ci_upper": float(np.percentile(boot_arr, 97.5)),
        "bootstrap_se": float(np.std(boot_arr)),
        "bootstrap_bias": float(np.mean(boot_arr) - point_est),
    }


def bootstrap_magnitude(
    samples: Dict[str, Dict[str, List[str]]],
    probe_order: List[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    n_probes = len(probe_order)

    def compute_mag(sample_dict):
        dists = {p: empirical_distribution(sample_dict[p]) for p in probe_order}
        D = np.zeros((n_probes, n_probes))
        for i in range(n_probes):
            for j in range(i + 1, n_probes):
                d = sqrt_jsd(dists[probe_order[i]], dists[probe_order[j]])
                D[i, j] = d
                D[j, i] = d
        return scalar_magnitude(D)

    point_est = compute_mag(samples)
    k = len(next(iter(samples.values())))

    boot_mags = []
    for _ in range(n_resamples):
        resampled = {}
        for p in probe_order:
            arr = np.array(samples[p])
            idx = rng.choice(k, size=k, replace=True)
            resampled[p] = arr[idx].tolist()
        boot_mags.append(compute_mag(resampled))

    boot_arr = np.array(boot_mags)
    return {
        "point_estimate": float(point_est),
        "bootstrap_mean": float(np.mean(boot_arr)),
        "ci_lower": float(np.percentile(boot_arr, 2.5)),
        "ci_upper": float(np.percentile(boot_arr, 97.5)),
        "bootstrap_se": float(np.std(boot_arr)),
        "bootstrap_bias": float(np.mean(boot_arr) - point_est),
    }
