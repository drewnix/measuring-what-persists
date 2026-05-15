"""√JSD distance computation from token distributions."""

import numpy as np
from collections import Counter
from typing import Dict, List, Union


def empirical_distribution(samples: List[str]) -> Dict[str, float]:
    counts = Counter(samples)
    total = len(samples)
    return {token: count / total for token, count in counts.items()}


def jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    all_tokens = set(p.keys()) | set(q.keys())
    p_arr = np.array([p.get(t, 0.0) for t in all_tokens])
    q_arr = np.array([q.get(t, 0.0) for t in all_tokens])
    m = 0.5 * (p_arr + q_arr)
    kl_pm = np.sum(p_arr[p_arr > 0] * np.log(p_arr[p_arr > 0] / m[p_arr > 0]))
    kl_qm = np.sum(q_arr[q_arr > 0] * np.log(q_arr[q_arr > 0] / m[q_arr > 0]))
    return 0.5 * (kl_pm + kl_qm)


def sqrt_jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    return np.sqrt(jsd(p, q))


def distance_matrix_from_distributions(
    distributions: Dict[str, Dict[str, float]],
    probe_order: List[str],
) -> np.ndarray:
    n = len(probe_order)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = sqrt_jsd(distributions[probe_order[i]], distributions[probe_order[j]])
            D[i, j] = d
            D[j, i] = d
    return D


SQRT_LN2 = np.sqrt(np.log(2))
