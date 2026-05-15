"""Magnitude homology group computation via boundary operator ranks."""

import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple


def _is_nondegenerate(chain: Tuple[int, ...], D: np.ndarray, eps: float = 1e-10) -> bool:
    for i in range(len(chain) - 2):
        a, b, c = chain[i], chain[i + 1], chain[i + 2]
        if abs(D[a, c] - (D[a, b] + D[b, c])) < eps:
            return False
    return True


def _chain_length(chain: Tuple[int, ...], D: np.ndarray) -> float:
    return sum(D[chain[i], chain[i + 1]] for i in range(len(chain) - 1))


def _bin_length(length: float, bin_width: float = 1.0) -> int:
    return int(round(length / bin_width))


def compute_mh_ranks(
    D: np.ndarray,
    max_degree: int = 3,
    length_bin_width: float = 1.0,
    betweenness_eps: float = 1e-10,
) -> Dict[int, Dict[int, int]]:
    n = D.shape[0]
    points = list(range(n))
    all_lengths = set()

    chains_by_degree_length: Dict[int, Dict[int, List[Tuple[int, ...]]]] = {}

    for degree in range(max_degree + 1):
        chains_by_degree_length[degree] = {}
        chain_len = degree + 1
        for chain in combinations(points, chain_len):
            import itertools
            for perm in itertools.permutations(chain):
                if not _is_nondegenerate(perm, D, betweenness_eps):
                    continue
                ell = _chain_length(perm, D)
                ell_bin = _bin_length(ell, length_bin_width)
                all_lengths.add(ell_bin)
                if ell_bin not in chains_by_degree_length[degree]:
                    chains_by_degree_length[degree][ell_bin] = []
                chains_by_degree_length[degree][ell_bin].append(perm)

    ranks: Dict[int, Dict[int, int]] = {}
    for degree in range(max_degree + 1):
        ranks[degree] = {}
        for ell_bin in sorted(all_lengths):
            chains_k = chains_by_degree_length.get(degree, {}).get(ell_bin, [])
            chains_k1 = chains_by_degree_length.get(degree + 1, {}).get(ell_bin, [])
            chains_km1 = chains_by_degree_length.get(degree - 1, {}).get(ell_bin, []) if degree > 0 else []

            n_chains = len(chains_k)
            if n_chains == 0:
                continue

            chain_to_idx_k = {c: i for i, c in enumerate(chains_k)}

            if degree > 0 and chains_km1:
                chain_to_idx_km1 = {c: i for i, c in enumerate(chains_km1)}
                boundary_in = np.zeros((len(chains_km1), n_chains))
                for j, chain in enumerate(chains_k):
                    for face_idx in range(len(chain)):
                        face = chain[:face_idx] + chain[face_idx + 1:]
                        if face in chain_to_idx_km1:
                            sign = (-1) ** face_idx
                            boundary_in[chain_to_idx_km1[face], j] = sign
                rank_boundary_in = np.linalg.matrix_rank(boundary_in)
            else:
                rank_boundary_in = 0

            if chains_k1:
                boundary_out = np.zeros((n_chains, len(chains_k1)))
                for j, chain in enumerate(chains_k1):
                    for face_idx in range(len(chain)):
                        face = chain[:face_idx] + chain[face_idx + 1:]
                        if face in chain_to_idx_k:
                            sign = (-1) ** face_idx
                            boundary_out[chain_to_idx_k[face], j] = sign
                rank_boundary_out = np.linalg.matrix_rank(boundary_out)
            else:
                rank_boundary_out = 0

            mh_rank = n_chains - rank_boundary_in - rank_boundary_out
            if mh_rank != 0:
                ranks[degree][ell_bin] = mh_rank

    return ranks


def check_betweenness(
    D: np.ndarray,
    labels: List[str],
    eps: float = 0.05,
) -> List[Tuple[str, str, str]]:
    n = D.shape[0]
    betweenness = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i == j or j == k or i == k:
                    continue
                if abs(D[i, k] - (D[i, j] + D[j, k])) < eps:
                    betweenness.append((labels[i], labels[j], labels[k]))
    return betweenness
