#!/usr/bin/env python3
"""
Verify every computable number in "Measuring What Persists" against experimental data.

Exit code 0: all checks pass. Exit code 1: any check fails.
Warnings (e.g., rounding inconsistencies) don't cause failure.

Usage:
    python paper_table_check.py [--data-dir PATH]

Default data directory: data/
"""

import argparse
import json
import sys
import os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from src.distances import sqrt_jsd, empirical_distribution, SQRT_LN2
from src.magnitude import scalar_magnitude, similarity_matrix, magnitude_weights
from src.perturbation import (
    eigenvalues_equilateral,
    condition_number,
    perturbation_coefficient,
    worst_case_bound_coefficient,
    z_inverse_analytic,
    z_inverse_ones,
    first_order_prediction,
    second_order_shape_term,
    mode_decomposition,
    coefficient_ratio,
    A, D_EQUILATERAL,
)

# ─── Utilities ───────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0
WARN = 0


def check(description, computed, expected, tol=1e-3, fmt=".4f"):
    global PASS, FAIL
    if isinstance(expected, str):
        ok = str(computed) == expected
        comp_str = str(computed)
        exp_str = expected
    else:
        ok = abs(computed - expected) <= tol
        comp_str = f"{computed:{fmt}}"
        exp_str = f"{expected:{fmt}}"
    if ok:
        PASS += 1
        print(f"  ✓ {description}: {comp_str}")
    else:
        FAIL += 1
        print(f"  ✗ {description}: got {comp_str}, expected {exp_str}")


def warn(description, message):
    global WARN
    WARN += 1
    print(f"  ⚠️  {description}: {message}")


def section(title):
    print(f"\n=== {title} ===")


# ─── Data loading ────────────────────────────────────────────────────────────

def load_json(data_dir, filename):
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_data(data_dir):
    data = {}
    # Try dated names first (qai repo), then clean names (public repo)
    files = {
        "drift": ["drift_experiment_2026-05-06.json", "drift_experiment.json"],
        "null_model": ["null_model_equilateral_2026-04-28.json", "null_model_control.json"],
        "canonical_7probe": ["canonical_7probe_homology_2026-04-27.json", "cross_condition_canonical.json"],
        "bootstrap": ["bootstrap_validation_2026-05-08.json", "bootstrap_validation.json"],
        "reconstitution": ["reconstitution_v2_2026-05-06.json", "reconstitution.json"],
        "cross_condition": ["cross_condition_2026-04-26.json", "cross_condition_3probe.json"],
    }
    for key, candidates in files.items():
        found = None
        for filename in candidates:
            found = load_json(data_dir, filename)
            if found is not None:
                break
        data[key] = found
        if found is None:
            print(f"  ⚠️  Missing: {' or '.join(candidates)}")
    return data


def get_drift_distances(drift_data, condition, cond_type):
    dm = drift_data["distance_matrices"][condition][cond_type]
    probes = ["Q1", "Q3", "B4"]
    D = np.zeros((3, 3))
    for i, pi in enumerate(probes):
        for j, pj in enumerate(probes):
            D[i, j] = dm[pi][pj]
    return D


# ─── Section checks ─────────────────────────────────────────────────────────

def check_section2_framework(data):
    section("Section 2 / Appendix A: Perturbation Theory Foundations")

    a = A
    print(f"  a = e^(-√(ln 2)) = {a:.10f}")

    l1, l2 = eigenvalues_equilateral(3)
    check("λ₁ = 1 + 2a", l1, 1 + 2 * a, tol=1e-10)
    check("λ₂ = λ₃ = 1 - a", l2, 1 - a, tol=1e-10)
    check("Paper λ₁ ≈ 1.87 (rounded)", round(l1, 2), 1.87, tol=0.005)
    check("Paper λ₂ ≈ 0.57 (rounded)", round(l2, 2), 0.57, tol=0.005)

    kappa = condition_number(3)
    check("κ(Z)", kappa, (1 + 2 * a) / (1 - a), tol=1e-10)
    check("Paper κ(Z) ≈ 3.31", kappa, 3.31, tol=0.005)

    bound_coeff = worst_case_bound_coefficient(3)
    check("Bound coefficient n·(1/(1-a))²", bound_coeff, 3 * (1 / (1 - a)) ** 2, tol=1e-10)

    pert_coeff = perturbation_coefficient(3)
    check("Perturbation coefficient 2a/(1+2a)²", pert_coeff, 0.2488, tol=0.001)

    Z_inv = z_inverse_analytic(3)
    Z = similarity_matrix(np.array([[0, D_EQUILATERAL, D_EQUILATERAL],
                                     [D_EQUILATERAL, 0, D_EQUILATERAL],
                                     [D_EQUILATERAL, D_EQUILATERAL, 0]]))
    Z_inv_numeric = np.linalg.inv(Z)
    check("Z⁻¹ analytic matches numeric", np.max(np.abs(Z_inv - Z_inv_numeric)), 0.0, tol=1e-10, fmt=".2e")

    z_inv_1 = Z_inv @ np.ones(3)
    expected_val = 1 / (1 + 2 * a)
    check("Z⁻¹·1 = (1/(1+2a))·1", z_inv_1[0], expected_val, tol=1e-10)
    check("Paper Z⁻¹·1 ≈ 0.535", expected_val, 0.535, tol=0.001)

    ratio = coefficient_ratio()
    check("(1-a)²/(1+2a)² ≈ 0.09", ratio, (1 - a) ** 2 / (1 + 2 * a) ** 2, tol=1e-10)


def check_section4_entropy(data):
    section("Section 4: Entropy and Leading Indicators")

    drift = data["drift"]
    if drift is None:
        warn("Section 4", "drift data not available")
        return

    # First-token check at medium base
    q1_base_samples = drift["samples"]["Q1"]["medium_base"]
    q1_base_first = [s.split()[0] for s in q1_base_samples if s]
    base_counts = Counter(q1_base_first)
    top_token, top_count = base_counts.most_common(1)[0]
    check("Q1 base top first-token = 'This'", top_token, "This")

    # Null model unique prefix counts at k=200
    null = data["null_model"]
    if null is not None:
        k200_card = null.get("k200_card", {})
        k200_base = null.get("k200_base", {})
        if "prefix_lists" in k200_card:
            card_q3 = k200_card["prefix_lists"].get("card_Q3_voice", [])
            check("Null-model card Q3 unique prefixes (k=200)", len(set(card_q3)), 55, tol=0)
        if "prefix_lists" in k200_base:
            base_q3 = k200_base["prefix_lists"].get("base_Q3_voice", [])
            check("Null-model base Q3 unique prefixes (k=200)", len(set(base_q3)), 1, tol=0)


def check_section52_equilateral(data):
    section("Section 5.2: Equilateral Baseline")

    drift = data["drift"]
    if drift is None:
        warn("Section 5.2", "drift data not available")
        return

    probes = ["Q1", "Q3", "B4"]
    D_baseline_card = get_drift_distances(drift, "baseline", "card")

    for i in range(3):
        for j in range(i + 1, 3):
            check(f"Baseline d({probes[i]},{probes[j]})", D_baseline_card[i, j], SQRT_LN2, tol=0.0001)

    mag_baseline = scalar_magnitude(D_baseline_card)
    check("Baseline |M| (recomputed)", mag_baseline, 1.6044, tol=0.0001)

    stored_mag = drift["homology"]["baseline"]["card"]["scalar_magnitude"]
    check("Baseline |M| (stored)", stored_mag, mag_baseline, tol=1e-8)

    # Null model control — Table 2
    null = data["null_model"]
    if null is not None:
        for config_key, label in [("k50_card", "Card k=50"), ("k50_base", "Base k=50"),
                                   ("k200_card", "Card k=200"), ("k200_base", "Base k=200")]:
            config = null[config_key]
            dm = config["distance_matrix"]
            if isinstance(dm, list):
                D_null = np.array(dm)
            else:
                probe_list = sorted(dm.keys())
                D_null = np.zeros((3, 3))
                for i, pi in enumerate(probe_list):
                    for j, pj in enumerate(probe_list):
                        D_null[i, j] = dm[pi][pj]
            for i in range(3):
                for j in range(i + 1, 3):
                    check(f"Null {label} d[{i},{j}] = √(ln2)", D_null[i, j], SQRT_LN2, tol=0.0001)
            mag_null = scalar_magnitude(D_null)
            check(f"Null {label} |M|", mag_null, 1.6044, tol=0.0001)


def check_section53_cross_condition(data):
    section("Section 5.3: Cross-Condition (7-probe)")

    cc = data["canonical_7probe"]
    if cc is None:
        warn("Section 5.3", "canonical 7-probe data not available")
        return

    expected_cc1 = {"Q1": 0.2762, "Q2": 0.6103, "Q3": 0.8326,
                    "B1": 0.7361, "B2": 0.4169, "B3": 0.8326, "B4": 0.5000}
    for probe, expected in expected_cc1.items():
        actual = cc["cc1_values"].get(probe)
        if actual is not None:
            check(f"cc₁({probe})", actual, expected, tol=0.001)

    mag_6pt = cc["six_point"]["scalar_magnitude"]
    D_6pt = np.array(cc["six_point"]["distance_matrix"])
    mag_6pt_recomputed = scalar_magnitude(D_6pt)
    check("|M|_cc recomputed from 6-pt matrix", mag_6pt_recomputed, mag_6pt, tol=1e-6)
    check("§5.3 |M|_cc ≈ 1.2779", mag_6pt, 1.2779, tol=0.001)


def check_section54_three_probe_cc(data):
    section("Section 5.4: Three-Probe Cross-Condition")

    cc_data = data["cross_condition"]
    if cc_data is None:
        warn("Section 5.4", "cross_condition_2026-04-26.json not found")
        return

    cc1 = cc_data["cc_1"]
    q1_cc1 = cc1["Q1_disagreement"]["point"]
    q3_cc1 = cc1["Q3_voice"]["point"]
    b4_cc1 = cc1["B4_depth"]["point"]

    d_q1_q3 = abs(q1_cc1 - q3_cc1)
    d_q1_b4 = abs(q1_cc1 - b4_cc1)
    d_q3_b4 = abs(q3_cc1 - b4_cc1)

    check("3-probe cc d(Q1,Q3)", d_q1_q3, 0.8326, tol=0.001)
    check("3-probe cc d(Q1,B4)", d_q1_b4, 0.3425, tol=0.001)
    check("3-probe cc d(Q3,B4)", d_q3_b4, 0.4901, tol=0.001)

    b4_position = d_q1_b4 / d_q1_q3
    check("B4 at ~41% of Q1-Q3 distance", b4_position, 0.41, tol=0.01)

    D_3 = np.array([[0, d_q1_q3, d_q1_b4],
                     [d_q1_q3, 0, d_q3_b4],
                     [d_q1_b4, d_q3_b4, 0]])
    mag_3probe = scalar_magnitude(D_3)
    check("3-probe cc |M|_cc", mag_3probe, 1.4098, tol=0.001)


def check_section56_drift(data):
    section("Section 5.6: Drift Experiment")

    drift = data["drift"]
    if drift is None:
        warn("Section 5.6", "drift data not available")
        return

    probes = ["Q1", "Q3", "B4"]
    D_med_card = get_drift_distances(drift, "medium", "card")
    D_long_card = get_drift_distances(drift, "long", "card")

    # Table 5 distances
    check("Medium d(Q1,Q3)", D_med_card[0, 1], 0.8326, tol=0.001)
    check("Medium d(Q1,B4)", D_med_card[0, 2], 0.7932, tol=0.001)
    check("Medium d(Q3,B4)", D_med_card[1, 2], 0.8039, tol=0.001)
    check("Long d(Q1,Q3)", D_long_card[0, 1], 0.8058, tol=0.001)
    check("Long d(Q1,B4)", D_long_card[0, 2], 0.8242, tol=0.001)
    check("Long d(Q3,B4)", D_long_card[1, 2], 0.8051, tol=0.001)

    # Base condition equilateral at all lengths
    for length in ["baseline", "medium", "long"]:
        D_base = get_drift_distances(drift, length, "base")
        for i in range(3):
            for j in range(i + 1, 3):
                check(f"Base {length} d({probes[i]},{probes[j]})", D_base[i, j], SQRT_LN2, tol=0.0001)

    # Drift magnitudes
    mag_medium = scalar_magnitude(D_med_card)
    mag_long = scalar_magnitude(D_long_card)
    check("Medium |M| (recomputed)", mag_medium, 1.5875, tol=0.001)
    check("Long |M| (recomputed)", mag_long, 1.5888, tol=0.001)

    stored_med = drift["homology"]["medium"]["card"]["scalar_magnitude"]
    stored_long = drift["homology"]["long"]["card"]["scalar_magnitude"]
    check("Medium |M| (stored vs recomputed)", stored_med, mag_medium, tol=1e-8)
    check("Long |M| (stored vs recomputed)", stored_long, mag_long, tol=1e-8)

    # Contraction, not collapse — triangle inequality gap at long
    tri_sum = D_long_card[0, 1] + D_long_card[1, 2]
    check("d(Q1,Q3) + d(Q3,B4) at long", tri_sum, 1.6109, tol=0.001)

    # Zero betweenness
    for cond in ["medium", "long"]:
        betweenness = drift["homology"][cond]["card"].get("betweenness", [])
        check(f"Zero betweenness at {cond}", len(betweenness), 0, tol=0)

    # MH₁ rank splitting at medium
    mh_medium = drift["homology"]["medium"]["card"].get("homology_ranks", {})
    if mh_medium and "1" in mh_medium:
        mh1 = mh_medium["1"]
        check("MH₁ rank at ℓ=15 (medium)", mh1.get("15", -1), 2, tol=0)
        check("MH₁ rank at ℓ=16 (medium)", mh1.get("16", -1), 4, tol=0)


def check_bootstrap(data):
    section("Section 5.6: Bootstrap Validation")

    boot = data["bootstrap"]
    drift = data["drift"]
    if boot is None:
        warn("Bootstrap", "bootstrap data not available")
        return

    cis = boot["f004_bootstrap_cis"]

    # Table 7 — individual distance CIs
    checks_table7 = [
        ("Medium Q1-B4", "medium", "Q1_vs_B4", 0.7932, 0.7522, 0.8326, -4.7),
        ("Medium Q3-B4", "medium", "Q3_vs_B4", 0.8039, 0.7865, 0.8326, -3.4),
        ("Long Q1-Q3", "long", "Q1_vs_Q3", 0.8058, 0.7686, 0.8326, -3.2),
        ("Long Q3-B4", "long", "Q3_vs_B4", 0.8051, 0.7821, 0.8326, -3.3),
        ("Long Q1-B4", "long", "Q1_vs_B4", 0.8242, 0.8121, 0.8326, -1.0),
    ]
    for label, cond, pair, pt, ci_lo, ci_up, pct in checks_table7:
        d = cis[cond]["card"][pair]
        check(f"{label} point est", d["point_estimate"], pt, tol=0.001)
        check(f"{label} CI lower", d["ci_lower"], ci_lo, tol=0.002)
        check(f"{label} CI upper", d["ci_upper"], ci_up, tol=0.001)

    # Magnitude bootstrap — compute from raw samples
    if drift is not None and "samples" in drift:
        print("\n  Computing magnitude bootstrap from raw samples (1000 resamples)...")
        from src.bootstrap import bootstrap_magnitude

        for condition, label, exp_mean, exp_ci_lo, exp_ci_up in [
            ("medium", "Medium", 1.5907, 1.5782, 1.6023),
            ("long", "Long", 1.5914, 1.5781, 1.6023),
        ]:
            samples = {}
            for probe in ["Q1", "Q3", "B4"]:
                samples[probe] = drift["samples"][probe][f"{condition}_card"]
            result = bootstrap_magnitude(samples, ["Q1", "Q3", "B4"], n_resamples=1000, seed=42)
            check(f"{label} bootstrap mean", result["bootstrap_mean"], exp_mean, tol=0.003)
            check(f"{label} CI lower", result["ci_lower"], exp_ci_lo, tol=0.005)
            check(f"{label} CI upper", result["ci_upper"], exp_ci_up, tol=0.005)
            check(f"{label} CI upper < baseline (1.6044)", float(result["ci_upper"] < 1.6044), 1.0, tol=0)

        # Base condition — zero variance
        for condition in ["medium", "long"]:
            samples = {}
            for probe in ["Q1", "Q3", "B4"]:
                samples[probe] = drift["samples"][probe][f"{condition}_base"]
            result = bootstrap_magnitude(samples, ["Q1", "Q3", "B4"], n_resamples=1000, seed=42)
            check(f"Base {condition} bootstrap SE", result["bootstrap_se"], 0.0, tol=1e-6)

    # Signal-to-bias ratios
    ratios = []
    for cond, pair in [("medium", "Q1_vs_B4"), ("medium", "Q3_vs_B4"),
                       ("long", "Q1_vs_Q3"), ("long", "Q3_vs_B4"), ("long", "Q1_vs_B4")]:
        d = cis[cond]["card"][pair]
        signal = SQRT_LN2 - d["point_estimate"]
        bias = d["bootstrap_bias"]
        if bias > 0 and signal > 0:
            ratios.append(signal / bias)
    if ratios:
        check(f"Signal-to-bias range includes 2.2×", float(min(ratios) >= 1.5), 1.0, tol=0)
        check(f"Signal-to-bias min ≈ 2.2×", min(ratios), 2.2, tol=0.5, fmt=".1f")
        check(f"Signal-to-bias max ≈ 11.7×", max(ratios), 11.7, tol=2.0, fmt=".1f")


def check_perturbation_theory(data):
    section("Appendix A: Perturbation Theory (Self-Consistency)")

    drift = data["drift"]
    if drift is None:
        warn("Perturbation theory", "drift data not available")
        return

    D_baseline = get_drift_distances(drift, "baseline", "card")
    D_medium = get_drift_distances(drift, "medium", "card")
    D_long = get_drift_distances(drift, "long", "card")
    baseline_mag = scalar_magnitude(D_baseline)

    # Distance perturbations (upper triangle order: Q1-Q3, Q1-B4, Q3-B4)
    delta_d_medium = np.array([
        D_medium[0, 1] - D_baseline[0, 1],
        D_medium[0, 2] - D_baseline[0, 2],
        D_medium[1, 2] - D_baseline[1, 2],
    ])
    delta_d_long = np.array([
        D_long[0, 1] - D_baseline[0, 1],
        D_long[0, 2] - D_baseline[0, 2],
        D_long[1, 2] - D_baseline[1, 2],
    ])

    # Individual distance changes
    check("Medium δd(Q1,Q3) ≈ 0.000", delta_d_medium[0], 0.000, tol=0.001)
    check("Medium δd(Q1,B4) ≈ -0.0393", delta_d_medium[1], -0.0393, tol=0.001)
    check("Medium δd(Q3,B4) ≈ -0.0286", delta_d_medium[2], -0.0286, tol=0.001)
    check("Long δd(Q1,Q3) ≈ -0.0268", delta_d_long[0], -0.0268, tol=0.001)
    check("Long δd(Q1,B4) ≈ -0.0084", delta_d_long[1], -0.0084, tol=0.001)
    check("Long δd(Q3,B4) ≈ -0.0275", delta_d_long[2], -0.0275, tol=0.001)

    # Perimeter changes
    dp_medium = float(np.sum(delta_d_medium))
    dp_long = float(np.sum(delta_d_long))
    check("Medium δP", dp_medium, -0.0679, tol=0.001)
    check("Long δP", dp_long, -0.0626, tol=0.001)

    # First-order predictions
    pred_medium = first_order_prediction(dp_medium)
    pred_long = first_order_prediction(dp_long)
    obs_medium = scalar_magnitude(D_medium) - baseline_mag
    obs_long = scalar_magnitude(D_long) - baseline_mag

    check("Medium first-order δ|M|", pred_medium, -0.01690, tol=0.0002, fmt=".5f")
    check("Medium observed δ|M|", obs_medium, -0.01693, tol=0.0002, fmt=".5f")

    agreement_medium = abs(pred_medium - obs_medium) / abs(obs_medium) * 100
    check("Medium self-consistency %", agreement_medium, 0.19, tol=0.15, fmt=".2f")

    check("Long first-order δ|M|", pred_long, -0.01557, tol=0.0002, fmt=".5f")
    check("Long observed δ|M|", obs_long, -0.01559, tol=0.0002, fmt=".5f")

    agreement_long = abs(pred_long - obs_long) / abs(obs_long) * 100
    check("Long self-consistency %", agreement_long, 0.11, tol=0.15, fmt=".2f")

    # Second-order shape terms (Σ_{i<j} convention)
    shape_medium = second_order_shape_term(delta_d_medium)
    shape_long = second_order_shape_term(delta_d_long)
    check("Medium 2nd-order shape", shape_medium, 0.000024, tol=0.000005, fmt=".6f")
    check("Long 2nd-order shape", shape_long, 0.000007, tol=0.000003, fmt=".6f")

    shape_pct_med = abs(shape_medium) / abs(obs_medium) * 100
    shape_pct_long = abs(shape_long) / abs(obs_long) * 100
    check("Medium shape < 0.3% of signal", float(shape_pct_med < 0.3), 1.0, tol=0)
    check("Long shape < 0.3% of signal", float(shape_pct_long < 0.3), 1.0, tol=0)

    # Mode decomposition
    decomp_medium = mode_decomposition(delta_d_medium)
    check("Medium shearing variance fraction", decomp_medium["shearing_variance_fraction"], 0.733, tol=0.005, fmt=".3f")
    check("Breathing perimeter = total perimeter (medium)",
          decomp_medium["breathing_perimeter"], dp_medium, tol=1e-10, fmt=".6f")
    check("Shearing perimeter = 0 (medium)",
          abs(decomp_medium["shearing_perimeter"]), 0.0, tol=1e-10, fmt=".6f")

    decomp_long = mode_decomposition(delta_d_long)
    check("Breathing perimeter = total perimeter (long)",
          decomp_long["breathing_perimeter"], dp_long, tol=1e-10, fmt=".6f")
    check("Shearing perimeter = 0 (long)",
          abs(decomp_long["shearing_perimeter"]), 0.0, tol=1e-10, fmt=".6f")

    # Worst-case bound
    delta_Z_medium = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                delta_Z_medium[i, j] = -np.exp(-D_baseline[i, j]) * (D_medium[i, j] - D_baseline[i, j])
    norm_dZ = np.linalg.norm(delta_Z_medium, ord=2)
    bound = worst_case_bound_coefficient(3) * norm_dZ
    check("Worst-case bound ≈ 0.19", bound, 0.19, tol=0.03, fmt=".3f")
    check("Observed << bound", float(abs(obs_medium) < bound), 1.0, tol=0)


def check_section62_sycophancy(data):
    section("Section 6.2: Sycophancy / Reconstitution")

    recon = data["reconstitution"]
    if recon is None:
        warn("Section 6.2", "reconstitution data not available")
        return

    d_hd = recon["distance_matrices"]["pairwise"]["healthy_vs_degraded"]["sqrt_jsd"]
    check("√JSD(healthy, degraded)", d_hd, 0.6233, tol=0.001)

    pct_max = d_hd / SQRT_LN2 * 100
    check("% of maximum √(ln 2)", pct_max, 74.9, tol=0.1, fmt=".1f")

    d_rs = recon["distance_matrices"]["pairwise"]["healthy_vs_reconstituted_system"]["sqrt_jsd"]
    d_ri = recon["distance_matrices"]["pairwise"]["healthy_vs_reconstituted_incontext"]["sqrt_jsd"]
    check("√JSD(healthy, recon_system)", d_rs, 0.6233, tol=0.001)
    check("√JSD(healthy, recon_incontext)", d_ri, 0.6233, tol=0.001)

    # Recompute from samples
    samples = recon["samples"]["Q3"]
    dist_healthy = empirical_distribution(samples["healthy"])
    dist_degraded = empirical_distribution(samples["degraded"])
    computed_d = sqrt_jsd(dist_healthy, dist_degraded)
    check("√JSD recomputed from samples", computed_d, d_hd, tol=0.001)


def check_section55_cc_homology(data):
    section("Section 5.5: Cross-Condition Homology")

    cc = data["canonical_7probe"]
    if cc is None:
        warn("Section 5.5", "canonical 7-probe data not available")
        return

    mag_7pt = cc["seven_point"]["scalar_magnitude"]
    mag_6pt = cc["six_point"]["scalar_magnitude"]

    # §5.5 says both 7-pt and 6-pt yield 1.2779
    check("§5.5 |M|_cc (7-point) ≈ 1.2779", mag_7pt, 1.2779, tol=0.001)
    check("§5.5 |M|_cc (6-point) ≈ 1.2779", mag_6pt, 1.2779, tol=0.001)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify paper numbers against data")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    data_dir = os.path.abspath(data_dir)
    print(f"Data directory: {data_dir}")
    print(f"√(ln 2) = {SQRT_LN2:.10f}")

    data = load_data(data_dir)

    check_section2_framework(data)
    check_section4_entropy(data)
    check_section52_equilateral(data)
    check_section53_cross_condition(data)
    check_section54_three_probe_cc(data)
    check_section56_drift(data)
    check_bootstrap(data)
    check_perturbation_theory(data)
    check_section62_sycophancy(data)
    check_section55_cc_homology(data)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {PASS} passed, {WARN} warnings, {FAIL} failures")
    if FAIL > 0:
        print("EXIT: FAIL")
    else:
        print("EXIT: PASS (all checks passed)")
    print(f"{'=' * 60}")

    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
