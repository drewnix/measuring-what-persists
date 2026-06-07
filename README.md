# Measuring What Persists

Reproducibility package for:

> **Conditioning Mechanisms and a Geometric Framework for AI Agent Identity**
> and Andrew Tanner / Anisotrope AI

**[Read the paper (PDF)](measuring-what-persists.pdf)**

## Quick start

```bash
pip install -r requirements.txt
python paper_table_check.py
```

This runs 134 independent checks verifying every computable number in the paper against the experimental data. Expected output: all checks pass.

## Contents

```
measuring-what-persists/
├── measuring-what-persists.pdf   # The paper
├── paper_table_check.py          # Verify all paper numbers (run this first)
├── requirements.txt              # numpy, scipy, matplotlib, jupyter
├── data/
│   ├── README.md                 # Schema documentation
│   ├── drift_experiment.json     # Primary drift experiment (k=50, 3 probes, 3 lengths)
│   ├── null_model_control.json   # Equilateral control (k=50 and k=200)
│   ├── cross_condition_canonical.json  # 7-probe cross-condition battery
│   ├── cross_condition_3probe.json     # 3-probe operational text distances
│   ├── bootstrap_validation.json       # Bootstrap CIs (1,000 resamples)
│   └── reconstitution.json            # Sycophancy/reconstitution experiment
├── src/
│   ├── distances.py              # sqrt(JSD) computation
│   ├── magnitude.py              # Scalar magnitude from distance matrix
│   ├── homology.py               # Magnitude homology groups
│   ├── perturbation.py           # Perturbation theory (eigenvalues, predictions, mode decomposition)
│   └── bootstrap.py              # Bootstrap resampling for magnitude and distance CIs
└── notebooks/                    # (see below)
```

## Notebooks

Three Jupyter notebooks reproduce the paper's results with annotations:

| Notebook | Covers | Key verifications |
|----------|--------|-------------------|
| `01_reproduce_distances` | Tables 1-5 | All pairwise distances, null-model control, cross-condition cc1 values |
| `02_magnitude_and_perturbation_theory` | §5.2-5.6, Appendix A | Magnitude, eigenvalues, first-order self-consistency (0.19%/0.11%), mode decomposition (73.3% shearing), betweenness |
| `03_bootstrap_validation` | §5.6 Tables 6-7 | Magnitude bootstrap CIs, CI upper < baseline, signal-to-bias ratios (2.2x-11.7x) |

## What the check script verifies

The script checks every number that appears in the paper and can be computed from the data:

- **Distances:** All within-condition, cross-condition, and null-model distances
- **Magnitude:** Baseline (1.6044), medium (1.5875), long (1.5888), cross-condition (1.2779, 1.4098)
- **Eigenvalues:** Analytic eigenvalues of the equilateral similarity matrix
- **Perturbation theory:** First-order estimates, self-consistency percentages, second-order shape terms, mode decomposition
- **Bootstrap:** CIs on magnitude and individual distances, signal-to-bias ratios
- **Reconstitution:** sqrt(JSD) = 0.6233 (74.9% of maximum)

The script does NOT verify theoretical proofs (sqrt(JSD) canonicity, equilateral critical-point proposition). Those require mathematical peer review, not computation.

## Dependencies

- Python 3.10+
- numpy >= 1.24
- scipy >= 1.10
- matplotlib >= 3.7 (for notebooks only)
- jupyter >= 1.0 (for notebooks only)

No Anthropic API access is required. All computations use the pre-collected experimental data.
