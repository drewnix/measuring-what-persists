# Data

Experimental data for *Measuring What Persists: Magnitude Homology as a Structural Fingerprint for LLM Identity Drift*.

All experiments use Claude Sonnet (`claude-sonnet-4-6`) via the Anthropic API. Distributions are estimated empirically from `k` samples at `max_tokens=10`, `temperature=1.0`.

## Files

| File | Description | Key contents |
|------|-------------|--------------|
| `drift_experiment.json` | Primary drift experiment | k=50 samples for 3 probes (Q1, Q3, B4) at 3 context lengths (baseline, medium, long), both Card-conditioned and base-model. Raw prefix strings, distance matrices, homology, drift analysis. |
| `null_model_control.json` | Equilateral control | k=50 and k=200 for both conditions. Verifies equilateral structure is a probe-design property with zero bootstrap variance. |
| `cross_condition_canonical.json` | 7-probe cross-condition battery | cc1 values for all 7 probes, 6-point and 7-point distance matrices, betweenness, homology. |
| `cross_condition_3probe.json` | 3-probe cross-condition (operational texts) | cc1 and cc10 distances for Q1, Q3, B4. Source data for the collinear structure analysis. |
| `bootstrap_validation.json` | Bootstrap CIs | 1,000-resample bootstrap on distances: 95% CIs, bias, SE for all probe pairs at medium and long context. |
| `reconstitution.json` | Sycophancy/reconstitution experiment | Q3 probe under 4 conditions: healthy, degraded (10-turn sycophantic pressure), reconstituted (system), reconstituted (in-context). k=50, max_tokens=1. |

## Schema notes

- **Distance matrices** in `drift_experiment.json` are stored as nested dicts keyed by probe name: `distance_matrices[condition][card_or_base][probe_i][probe_j]`.
- **Distance matrices** in `null_model_control.json` are stored as 3x3 nested lists.
- **Samples** are raw 10-token prefix strings (or single tokens for reconstitution).
- **cc1 values** are cross-condition distances: sqrt(JSD) between Card-conditioned and base-model first-token distributions.
- All distances use sqrt(JSD) as the metric. Maximum possible value is sqrt(ln 2) = 0.8326.
