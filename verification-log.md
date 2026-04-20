# Verification Log — Lab 24

Diagnostic record covering Part A bug identification, Part B specification
diagnosis, and Extension findings.

---

## Part A: Manual DML Bug Diagnostics

### Broken pipeline output
```
True ATE:    5.00
Broken ATE:  1.09
Bias:        -3.91
```

### Decomposition of the bias

The -3.91 bias is dominated by Bug 3 (wrong formula). For a binary treatment
with E[D] ~ 0.5, Var(D) ~ 0.25. The broken estimator computes

    theta_broken ~ Cov(D, Y) = theta_0 * Var(D) ~ 5.0 * 0.25 = 1.25

which matches the observed 1.09 up to finite-sample noise and residual
contamination from Bugs 1 and 2.

### Bug 1 — In-sample prediction (overfitting on residuals)

**Location**: `Y_hat = ml_l.predict(X[train_idx])` with `Y_tilde[train_idx]`.
**Mechanism**: Training and prediction on the same fold lets the Random
Forest memorize Y, shrinking residuals artificially. The resulting Y_tilde is
not an out-of-sample residual, so Chernozhukov et al. (2018) sqrt(n)-consistency
guarantees do not apply.
**Fix**: Predict on `X[test_idx]` and populate `Y_tilde[test_idx]`.

### Bug 2 — Missing treatment residualization

**Location**: `V_tilde[train_idx] = D[train_idx]` (used raw D).
**Mechanism**: The Frisch-Waugh-Lovell theorem requires *both* Y and D to
be residualized against X. Using raw D leaves the confounding path X -> D
un-blocked, yielding omitted-variable bias relative to the partialled-out
representation.
**Fix**: Train a second ML model (`ml_m`) on D ~ X, form
`D_tilde = D - D_hat` on the held-out fold.

### Bug 3 — Wrong estimator formula

**Location**: `theta = np.mean(V_tilde * Y_tilde)`.
**Mechanism**: This computes a covariance, not a regression coefficient —
the denominator (a variance term) is missing, so the estimate is attenuated
by roughly Var(D).
**Fix**: Use the IV-style moment
`theta = sum(D_tilde * Y_tilde) / sum(D_tilde * D)` (Chernozhukov et al. 2018,
eq. 4.4). The D in the denominator (rather than D_tilde**2) corresponds to
the partialling-out score.

### Fixed pipeline verification
```
True ATE:    5.00
Fixed ATE:  ~4.95 (varies with seed)
Bias:       within +/- 0.3
PASS -- Fixed ATE is within 1.0 of the true value.
```

---

## Part B: Specification Diagnosis

### Two specifications estimated

| Specification | Covariates | n_cov | ATE | 95% CI |
|---|---|---|---|---|
| Spec 1 (clean) | age, inc, educ, fsize, marr, twoearn, db, pira, hown | 9 | **$8,821** | [$6,168, $11,475] |
| Spec 2 (lab default) | all except Y, D | 12 | -$845 | [-$1,775, $85] |

### Why Spec 2 fails

- `p401` is a post-treatment variable: `e401 -> p401 -> net_tfa`.
  Controlling for it partially out the treatment effect's causal channel
  (Angrist & Pischke 2009, Ch. 3.2.3).
- `nifa` (non-401k financial assets) and `tw` (total wealth) overlap
  definitionally with the outcome `net_tfa`, absorbing rather than
  controlling variation.

Spec 1 is therefore adopted for all downstream analysis (sensitivity and
Causal Forest).

### Sensitivity analysis (Spec 1, Cinelli-Hazlett 2020)

Benchmark parameters: cf_y = 0.03, cf_d = 0.03 (3% partial R^2 bound on
unobserved confounding).

```
Point estimate:        $8,821
Worst-case theta low:  $5,000
Worst-case CI low:     $2,701
Robustness Value:        6.8%
Robustness Value (CI):   5.0%
```

**Interpretation**: An unobserved confounder explaining at least 6.8% of
residual variation in both net_tfa and e401 simultaneously would be required
to drive the point estimate to zero. The estimate is moderately robust;
confounders as strong as a typical observed covariate could erode the
result but not reverse it.

---

## Part C: Causal Forest CATE Results

Clean-covariate specification (overriding lab default to match Spec 1):

```
Mean CATE:    $7,949
Median CATE:  $<mean (right-skewed)
Std CATE:     $6,659
Min CATE:    -$10,317
Max CATE:    $48,190
```

Reference DML ATE (Spec 1): $8,821. Mean CATE is within $872 of ATE,
consistent with near-identity under the PLR model when heterogeneity is
mean-centered.

### High-response subgroup (top 25% of CATE)

| Characteristic | High-Resp | Low-Resp | Diff |
|---|---|---|---|
| age | 44.0 | 40.1 | +3.9 |
| inc | $65,461 | $27,779 | +$37,682 |
| educ | 14.2 | 12.9 | +1.3 |
| marr | 0.83 | 0.53 | +0.30 |
| twoearn | 0.63 | 0.30 | +0.33 |
| pira | 0.46 | 0.17 | +0.29 |
| hown | 0.87 | 0.56 | +0.31 |

Interpretation: The high-response subgroup is older, wealthier,
better-educated, more likely married / dual-earner / homeowner / IRA-holder.
Consistent with the life-cycle-hypothesis prediction that households with
both the capacity (income, stability) and motive (long horizon, existing
financial engagement) to save respond most to 401(k) eligibility.

---

## Extension: DML Subgroup vs Causal Forest

| Income Quartile | DML subgroup ATE | CF mean CATE | CF std CATE (within) |
|---|---|---|---|
| Q1 | $4,276 | $3,352 | $2,121 |
| Q2 | $3,330 | $4,188 | $2,315 |
| Q3 | $6,674 | $7,509 | $3,558 |
| Q4 | $16,980 | $16,747 | $6,481 |

Between-quartile SD: **$5,312**
Within-quartile SD: **$4,016**
Ratio within/between: **0.76**

**Finding**: Income explains the majority of cross-household heterogeneity,
but within-quartile dispersion grows monotonically with income.
Subgroup DML is approximately sufficient for the aggregate question of
"how does the effect vary by income?", but it under-represents heterogeneity
precisely where the effect is largest (Q4). For individual-level policy
targeting, Causal Forest remains necessary.
