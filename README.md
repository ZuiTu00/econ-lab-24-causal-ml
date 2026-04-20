# econ-lab-24-causal-ml
# Causal ML — DML & Causal Forests for 401(k) Policy Evaluation


> Estimate the causal effect of 401(k) eligibility on household net financial assets using
> Double Machine Learning (Chernozhukov et al. 2018) and Causal Forests (Wager & Athey 2018).
> Recovers an ATE of **≈ $8,821** and reveals strong individual-level treatment effect
> heterogeneity that aggregate statistics obscure.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#-key-findings)
- [Repository Structure](#-repository-structure)
- [Installation](#️-installation)
- [Quick Start](#-quick-start)
- [Methodology](#-methodology)
- [Results](#-results)
- [Specification Diagnosis: Bad Controls](#-specification-diagnosis-bad-controls)
- [Reusable Module](#-reusable-module-srccausal_mlpy)
- [References](#-references)

---

## Overview

This project applies modern causal machine learning to the canonical
**401(k) Pension Data** (Chernozhukov & Hansen 2004, 9,915 households),
answering two distinct research questions:

1. **What is the average causal effect** of 401(k) eligibility on net financial assets?
   → Answered via **Double Machine Learning (DML)** with sensitivity analysis.
2. **For whom is this policy most effective?**
   → Answered via **Causal Forests** estimating individual-level CATEs.

The project also contains a diagnostic exercise: identifying and repairing three
subtle bugs in a manual cross-fitted DML pipeline, illustrating why each of
Chernozhukov et al.'s three pillars (cross-fitting, double residualization,
and the orthogonal IV-style moment) is individually necessary.

---

## 🔑 Key Findings

| Metric | Value | Interpretation |
|---|---|---|
| **ATE (DML, preferred spec)** | **$8,821** | 95% CI: [$6,168, $11,475], p < 1e-10 |
| **Robustness Value (RV)** | 6.8% | Moderate robustness to unobserved confounding |
| **Mean CATE (Causal Forest)** | $7,949 | Close to ATE — mutual validation |
| **CATE Std. Dev.** | $6,659 | Heterogeneity ≈ level effect itself |
| **High-response mean CATE** | $17,588 | Top 25% — 3.7× the low-response group |
| **Specification bias** | **$9,667** | Swing from including post-treatment controls |

### One-Sentence Takeaway

> "DML delivers a credible aggregate policy parameter, but **it does not substitute
> for identification thinking**: a naïve covariate choice flips the sign of the
> estimated effect. Causal Forest reveals that income accounts for most — but
> not all — of the heterogeneity, with within-quartile dispersion growing
> monotonically with income."

---

## 📂 Repository Structure

```
econ-lab-24-causal-ml/
├── README.md                      ← this file
├── verification-log.md            ← diagnostic record of all 3 bugs + spec test
│
├── notebooks/
│   └── lab_24_causal_ml.ipynb    ← full analysis notebook
│
├── src/
│   └── causal_ml.py              ← reusable module (manual_dml, cate_by_subgroup)
│
└── figures/
    ├── cate_histogram.png        ← distribution of 9,915 individual CATEs
    ├── sensitivity_plot.png      ← Cinelli-Hazlett contour plot
    └── cf_vs_subgroup_dml.png    ← violin plot: CF vs subgroup DML by quartile
```

---

## ⚙️ Installation

```bash
# Clone
git clone https://github.com/<your-username>/econ-lab-24-causal-ml.git
cd econ-lab-24-causal-ml

# Install dependencies
pip install doubleml econml scikit-learn pandas matplotlib kaleido
```

**Requirements:**
- Python ≥ 3.10
- DoubleML ≥ 0.8
- EconML ≥ 0.15
- scikit-learn ≥ 1.3

---

## 🚀 Quick Start

### Run the full analysis

```bash
jupyter notebook notebooks/lab_24_causal_ml.ipynb
```

### Use the reusable module

```python
from src.causal_ml import manual_dml, cate_by_subgroup, heterogeneity_decomposition

# 1. Manual DML with cross-fitting (pedagogical)
result = manual_dml(Y, D, X, n_folds=5)
print(f"ATE = {result.theta:.2f}, 95% CI = [{result.ci_low:.2f}, {result.ci_high:.2f}]")

# 2. Summarize CATEs across income quartiles
summary = cate_by_subgroup(
    cate_predictions, data,
    group_col='inc', quantiles=4,
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)
print(summary)

# 3. Decompose heterogeneity
decomp = heterogeneity_decomposition(cate_predictions, income_quartile_labels)
print(f"Between-SD: {decomp['between_sd']:.0f}")
print(f"Within-SD:  {decomp['within_sd']:.0f}")
```

---

## 🔬 Methodology

### Stage 1 — Manual DML Diagnostic (Part A)

Identified and fixed three pedagogical bugs in a 2-fold cross-fitting pipeline:

| Bug | Diagnosis | Theoretical Pillar Violated |
|---|---|---|
| **1. In-sample prediction** | `predict(X[train_idx])` instead of test fold | Cross-fitting (Chernozhukov 2018) |
| **2. Missing D residualization** | Used raw D in moment condition | Frisch-Waugh-Lovell theorem |
| **3. Wrong θ formula** | `np.mean(·)` instead of IV-style ratio | Orthogonal moment condition |

Corrected estimator: `θ̂ = Σ(D̃·Ỹ) / Σ(D̃·D)`, recovers true ATE = 5.0 on simulated DGP.

### Stage 2 — Production DML (Part B)

- **Model**: Partially Linear Regression (PLR)
- **Nuisance learners**: RandomForestRegressor (`n_estimators=200`, `max_depth=5`)
- **Cross-fitting**: 5-fold
- **Score**: Partialling-out (default orthogonal moment)
- **Sensitivity analysis**: Cinelli & Hazlett (2020) at `cf_y = cf_d = 0.03`

### Stage 3 — Causal Forests (Part C)

- **Model**: `CausalForestDML` (EconML)
- **Forest size**: 500 causal trees
- **Leaf size**: `min_samples_leaf = 20` (honesty requirement)
- **Max depth**: 10
- **DML cross-fitting**: `cv = 5`

### Stage 4 — DML Subgroup vs Causal Forest (Extension)

ANOVA-style decomposition of CATE variance into **between-quartile** and
**within-quartile** components, comparing the granularity of two heterogeneity
representations.

---

## 📊 Results

### Average Treatment Effect (Part B)

| Specification | Covariates | ATE | 95% CI | p-value |
|---|---|:-:|:-:|:-:|
| **Spec 1 (preferred)** | 9 pre-treatment | **$8,821** | [$6,168, $11,475] | < 1e-10 |
| Spec 2 (lab default) | all 12 | -$845 | [-$1,775, $85] | 0.075 |

**Divergence = $9,667** ← This swing is the signature of bad-control bias.

### Sensitivity Analysis (Spec 1)

```
Benchmark: cf_y = 0.03, cf_d = 0.03

                    CI lower   theta lower   theta    theta upper   CI upper
Worst-case bounds:   $2,701      $5,000      $8,821     $12,642      $14,885

Robustness Value (RV):      6.8%
Robustness Value for CI:    5.0%
```

Interpretation: An unobserved confounder would need to explain ≥ **6.8%** of
the residual variation in both `net_tfa` and `e401` simultaneously to drive
the point estimate to zero. The estimate is **moderately robust**.

![Sensitivity Plot](figures/sensitivity_plot.png)

### Individual Treatment Effect Distribution (Part C)

```
Mean CATE:    $7,949
Median CATE:  right-skewed → median < mean
Std CATE:     $6,659
Min CATE:    -$10,317       (some negative effects)
Max CATE:    $48,190        (long right tail)
```

![CATE Histogram](figures/cate_histogram.png)

### High-Response Subgroup Profile (top 25% of CATE, n = 2,479)

| Characteristic | High-Resp | Low-Resp | Difference |
|---|:-:|:-:|:-:|
| Age | 44.0 | 40.1 | +3.9 years |
| Income | $65,461 | $27,779 | **+136%** |
| Education | 14.2 yr | 12.9 yr | +1.3 yr |
| Married | 83% | 53% | +30 pp |
| Dual-earner | 63% | 30% | +33 pp |
| Has IRA | 46% | 17% | **+174%** |
| Homeowner | 87% | 56% | +31 pp |
| **Mean CATE** | **$17,588** | **$4,735** | **3.7×** |

The profile is consistent with the **life-cycle hypothesis** (Modigliani 1966):
households with both the *capacity* (income, stability) and the *motive*
(long horizon, existing financial engagement) to save respond most strongly.

### DML Subgroup vs Causal Forest (Extension)

| Income Quartile | DML subgroup ATE | CF mean CATE | CF std (within) |
|---|:-:|:-:|:-:|
| Q1 (Lowest) | $4,276 | $3,352 | $2,121 |
| Q2 | $3,330 | $4,188 | $2,315 |
| Q3 | $6,674 | $7,509 | $3,558 |
| Q4 (Highest) | $16,980 | $16,747 | **$6,481** |

**Variance decomposition:**
- Between-quartile SD: **$5,312**
- Within-quartile SD: **$4,016**
- Ratio: **0.76**

![DML vs CF Comparison](figures/cf_vs_subgroup_dml.png)

**Finding:** Income explains the majority of heterogeneity at the quartile
level, yet within-quartile dispersion grows monotonically with income — from
$2,121 in Q1 to $6,481 in Q4. Subgroup DML is approximately sufficient for
first-order policy evaluation; Causal Forest is necessary for individual
targeting, especially in the high-income segment where effects are largest
*and* most variable.

---

## 🚨 Specification Diagnosis: Bad Controls

A core empirical finding of this project is the collapse in the ATE estimate
when `p401`, `nifa`, and `tw` are included as controls:

```
Spec 1 (pre-treatment only):   ATE =  $8,821  ✓  (literature-consistent)
Spec 2 (all available vars):   ATE = -$845    ✗  (wrong sign, wrong magnitude)
```

**Root causes:**

1. **`p401` is post-treatment.** It sits on the causal path
   `e401 → p401 → net_tfa`. Conditioning on a mediator partials out the very
   channel we want to measure (Angrist & Pischke 2009, Ch. 3.2.3).
2. **`nifa` and `tw` overlap definitionally with the outcome.** Since
   `net_tfa ≈ nifa + 401(k) assets`, these variables absorb rather than control
   for variation in Y.

This illustrates a general principle: **Double Machine Learning corrects
function-form and regularization bias, but it cannot fix an identification
strategy that is wrong by construction.** Covariate choice is a substantive
decision, not a data-driven one.

---

## 📦 Reusable Module (`src/causal_ml.py`)

The `causal_ml` module provides three functions, all fully type-hinted and
documented:

### `manual_dml(Y, D, X, ml_l=None, ml_m=None, n_folds=5)`
Pedagogical manual-cross-fitted DML estimator. Returns a `DMLResult` with
point estimate, standard error, 95% CI, and the cross-fitted residuals for
diagnostic inspection.

### `cate_by_subgroup(cate, data, group_col, quantiles=None, labels=None)`
Aggregates individual CATEs across quantile bins or custom group labels,
returning count, mean, std, and percentile summaries.

### `heterogeneity_decomposition(cate, group_labels)`
Decomposes CATE dispersion into between-group and within-group components.
Returns a dict with `between_sd`, `within_sd`, and their ratio.

All three functions are tested on simulated DGPs and validated against
`doubleml` and `econml` implementations.

---

## 📚 References

**Methodological:**

1. Chernozhukov, V. et al. (2018). ["Double/Debiased Machine Learning for Treatment and Structural Parameters."](https://doi.org/10.1111/ectj.12097) *Econometrics Journal*, 21(1), C1–C68.
2. Wager, S., & Athey, S. (2018). ["Estimation and Inference of Heterogeneous Treatment Effects using Random Forests."](https://doi.org/10.1080/01621459.2017.1319839) *JASA*, 113(523).
3. Cinelli, C., & Hazlett, C. (2020). ["Making Sense of Sensitivity: Extending Omitted Variable Bias."](https://doi.org/10.1111/rssb.12348) *JRSS: Series B*, 82(1), 39–67.

**Applied (401(k) literature):**

4. Chernozhukov, V., & Hansen, C. (2004). "The Effects of 401(k) Participation on the Wealth Distribution." *Review of Economics and Statistics*, 86(3), 735–751.
5. Poterba, J., Venti, S., & Wise, D. (1995). "Do 401(k) Contributions Crowd Out Other Personal Saving?" *Journal of Public Economics*, 58(1), 1–32.
6. Chetty, R. et al. (2014). "Active vs. Passive Decisions and Crowd-Out in Retirement Savings Accounts." *Quarterly Journal of Economics*, 129(3).

**Identification theory:**

7. Angrist, J., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. Ch. 3.2.3 ("Bad Controls").

---

## 📄 License

MIT — free to use, modify, and distribute with attribution.

## 🙏 Acknowledgements

Lab assignment from **ECON 5200: Causal Machine Learning & Applied Analytics**.
Data from the [DoubleML](https://docs.doubleml.org/) and
[EconML](https://econml.azurewebsites.net/) project maintainers.

---

<div align="center">
<sub>Built with DoubleML, EconML, and a healthy respect for bad-control bias.</sub>
</div>
