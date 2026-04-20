"""
causal_ml.py
============

Reusable utilities for Double Machine Learning (DML) and heterogeneous
treatment effect analysis.

Implements the Partially Linear Regression (PLR) DML estimator from
Chernozhukov et al. (2018) "Double/Debiased Machine Learning for
Treatment and Structural Parameters," Econometrics Journal.

Two exported functions:
    manual_dml          - pedagogical manual cross-fitting DML estimator
    cate_by_subgroup    - aggregate individual CATE estimates into
                          subgroup summaries for policy analysis

Author: ECON 5200 — Lab 24
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


# =============================================================================
# Return containers
# =============================================================================

@dataclass
class DMLResult:
    """
    Container for manual DML estimation results.

    Attributes
    ----------
    theta : float
        Point estimate of the average treatment effect (ATE).
    se : float
        Asymptotic standard error (heteroskedasticity-robust, based on
        the orthogonal score function).
    ci_low, ci_high : float
        Lower and upper bounds of the 95 percent confidence interval.
    y_residuals, d_residuals : np.ndarray
        Cross-fitted residuals; diagnostic use only.
    n_folds : int
        Number of cross-fitting folds used.
    """
    theta: float
    se: float
    ci_low: float
    ci_high: float
    y_residuals: np.ndarray
    d_residuals: np.ndarray
    n_folds: int

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DMLResult(theta={self.theta:.3f}, se={self.se:.3f}, "
            f"95% CI=[{self.ci_low:.3f}, {self.ci_high:.3f}], "
            f"n_folds={self.n_folds})"
        )


# =============================================================================
# Main estimator
# =============================================================================

def manual_dml(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    ml_l: Optional[BaseEstimator] = None,
    ml_m: Optional[BaseEstimator] = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> DMLResult:
    """
    Manual Double Machine Learning estimator for the Partially Linear Regression model.

    Model
    -----
    Y = theta * D + g_0(X) + eps,     E[eps | X, D] = 0
    D = m_0(X) + nu,                  E[nu  | X]    = 0

    Estimator (Chernozhukov et al. 2018, IV-style moment):

        theta_hat = sum(D_tilde * Y_tilde) / sum(D_tilde * D)

    where Y_tilde and D_tilde are CROSS-FITTED residuals from the nuisance
    learners ml_l (for E[Y|X]) and ml_m (for E[D|X]).

    Parameters
    ----------
    Y : np.ndarray, shape (n,)
        Outcome variable.
    D : np.ndarray, shape (n,)
        Treatment variable (binary or continuous).
    X : np.ndarray, shape (n, p)
        Covariate matrix.
    ml_l : sklearn estimator, optional
        Learner for E[Y|X]. Defaults to RandomForestRegressor.
    ml_m : sklearn estimator, optional
        Learner for E[D|X]. Defaults to RandomForestRegressor.
    n_folds : int, default 5
        Number of cross-fitting folds. Chernozhukov et al. (2018)
        recommend 4 to 5 in practice.
    random_state : int, default 42
        Seed for KFold shuffling; passed through to default learners.

    Returns
    -------
    DMLResult
        Point estimate, standard error, 95% CI, and cross-fitted residuals.

    Notes
    -----
    - Each fold trains on its own subsample (n - n/K observations) and
      predicts on the held-out fold, so residuals are strictly
      out-of-sample. This implements the cross-fitting required for
      sqrt(n)-consistency when ML nuisance estimators converge at rates
      slower than sqrt(n).
    - Both Y and D are residualized (Frisch-Waugh-Lovell applied to
      the orthogonal score).
    - The IV-style denominator sum(D_tilde * D) improves numerical
      stability when propensity estimation is accurate relative to
      the plain OLS denominator sum(D_tilde**2).

    References
    ----------
    Chernozhukov, V. et al. (2018). "Double/Debiased Machine Learning
    for Treatment and Structural Parameters." Econometrics Journal.
    """
    Y = np.asarray(Y).ravel()
    D = np.asarray(D).ravel()
    X = np.asarray(X)
    n = len(Y)

    if ml_l is None:
        ml_l = RandomForestRegressor(
            n_estimators=200, max_depth=5, random_state=random_state
        )
    if ml_m is None:
        ml_m = RandomForestRegressor(
            n_estimators=200, max_depth=5, random_state=random_state
        )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    y_tilde = np.zeros(n)
    d_tilde = np.zeros(n)

    for train_idx, test_idx in kf.split(X):
        ml_l_fold = clone(ml_l)
        ml_m_fold = clone(ml_m)

        # Outcome nuisance: fit on train fold, predict on test fold
        ml_l_fold.fit(X[train_idx], Y[train_idx])
        y_tilde[test_idx] = Y[test_idx] - ml_l_fold.predict(X[test_idx])

        # Treatment nuisance: same structure
        ml_m_fold.fit(X[train_idx], D[train_idx])
        d_tilde[test_idx] = D[test_idx] - ml_m_fold.predict(X[test_idx])

    # IV-style moment (Chernozhukov et al. 2018, eq. 4.4)
    theta = float(np.sum(d_tilde * y_tilde) / np.sum(d_tilde * D))

    # Asymptotic variance from the orthogonal score
    psi = (y_tilde - theta * d_tilde) * d_tilde
    jacobian = np.mean(d_tilde * D)
    var_theta = np.mean(psi ** 2) / (jacobian ** 2) / n
    se = float(np.sqrt(var_theta))

    return DMLResult(
        theta=theta,
        se=se,
        ci_low=theta - 1.96 * se,
        ci_high=theta + 1.96 * se,
        y_residuals=y_tilde,
        d_residuals=d_tilde,
        n_folds=n_folds,
    )


# =============================================================================
# Subgroup aggregation for CATE analysis
# =============================================================================

def cate_by_subgroup(
    cate: np.ndarray,
    data: pd.DataFrame,
    group_col: Union[str, pd.Series],
    quantiles: Optional[int] = None,
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Summarize individual CATE estimates across a subgroup definition.

    Useful for comparing coarse (subgroup-DML) and fine (Causal Forest)
    heterogeneity representations: the mean CATE within each bin
    corresponds conceptually to a subgroup ATE, while the standard
    deviation quantifies within-bin heterogeneity that subgroup DML
    would average away.

    Parameters
    ----------
    cate : np.ndarray, shape (n,)
        Individual conditional average treatment effect estimates,
        typically from CausalForestDML.effect().
    data : pd.DataFrame
        Observations aligned with cate (same row order and length).
    group_col : str or pd.Series
        Column name in data used for grouping, or a Series of labels.
        If quantiles is given, group_col must be a numeric column.
    quantiles : int, optional
        If provided, cut group_col into this many quantile bins
        (e.g., 4 for quartiles). Otherwise group_col is used as-is.
    labels : sequence of str, optional
        Labels for the quantile bins (length must equal quantiles).

    Returns
    -------
    pd.DataFrame
        Rows indexed by group; columns include count, mean, std,
        min, max, and the 25th/50th/75th percentiles of CATE within
        the group.

    Examples
    --------
    >>> summary = cate_by_subgroup(
    ...     cate_predictions, data, 'inc', quantiles=4,
    ...     labels=['Q1', 'Q2', 'Q3', 'Q4']
    ... )
    """
    cate = np.asarray(cate).ravel()
    if len(cate) != len(data):
        raise ValueError(
            f"Length mismatch: cate has {len(cate)} rows but data has "
            f"{len(data)} rows."
        )

    df = data.copy()
    df['_cate'] = cate

    if quantiles is not None:
        if not isinstance(group_col, str):
            raise TypeError(
                "When 'quantiles' is given, 'group_col' must be a column name."
            )
        df['_group'] = pd.qcut(df[group_col], q=quantiles, labels=labels)
        group_key = '_group'
    else:
        if isinstance(group_col, str):
            group_key = group_col
        else:
            df['_group'] = pd.Series(group_col).values
            group_key = '_group'

    summary = df.groupby(group_key, observed=True)['_cate'].agg([
        'count', 'mean', 'std', 'min', 'max',
        ('p25', lambda s: s.quantile(0.25)),
        ('p50', lambda s: s.quantile(0.50)),
        ('p75', lambda s: s.quantile(0.75)),
    ])
    summary.columns = [
        'n', 'mean_cate', 'std_cate', 'min_cate', 'max_cate',
        'p25_cate', 'p50_cate', 'p75_cate',
    ]
    return summary.round(0)


# =============================================================================
# Variance decomposition utility
# =============================================================================

def heterogeneity_decomposition(
    cate: np.ndarray,
    group_labels: Union[np.ndarray, pd.Series],
) -> dict:
    """
    Decompose CATE dispersion into between-group and within-group
    components.

    Between-group variance is Var of group means; within-group variance
    is the mean of within-group variances. Their sum equals total
    variance by the law of total variance when groups have equal size.

    A within/between ratio above 1 indicates that subgroup averaging
    (e.g., subgroup DML by income quartile) misses more heterogeneity
    than it captures.

    Parameters
    ----------
    cate : np.ndarray, shape (n,)
        Individual CATE estimates.
    group_labels : array-like, shape (n,)
        Group assignment for each observation.

    Returns
    -------
    dict
        Keys: 'between_sd', 'within_sd', 'total_sd', 'ratio_within_between'.
    """
    cate = np.asarray(cate).ravel()
    labels = pd.Series(group_labels).reset_index(drop=True)
    df = pd.DataFrame({'cate': cate, 'group': labels})

    group_means = df.groupby('group', observed=True)['cate'].mean()
    group_vars = df.groupby('group', observed=True)['cate'].var(ddof=0)

    between_var = float(np.var(group_means.values, ddof=0))
    within_var = float(np.mean(group_vars.values))

    return {
        'between_sd': float(np.sqrt(between_var)),
        'within_sd': float(np.sqrt(within_var)),
        'total_sd': float(np.std(cate, ddof=0)),
        'ratio_within_between': float(np.sqrt(within_var / between_var))
            if between_var > 0 else float('inf'),
    }


__all__ = [
    'DMLResult',
    'manual_dml',
    'cate_by_subgroup',
    'heterogeneity_decomposition',
]
