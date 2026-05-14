"""Linear-probe utilities for the embedding probing experiments.

Functions return result dicts that get serialized straight to the
``outputs/probing/results/*.json`` files. Convention:

  - ``probe_classification`` reports both plain and balanced accuracy
    (relevant when targets are class-imbalanced — see
    `feedback_aux_drift_vs_basin.md`-style methodology hygiene). 5-fold CV.
  - ``probe_regression`` reports R². 5-fold CV.
  - ``majority_class_baseline`` is the "predict-the-mode" floor for any
    classification target. Compare every probe accuracy against this.
  - ``top_k_nn_accuracy`` is for the equivalence-class probe (Experiment 6)
    — replaces broken k-fold CV for high-cardinality targets.

All functions are CPU-only. The probes themselves are tiny linear models;
the only expensive step is for big embedding matrices.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate


def _isnan_safe(y: np.ndarray) -> np.ndarray:
    """np.isnan only works on float dtypes; integer arrays have no NaN."""
    if np.issubdtype(y.dtype, np.floating):
        return np.isnan(y)
    return np.zeros(len(y), dtype=bool)


def majority_class_baseline(y: np.ndarray) -> dict[str, float]:
    """Plain + balanced accuracy of always predicting the modal class."""
    valid = ~_isnan_safe(y)
    y_valid = y[valid] if not valid.all() else y
    values, counts = np.unique(y_valid, return_counts=True)
    mode = values[counts.argmax()]
    preds = np.full_like(y_valid, mode)
    return {
        "majority_class": int(mode) if np.issubdtype(y_valid.dtype, np.integer) else float(mode),
        "majority_acc": float(accuracy_score(y_valid, preds)),
        "majority_bal_acc": float(balanced_accuracy_score(y_valid, preds)),
    }


def probe_classification(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,
    max_iter: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """k-fold linear probe for classification. Returns mean ± std for both
    plain accuracy and balanced accuracy.

    Uses ``cross_validate`` with ``n_jobs=-1`` to parallelize across folds
    (~12× speedup over a sequential manual loop on a 4-core machine).

    Args:
        X: ``[N, D]`` embedding matrix.
        y: ``[N]`` integer labels.
        n_folds: stratified k-fold count.
        C: inverse regularization strength for LogisticRegression.
        max_iter: LogisticRegression iteration cap. 500 is enough for
            512-dim probes on 4694 samples to converge cleanly; 2000
            wastes time on already-converged folds.
        seed: KFold shuffle seed.

    Returns:
        Dict with ``acc_mean``, ``acc_std``, ``bal_acc_mean``, ``bal_acc_std``,
        ``n_samples``, ``n_classes``, ``n_folds``, ``elapsed_sec``, plus
        the majority-class baseline.
    """
    t0 = time.perf_counter()
    # Drop NaN samples (caller can pass arrays with NaN entries and we
    # silently restrict to the labeled subset).
    valid = ~_isnan_safe(y)
    if not valid.all():
        X = X[valid]
        y = y[valid]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    clf = LogisticRegression(C=C, max_iter=max_iter)
    # error_score=np.nan: a fold whose train split is single-class (rare
    # for tiny minority labels with stratified folds, but possible) gets
    # a NaN score instead of crashing the entire probe. We then nan-mean
    # the remaining folds.
    scores = cross_validate(
        clf, X, y, cv=skf,
        scoring=["accuracy", "balanced_accuracy"],
        n_jobs=-1, return_train_score=False,
        error_score=np.nan,
    )
    out = {
        "n_samples": int(len(y)),
        "n_classes": int(len(np.unique(y))),
        "n_folds": int(n_folds),
        "n_failed_folds": int(np.isnan(scores["test_accuracy"]).sum()),
        "acc_mean": float(np.nanmean(scores["test_accuracy"])),
        "acc_std": float(np.nanstd(scores["test_accuracy"])),
        "bal_acc_mean": float(np.nanmean(scores["test_balanced_accuracy"])),
        "bal_acc_std": float(np.nanstd(scores["test_balanced_accuracy"])),
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    out.update(majority_class_baseline(y))
    return out


def probe_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    alpha: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """k-fold linear probe for regression. Returns R² mean ± std.

    Args:
        X: ``[N, D]`` embedding matrix.
        y: ``[N]`` continuous targets.
        n_folds: k-fold count.
        alpha: Ridge regularization strength.
        seed: KFold shuffle seed.

    Returns:
        Dict with ``r2_mean``, ``r2_std``, ``y_mean``, ``y_std``,
        ``n_samples``, ``n_folds``, ``elapsed_sec``, ``baseline_r2`` (always
        0.0 — predicting the mean).
    """
    t0 = time.perf_counter()
    # Drop NaN samples (e.g., spine_depth is undefined for trivial /
    # family-C equations and comes through as NaN).
    valid = ~np.isnan(y)
    if not valid.all():
        X = X[valid]
        y = y[valid]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    reg = Ridge(alpha=alpha)
    scores = cross_validate(reg, X, y, cv=kf, scoring="r2", n_jobs=-1)
    return {
        "n_samples": int(len(y)),
        "n_folds": int(n_folds),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "r2_mean": float(np.mean(scores["test_score"])),
        "r2_std": float(np.std(scores["test_score"])),
        "baseline_r2": 0.0,  # predicting mean = R² of 0
        "elapsed_sec": float(time.perf_counter() - t0),
    }


def top_k_nn_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any]:
    """Nearest-neighbor membership accuracy.

    For each row i, find the k nearest rows in cosine-distance and
    check whether ANY of them shares the same label as row i. Report
    hit rate per k.

    Self is excluded from the neighbor set.

    Args:
        X: ``[N, D]`` embedding matrix.
        y: ``[N]`` integer labels.
        k_values: list of k values to report.

    Returns:
        Dict with ``top_k_acc[str(k)]`` for each k, ``n_samples``,
        ``n_classes``, ``elapsed_sec``, ``baseline_top_k`` (random
        baseline = 1 - (1 - p)^k for class proportions).
    """
    t0 = time.perf_counter()
    # Cosine-distance via L2-normalized X. (Many of our embeddings are
    # already L2-normalized; renormalize to be safe — idempotent.)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms
    # Pairwise cosine SIM (= dot product on unit sphere). N×N matrix.
    sim = X_norm @ X_norm.T
    np.fill_diagonal(sim, -np.inf)  # exclude self
    n = len(y)

    # Random baseline for top_k (class proportions).
    # With class probs p_c (size sum=1), random top-k accuracy = 1 - prod_c (1 - p_c)^(k * p_c)?
    # Simpler closed form: if neighbors were random, P(at least one of k neighbors shares
    # label c | own label is c) = 1 - (1 - p_c)^k. Expected accuracy = sum_c p_c * (1 - (1 - p_c)^k).
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    baseline = {}
    for k in k_values:
        baseline[str(k)] = float((probs * (1 - (1 - probs) ** k)).sum())

    out_top_k: dict[str, float] = {}
    max_k = max(k_values)
    # argpartition on negative sim → smallest first → largest sim first (i.e., nearest)
    # Get top max_k+1 indices per row (max_k after self exclusion).
    nn_idx = np.argpartition(-sim, max_k, axis=1)[:, :max_k]
    for k in k_values:
        hits = 0
        for i in range(n):
            kk = nn_idx[i, :k]
            if (y[kk] == y[i]).any():
                hits += 1
        out_top_k[str(k)] = hits / n

    return {
        "n_samples": int(n),
        "n_classes": int(len(np.unique(y))),
        "k_values": list(k_values),
        "top_k_acc": out_top_k,
        "baseline_top_k": baseline,
        "elapsed_sec": float(time.perf_counter() - t0),
    }


def probe_classification_with_mask(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Probe classification on the masked subset; return None if subset too small."""
    Xm = X[mask]
    ym = y[mask]
    if len(ym) < 50 or len(np.unique(ym)) < 2:
        return None
    return probe_classification(Xm, ym, **kwargs)


def probe_regression_with_mask(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Probe regression on the masked subset; return None if subset too small."""
    Xm = X[mask]
    ym = y[mask]
    if len(ym) < 50:
        return None
    if np.std(ym) < 1e-9:  # no variance → R² undefined
        return None
    return probe_regression(Xm, ym, **kwargs)
