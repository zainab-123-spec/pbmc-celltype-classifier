"""Pre-processing written out by hand (so it is easy to explain in an interview)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin


def log_normalize(counts, target_sum: float = 1e4) -> np.ndarray:
    """Library-size normalisation then log1p: log(1 + counts / total * 10,000).

    Cells are sequenced to different depths; dividing by each cell's total
    UMI count removes that technical difference. log1p tames huge values.
    """
    x = counts.toarray() if sp.issparse(counts) else np.asarray(counts, dtype=float)
    totals = x.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    return np.log1p(x / totals * target_sum)


def standardize_per_dataset(x: np.ndarray) -> np.ndarray:
    """Z-score every gene *within one dataset* (mean 0, sd 1), clipped to [-10, 10].

    A simple batch-effect correction: each dataset is put on its own scale
    before the classifier sees it. Uses no labels, so it is leakage-free.
    """
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1
    return np.clip((x - mu) / sd, -10, 10)


class TopVarianceGenes(BaseEstimator, TransformerMixin):
    """Keep the ``n_genes`` most variable genes, chosen on the *training* fold only."""

    def __init__(self, n_genes: int = 2000):
        self.n_genes = n_genes

    def fit(self, x, y=None):
        var = np.asarray(x).var(axis=0)
        k = min(self.n_genes, var.size)
        self.idx_ = np.sort(np.argsort(var)[::-1][:k])
        return self

    def transform(self, x):
        return np.asarray(x)[:, self.idx_]
