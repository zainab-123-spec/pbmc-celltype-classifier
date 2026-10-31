"""Classifiers: a no-training marker-gene baseline and three learned models."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .preprocess import TopVarianceGenes

# Textbook PBMC marker genes (Seurat / scanpy PBMC3k tutorial)
MARKERS = {
    "CD4 T": ["IL7R", "CD3E", "LDHB"],
    "CD8 T": ["CD8A", "CD8B", "CD3E"],
    "B": ["MS4A1", "CD79A", "CD79B"],
    "NK": ["GNLY", "NKG7", "GZMB"],
    "Monocyte": ["LYZ", "CD14", "FCGR3A", "S100A8"],
    "Dendritic": ["FCER1A", "CST3", "CLEC10A"],
    "Megakaryocyte": ["PPBP", "PF4"],
}


class MarkerScoreClassifier(BaseEstimator, ClassifierMixin):
    """Predict the cell type whose marker genes have the highest mean z-score.

    Needs only gene names, no training labels: the "biologist with a
    marker list" baseline every ML model has to beat.
    """

    def __init__(self, gene_names, markers=None):
        self.gene_names = gene_names
        self.markers = markers

    def fit(self, x=None, y=None):
        markers = self.markers or MARKERS
        pos = {g: i for i, g in enumerate(self.gene_names)}
        self.classes_ = np.array([c for c in markers if any(g in pos for g in markers[c])])
        self.idx_ = [[pos[g] for g in markers[c] if g in pos] for c in self.classes_]
        return self

    def decision_function(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - x.mean(0)) / np.where(x.std(0) == 0, 1, x.std(0))
        return np.column_stack([z[:, idx].mean(1) for idx in self.idx_])

    def predict(self, x):
        return self.classes_[np.argmax(self.decision_function(x), axis=1)]


def make_model(name: str, n_genes: int = 2000, n_pcs: int = 50, seed: int = 0):
    """sklearn pipelines; gene selection, scaling and PCA are fitted inside each fold."""
    head = [TopVarianceGenes(n_genes), StandardScaler()]
    if name == "logreg":
        return make_pipeline(*head, LogisticRegression(C=0.1, max_iter=3000))
    if name == "knn_pca":
        return make_pipeline(*head, PCA(n_pcs, random_state=seed), KNeighborsClassifier(15))
    if name == "random_forest":
        return make_pipeline(
            TopVarianceGenes(n_genes),
            RandomForestClassifier(300, random_state=seed, n_jobs=-1, class_weight="balanced"),
        )
    raise ValueError(f"unknown model {name}")


MODEL_NAMES = ["logreg", "knn_pca", "random_forest"]
