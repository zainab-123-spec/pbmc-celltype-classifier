import numpy as np
import pandas as pd

from scpbmc.data import COARSE_3K, COARSE_68K, LABELS_CSV, to_coarse
from scpbmc.evaluate import cross_validate, scores
from tests.test_models import synthetic_cells


def test_coarse_maps_share_label_space():
    assert set(COARSE_68K.values()) <= set(COARSE_3K.values())


def test_to_coarse_unmapped_is_none():
    out = to_coarse(["B cells", "CD34+"], COARSE_3K)
    assert out[0] == "B" and out[1] is None


def test_labels_file_matches_tutorial():
    labels = pd.read_csv(LABELS_CSV)
    assert len(labels) == 2638
    assert labels["barcode"].is_unique
    assert set(labels["cell_type"]) == set(COARSE_3K)


def test_scores_perfect_and_imbalanced():
    s = scores(["a", "b"], ["a", "b"])
    assert s == {"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0}
    y = np.array(["a"] * 9 + ["b"])
    s = scores(y, np.array(["a"] * 10))
    assert s["accuracy"] == 0.9 and s["balanced_accuracy"] == 0.5


def test_cross_validate_reports_mean_and_std():
    x, y = synthetic_cells(n_per=20)
    # rename the fake types / marker blocks to real PBMC markers
    y = np.array([{"A": "B", "B": "NK", "C": "Monocyte"}[t] for t in y])
    genes = [f"g{i}" for i in range(30)]
    genes[0:3] = ["MS4A1", "CD79A", "CD79B"]
    genes[5:8] = ["GNLY", "NKG7", "GZMB"]
    genes[10:13] = ["LYZ", "CD14", "S100A8"]
    out = cross_validate(x, y, model_names=["logreg"], gene_names=genes, n_splits=3)
    assert set(out) == {"logreg", "marker_baseline"}
    assert 0.9 < out["logreg"]["macro_f1"]["mean"] <= 1.0
    assert out["logreg"]["accuracy"]["std"] >= 0
    assert out["marker_baseline"]["accuracy"]["mean"] > 0.9
