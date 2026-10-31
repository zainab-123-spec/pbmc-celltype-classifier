import numpy as np
import pytest

from scpbmc.models import MARKERS, MODEL_NAMES, MarkerScoreClassifier, make_model


def synthetic_cells(n_per=40, seed=0):
    """Fake cells: every type over-expresses its own block of 5 'marker' genes."""
    rng = np.random.default_rng(seed)
    types = ["A", "B", "C"]
    x, y = [], []
    for k, t in enumerate(types):
        block = rng.normal(0, 1, size=(n_per, 30))
        block[:, k * 5:(k + 1) * 5] += 4
        x.append(block)
        y += [t] * n_per
    return np.vstack(x), np.array(y)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_learn_separable_types(name):
    x, y = synthetic_cells()
    model = make_model(name, n_genes=20, n_pcs=5).fit(x[::2], y[::2])
    assert np.mean(model.predict(x[1::2]) == y[1::2]) > 0.9


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        make_model("transformer")


def test_marker_classifier_needs_no_labels():
    x, y = synthetic_cells()
    genes = [f"g{i}" for i in range(30)]
    markers = {"A": ["g0", "g1"], "B": ["g5", "g6"], "C": ["g10", "g11"]}
    clf = MarkerScoreClassifier(genes, markers).fit()
    assert np.mean(clf.predict(x) == y) > 0.95


def test_marker_classifier_skips_missing_types():
    genes = ["MS4A1", "CD79A", "LYZ"]
    clf = MarkerScoreClassifier(genes).fit()
    assert set(clf.classes_) == {"B", "Monocyte"}
    assert clf.decision_function(np.random.default_rng(0).random((5, 3))).shape == (5, 2)


def test_marker_list_covers_all_coarse_types():
    assert set(MARKERS) == {"CD4 T", "CD8 T", "B", "NK", "Monocyte", "Dendritic", "Megakaryocyte"}
