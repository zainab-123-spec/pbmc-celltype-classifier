import numpy as np
import scipy.sparse as sp

from scpbmc.preprocess import TopVarianceGenes, log_normalize, standardize_per_dataset


def test_log_normalize_rows_sum_to_target_before_log():
    counts = np.array([[1, 3, 0], [10, 0, 10]], dtype=float)
    x = log_normalize(counts)
    np.testing.assert_allclose(np.expm1(x).sum(axis=1), [1e4, 1e4])


def test_log_normalize_depth_invariant():
    counts = np.array([[1, 2, 3]], dtype=float)
    np.testing.assert_allclose(log_normalize(counts), log_normalize(counts * 7))


def test_log_normalize_accepts_sparse_and_zero_rows():
    counts = sp.csr_matrix(np.array([[0, 0, 0], [2, 2, 0]], dtype=float))
    x = log_normalize(counts)
    assert x.shape == (2, 3)
    assert np.all(x[0] == 0)
    assert np.all(np.isfinite(x))


def test_standardize_mean_zero_sd_one():
    rng = np.random.default_rng(0)
    x = standardize_per_dataset(rng.normal(5, 3, size=(200, 4)))
    np.testing.assert_allclose(x.mean(0), 0, atol=1e-9)
    np.testing.assert_allclose(x.std(0), 1, atol=1e-2)


def test_standardize_constant_gene_and_clip():
    x = np.zeros((50, 2))
    x[0, 1] = 1000
    z = standardize_per_dataset(x)
    assert np.all(z[:, 0] == 0)
    assert z.max() <= 10


def test_top_variance_genes_picks_most_variable():
    rng = np.random.default_rng(1)
    x = rng.normal(0, [0.1, 5, 0.2, 3], size=(100, 4))
    t = TopVarianceGenes(2).fit(x)
    assert list(t.idx_) == [1, 3]
    assert t.transform(x).shape == (100, 2)


def test_top_variance_genes_fitted_on_train_only():
    rng = np.random.default_rng(2)
    train = rng.normal(0, [5, 0.1], size=(100, 2))
    test = rng.normal(0, [0.1, 5], size=(100, 2))
    t = TopVarianceGenes(1).fit(train)
    assert list(t.idx_) == [0]
    assert t.transform(test).shape == (100, 1)


def test_top_variance_genes_k_larger_than_genes():
    x = np.random.default_rng(3).normal(size=(10, 3))
    assert TopVarianceGenes(100).fit_transform(x).shape == (10, 3)
