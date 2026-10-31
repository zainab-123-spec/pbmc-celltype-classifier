"""Figures for the README (all generated from results/metrics.json + the data)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_umap(out="results/figures/umap_pbmc3k.png"):
    """UMAP of PBMC3k coloured by cell type (coordinates from scanpy's processed copy)."""
    import scanpy as sc

    ad = sc.datasets.pbmc3k_processed()
    xy = ad.obsm["X_umap"]
    labels = ad.obs["louvain"].astype(str).to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for lab in sorted(set(labels)):
        m = labels == lab
        ax.scatter(xy[m, 0], xy[m, 1], s=4, label=f"{lab} ({m.sum()})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("PBMC 3k: 2,638 cells, 8 annotated types")
    ax.legend(markerscale=3, fontsize=7, loc="best")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_within_vs_cross(metrics: dict, out="results/figures/within_vs_cross.png"):
    models = ["marker_baseline", "logreg", "knn_pca", "random_forest"]
    within = [metrics["cv_coarse"][m]["macro_f1"]["mean"] for m in models]
    within_sd = [metrics["cv_coarse"][m]["macro_f1"]["std"] for m in models]
    cross = [
        metrics["cross"]["marker_baseline" if m == "marker_baseline" else f"{m}|none"]["macro_f1"]
        for m in models
    ]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, within, 0.4, yerr=within_sd, capsize=3, label="PBMC3k 5-fold CV")
    ax.bar(x + 0.2, cross, 0.4, label="Train 3k -> test 68k (new donor)")
    for i in range(len(models)):
        top = within[i] + within_sd[i] + 0.01
        ax.text(x[i] - 0.2, top, f"{within[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + 0.2, cross[i] + 0.01, f"{cross[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["marker genes\n(no training)", "logistic reg.", "PCA + kNN", "random forest"])
    ax.set_ylabel("macro F1 (coarse types)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Within-dataset scores overstate real-world accuracy")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cross_confusion(metrics: dict, out="results/figures/cross_confusion.png"):
    conf = metrics["cross"]["best_confusion"]
    labels = conf["labels"]
    m = np.array(conf["matrix"])
    rows = m.sum(1) > 0
    m, ylabels = m[rows], [lab for lab, r in zip(labels, rows) if r]
    frac = m / m.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, m[i, j], ha="center", va="center",
                    color="white" if frac[i, j] > 0.6 else "black", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_xlabel("predicted (model trained on PBMC3k)")
    ax.set_ylabel("true (PBMC68k FACS label)")
    ax.set_title(f"Cross-dataset confusion: {metrics['cross']['best']}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    metrics = json.loads(Path("results/metrics.json").read_text())
    plot_umap()
    plot_within_vs_cross(metrics)
    plot_cross_confusion(metrics)


if __name__ == "__main__":
    main()
