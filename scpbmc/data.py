"""Data loading.

* PBMC 3k (10x Genomics, public): raw UMI counts are downloaded from 10x's CDN.
  Cell-type labels (data/pbmc3k_labels.csv) are the expert annotations of the
  standard Seurat / scanpy tutorial clustering, taken from
  ``scanpy.datasets.pbmc3k_processed()``.
* PBMC 68k reduced (Zheng et al., 2017): a 700-cell subset shipped inside scanpy.
  Its ``bulk_labels`` come from correlating each cell with FACS-sorted bulk
  profiles, i.e. an *independent* labelling method from a *different* donor.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PBMC3K_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/"
    "pbmc3k_filtered_gene_bc_matrices.tar.gz"
)
LABELS_CSV = Path(__file__).resolve().parent.parent / "data" / "pbmc3k_labels.csv"

# Fine PBMC3k labels -> shared coarse labels
COARSE_3K = {
    "CD4 T cells": "CD4 T",
    "CD8 T cells": "CD8 T",
    "B cells": "B",
    "NK cells": "NK",
    "CD14+ Monocytes": "Monocyte",
    "FCGR3A+ Monocytes": "Monocyte",
    "Dendritic cells": "Dendritic",
    "Megakaryocytes": "Megakaryocyte",
}
# PBMC68k FACS-based labels -> the same coarse labels (CD34+ has no 3k equivalent)
COARSE_68K = {
    "CD4+/CD25 T Reg": "CD4 T",
    "CD4+/CD45RO+ Memory": "CD4 T",
    "CD4+/CD45RA+/CD25- Naive T": "CD4 T",
    "CD8+ Cytotoxic T": "CD8 T",
    "CD8+/CD45RA+ Naive Cytotoxic": "CD8 T",
    "CD19+ B": "B",
    "CD56+ NK": "NK",
    "CD14+ Monocyte": "Monocyte",
    "Dendritic": "Dendritic",
}


def download_pbmc3k(cache_dir: str | Path = "data/raw") -> Path:
    """Download + extract the 10x PBMC3k filtered matrix (7.6 MB). Returns the mtx folder."""
    cache_dir = Path(cache_dir)
    mtx_dir = cache_dir / "filtered_gene_bc_matrices" / "hg19"
    if (mtx_dir / "matrix.mtx").exists():
        return mtx_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    tgz = cache_dir / "pbmc3k.tar.gz"
    with urllib.request.urlopen(PBMC3K_URL, timeout=120) as resp:
        tgz.write_bytes(resp.read())
    with tarfile.open(tgz) as tf:
        tf.extractall(cache_dir, filter="data")
    return mtx_dir


def load_pbmc3k(cache_dir: str | Path = "data/raw"):
    """Return (counts AnnData restricted to the 2,638 annotated cells, fine labels)."""
    import scanpy as sc

    adata = sc.read_10x_mtx(download_pbmc3k(cache_dir), var_names="gene_symbols")
    adata.var_names_make_unique()
    labels = pd.read_csv(LABELS_CSV, index_col="barcode")["cell_type"]
    adata = adata[labels.index].copy()
    return adata, labels.to_numpy()


def load_pbmc68k_reduced():
    """Return (log-normalised expression DataFrame cells x 765 genes, fine FACS labels)."""
    import scanpy as sc

    b = sc.datasets.pbmc68k_reduced()
    x = b.raw.X
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return pd.DataFrame(x, index=b.obs_names, columns=b.raw.var_names), b.obs[
        "bulk_labels"
    ].astype(str).to_numpy()


def to_coarse(labels, mapping: dict[str, str]) -> np.ndarray:
    """Map fine labels to coarse ones; unmapped labels become None."""
    return np.array([mapping.get(lab) for lab in labels], dtype=object)
