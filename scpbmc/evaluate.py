"""Experiments: (1) within-dataset 5-fold CV, (2) train on PBMC3k -> test on PBMC68k."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from .data import COARSE_3K, COARSE_68K, load_pbmc3k, load_pbmc68k_reduced, to_coarse
from .models import MODEL_NAMES, MarkerScoreClassifier, make_model
from .preprocess import log_normalize, standardize_per_dataset


def scores(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def cross_validate(x, y, model_names=MODEL_NAMES, gene_names=None, n_splits=5, seed=0):
    """Stratified K-fold CV. Returns mean and std of each score, per model."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_model: dict[str, list[dict]] = {m: [] for m in model_names}
    if gene_names is not None:
        per_model["marker_baseline"] = []
    for tr, te in skf.split(x, y):
        for m in model_names:
            pred = make_model(m, seed=seed).fit(x[tr], y[tr]).predict(x[te])
            per_model[m].append(scores(y[te], pred))
        if gene_names is not None:
            pred = MarkerScoreClassifier(gene_names).fit().predict(x[te])
            per_model["marker_baseline"].append(scores(y[te], pred))
    out = {}
    for m, folds in per_model.items():
        out[m] = {
            k: {"mean": float(np.mean([f[k] for f in folds])),
                "std": float(np.std([f[k] for f in folds]))}
            for k in folds[0]
        }
    return out


def run(cache_dir="data/raw", out_dir="results", seed=0) -> dict:
    adata, fine = load_pbmc3k(cache_dir)
    # normalise on all genes, then drop genes seen in < 3 cells (label-free filter)
    x3k = log_normalize(adata.X).astype(np.float32)
    detected = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    keep_genes = detected >= 3
    x3k = x3k[:, keep_genes]
    genes = np.array(adata.var_names)[keep_genes]
    y3k = to_coarse(fine, COARSE_3K)
    results: dict = {"n_cells_3k": int(len(y3k)), "n_genes_3k": int(len(genes))}

    # (1) within-dataset CV, fine and coarse labels
    results["cv_fine"] = cross_validate(x3k, np.asarray(fine), seed=seed)
    results["cv_coarse"] = cross_validate(x3k, y3k, gene_names=genes, seed=seed)

    # (2) cross-dataset: different donor, different labelling method
    x68, fine68 = load_pbmc68k_reduced()
    y68 = to_coarse(fine68, COARSE_68K)
    keep = np.array([v is not None for v in y68])
    shared = [g for g in x68.columns if g in set(genes)]
    gidx = [int(np.where(genes == g)[0][0]) for g in shared]
    a_tr, a_te = x3k[:, gidx], x68[shared].to_numpy()[keep]
    y_te = y68[keep].astype(str)
    results["cross"] = {"n_cells_68k": int(keep.sum()), "n_shared_genes": len(shared),
                        "dropped_cd34": int((~keep).sum())}
    preds = {}
    for scaling in ("none", "per_dataset_zscore"):
        if scaling == "none":
            tr_x, te_x = a_tr, a_te
        else:
            tr_x, te_x = standardize_per_dataset(a_tr), standardize_per_dataset(a_te)
        for m in MODEL_NAMES:
            model = make_model(m, n_genes=len(shared), seed=seed).fit(tr_x, y3k.astype(str))
            p = model.predict(te_x)
            preds[f"{m}|{scaling}"] = p
            results["cross"][f"{m}|{scaling}"] = scores(y_te, p)
    p = MarkerScoreClassifier(shared).fit().predict(a_te)
    preds["marker_baseline"] = p
    results["cross"]["marker_baseline"] = scores(y_te, p)
    # Within-3k CV restricted to the same shared genes (fair comparison)
    results["cv_coarse_shared_genes"] = cross_validate(
        a_tr, y3k.astype(str), model_names=["logreg"], seed=seed
    )

    best = max((k for k in preds if "|" in k), key=lambda k: results["cross"][k]["macro_f1"])
    labels = ["CD4 T", "CD8 T", "B", "NK", "Monocyte", "Dendritic", "Megakaryocyte"]
    results["cross"]["best"] = best
    results["cross"]["best_confusion"] = {
        "labels": labels,
        "matrix": confusion_matrix(y_te, preds[best], labels=labels).tolist(),
    }
    results["cross"]["best_per_class_f1"] = dict(zip(
        labels[:6], f1_score(y_te, preds[best], labels=labels[:6], average=None).tolist()
    ))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    res = run()
    for block in ("cv_fine", "cv_coarse"):
        for m, s in res[block].items():
            f = s["macro_f1"]
            print(f"{block:10s} {m:16s} macroF1={f['mean']:.3f}±{f['std']:.3f}")
    for k, v in res["cross"].items():
        if isinstance(v, dict) and "macro_f1" in v:
            print(f"cross      {k:32s} macroF1={v['macro_f1']:.3f} acc={v['accuracy']:.3f}")


if __name__ == "__main__":
    main()
