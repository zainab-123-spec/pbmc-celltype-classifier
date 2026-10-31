# Learning guide (read this before an interview)

## The project in one breath
"I trained cell-type classifiers on a public single-cell dataset and tested them on a different
donor. Inside one dataset they scored about 0.90 macro F1, but on the new donor the best one scored
0.81. I also found that a common normalisation trick made it worse, and I explained why."

## Key ideas, very simply

**Single-cell RNA-seq.** For every cell, we count how many RNA molecules (UMIs) of each gene we
saw. The result is a table: cells × genes. It is mostly zeros (about 97%).

**PBMC.** Blood immune cells: T cells (CD4 helpers, CD8 killers), B cells (make antibodies), NK
cells, monocytes, dendritic cells and megakaryocytes (platelet makers).

**Normalisation.** Some cells get sequenced more deeply than others. We divide each cell by its
total count and multiply by 10,000, so every cell has the same "budget". Then we take `log(1+x)`
so a few huge genes don't dominate.

**Highly variable genes.** Most genes look the same in every cell. Genes that vary a lot carry the
cell-type signal. We keep the top 2,000, choosing them **only on training cells**.

**Why choose genes only on training cells?** If you look at test cells to pick genes, test
information leaks into the model and the score is too high. The `Pipeline` re-does the selection
in every fold.

**Macro F1.** The F1 score for each class, averaged with equal weight. Accuracy can look great
just by getting the big CD4 T class right. Macro F1 punishes a model that ignores the 15
megakaryocytes.

**Marker-gene baseline.** Biologists know *MS4A1* means B cell and *LYZ* means monocyte. The
baseline picks the type whose markers are highest. It needs no training. If ML can't beat it,
ML is not worth it.

**Cross-dataset test.** Train on donor A (PBMC 3k), test on donor B (PBMC 68k). This is like
testing a model on a new hospital's data. It is the real test.

**Batch effect.** Differences between datasets caused by the lab, machine or day, not by
biology.

## The surprising result
Z-scoring each dataset on its own sounds like a good batch fix. But it subtracts each dataset's
**average cell**. The 68k subset is 35% dendritic cells and PBMC 3k is 1.4%, so their averages are
very different for biological reasons. Subtracting them distorts every cell, and macro F1 fell
from 0.79 to 0.55 for logistic regression. The lesson: a normalisation step has assumptions, and
here the assumption was "same cell mix".

## Interview Q&A

**Q: Why did you test on another dataset?**
A: PBMC 3k labels were made by clustering the same data, so within-dataset CV partly asks the
model to re-find the clusters. An independent donor with labels from a different method (FACS
correlation) is a more honest test. The score dropped from 0.918 to 0.786 for logistic
regression on the same genes.

**Q: How did you avoid data leakage?**
A: Gene filtering by "detected in at least 3 cells" uses no labels. Variable-gene selection,
scaling and PCA are all inside an sklearn Pipeline, so they are fitted on the training fold only.
The external test set is never used for any choice.

**Q: Why macro F1 instead of accuracy?**
A: Classes go from 1,144 cells down to 15. A model that never predicts megakaryocytes still gets
high accuracy. Macro F1 weights every class equally.

**Q: What were the main errors?**
A: CD8 T cells called CD4 T (37 of 97), and FACS-labelled dendritic cells called monocytes (53 of
240). Both are closely related lineages, and the 68k labels are noisy for these.

**Q: Random forest vs logistic regression?**
A: Random forest was slightly better both within (0.902 vs 0.888) and across datasets (0.807 vs
0.786). The within-dataset fold sd is about 0.03-0.05, so the within-dataset difference is not
clearly significant.

**Q: What would you do next?**
A: Use a proper integration method (e.g. scVI, Harmony or Seurat anchors) and train on several
donors. I would also test on a full-gene dataset (not the 764-gene reduced one), and compare
with a reference-mapping tool such as CellTypist.

**Q: What's the limitation you'd mention first?**
A: Neither label set is true ground truth. The 68k test is small (687 cells, 31 NK) and has no
megakaryocytes.

## Words to know
UMI, sparse matrix, log1p, highly variable genes, PCA, UMAP, kNN, stratified K-fold, macro F1,
batch effect, data leakage, FACS.
