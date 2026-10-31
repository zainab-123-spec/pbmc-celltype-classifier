# How to put this on your Kaggle profile (10 minutes)

1. Go to https://www.kaggle.com/code and click **New Notebook**, then **File, Import Notebook**.
2. Upload `kaggle/pbmc-celltype-cross-donor-kaggle.ipynb`.
3. In the right panel, open **Settings** and turn **Internet** on. You may need to verify your
   phone once. CPU is enough. The notebook installs `scanpy` if it is missing and downloads about
   30 MB (PBMC 3k raw + processed from scanpy's mirrors).
4. Click **Run All**. It takes about 1-3 minutes. Check that the numbers match the README
   (random forest: about 0.902 within-CV macro F1, 0.807 on the new donor).
5. Click **Save Version**, choose **Save & Run All (Commit)**, then set the notebook to **Public**.
6. Title: `Single-cell PBMC Cell Types: Does It Work on a New Donor?`
   Tags: `biology`, `healthcare`, `classification`, `genetics`.
7. Optional: Kaggle has public PBMC 3k datasets. You can also click **Add Data** and link one, but
   the notebook does not need it.
