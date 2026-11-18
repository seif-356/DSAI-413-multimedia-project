# Project 2 — Multimodal Human Activity Recognition

`Project2_Multimodal_HAR.ipynb` — run top to bottom on **Google Colab (T4 GPU)**.
`build_notebook.py` regenerates the notebook: `python build_notebook.py Project2_Multimodal_HAR.ipynb`.

- **Dataset:** UCF101 (official split 1, group-disjoint), with classes chosen automatically by audio availability.
- **Video:** ResNet-50 features from RGB frames + Farneback optical-flow images → `FV` (4096)
- **Audio:** MFCC + Δ + ΔΔ + spectral features → `FA` (278)
- **Text:** Whisper transcripts → TF-IDF (SVD-256) / BERT mean-pooled (768) → `FT`
- **Fusion:** `[FV, FA, FT]` → MLP (FC + Softmax)
- **Metrics:** accuracy, macro P/R/F1, Cohen's κ, confusion matrices, mean ± std over 3 seeds, and McNemar's test for fusion vs the best single modality.

Outputs go to `MyDrive/MM_Project2/results/` (`results_table.csv`, `confusion_matrices.png`, `summary.json`, `transcripts.csv`). These feed the Word report.

Known weakness: UCF101 has no transcripts, and many clips contain no speech, so the text branch will be weak. The notebook reports the share of clips that have speech.

Expected runtime on a T4: about 1–2 h. Whisper is the slowest step, and features are cached on Drive.
