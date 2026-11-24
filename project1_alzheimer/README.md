# Project 1 — Multimodal Alzheimer's Classification (MRI + clinical text)

`Project1_Multimodal_Alzheimer.ipynb` — Colab, T4 GPU, Run all (~20–30 min).

- **Dataset:** OASIS-1 cross-sectional via public Kaggle mirrors (`ninadaithal/imagesoasis` MRI slices + `jboysen/mri-and-alzheimers` clinical CSV), joined on subject ID. ADNI not used: access approval exceeds the deadline.
- **Labels:** CDR 0 → CN, 0.5 → MCI, ≥1 → AD. CDR never appears in the text.
- **MRI:** ResNet-50 + ViT-B/16 (frozen) on 8 axial slices, averaged → `FI`.
- **Text:** clinical note → TF-IDF / BERT → `FT`. Fusion `[FI, FT]` → FC + Softmax.
- **Protocol:** patient-level stratified 5-fold CV × 3 seeds, identical folds for every model; McNemar test fusion vs best single.
- **Ablation:** MMSE removed from text — MMSE nearly defines the diagnosis, so the main text result is optimistic.

Untested parts: the Kaggle file-name pattern (the notebook prints examples and stops with a clear message if it differs) and the GPU feature extraction. Everything else was tested on synthetic data.
