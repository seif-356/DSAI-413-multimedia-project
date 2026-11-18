# Project 1 — Multimodal Alzheimer's Classification (MRI + clinical text) — PLAN

Open risks to resolve before coding:
1. **ADNI access** requires an application approved by the data committee. Approval can take longer than the deadline. Fallback: **OASIS** (MRI + clinical CSV: age, education, SES, MMSE, CDR).
2. **Label leakage:** the CN/MCI/AD diagnosis is largely *defined* by MMSE/CDR. If those fields are in the text, text-only accuracy will be near 100% and the fusion comparison becomes meaningless. Either exclude CDR from the text or report both versions explicitly.
3. **Patient-level split:** multiple scans per patient must stay on one side of the split.
