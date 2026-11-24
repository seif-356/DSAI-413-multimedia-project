# Assignment — Projects at Lecture 6 (transcribed from the course PDF)

Page images: `page1.jpg`, `page2.jpg`, `page3.jpg`. Original PDF: each project 12 marks, due 25-11-2026. **Updated PDF (uploaded later): 6 marks each, 28-11-2026.**
**Moodle (binding): "Project (Two Tasks)" — 10 marks, due Wed 25 Nov 2026 12:00 AM, 1 attempt.** Confirm with the instructor which deadline applies; submit by the Moodle one.

## Project 1 — Multimodal Alzheimer's Disease Classification Using MRI and Clinical Text
Select a multimodal dataset; combine two modalities aligned from the **same patients**. Recommended: **ADNI** (brain MRI scans + clinical/cognitive assessment records).

1. Describe the two data modalities.
2. Build the proposed architecture:
   - MRI branch: CNN + ViT → image features `FI`
   - Text branch: TF-IDF / BERT → text features `FT`
   - Feature-level fusion: `F_fusion = [FI, FT]` (concatenation)
   - Deep learning classifier: fully connected layers + softmax → CN / MCI / AD
3. Apply suitable evaluation metrics.
4. Compare: CNN+ViT on images only; TF-IDF/BERT on text only; CNN+ViT+TF-IDF/BERT on image+text.
   **Note:** use exactly the same patient-level test set.
5. Does multimodal fusion improve performance?

## Project 2 — Human Activity Recognition from Videos
Inputs: video + audio + transcript.
- Video → CNN + optical flow
- Audio → MFCC + spectral features
- Text → TF-IDF / BERT
- Fusion → `[FV, FA, FT]`

Then select a deep learning model for classification and use suitable evaluation metrics.

## Submission (email to the instructor — address on page 3)
1. Python code files of both projects, containing the performance results.
2. The dataset uploaded to a Google Drive link.
3. A Word report with the performance of each model.
