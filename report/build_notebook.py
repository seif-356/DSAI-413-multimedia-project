import nbformat as nbf
import sys

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# DSAI 413 — Build the Word report from both projects' results
Run **after** Project 1 and Project 2 notebooks have finished. Reads `MyDrive/MM_Project1/results` and `MyDrive/MM_Project2/results`,
writes `MyDrive/DSAI413_Report.docx`. Edit the names and links in the first cell. Read the report before submitting.
""")

code(r"""
TEAM        = ['Student 1 name — ID', 'Student 2 name — ID']        # <-- EDIT
DRIVE_LINK1 = 'https://drive.google.com/...  (MM_Project1/dataset)'  # <-- EDIT: share link of the dataset folder
DRIVE_LINK2 = 'https://drive.google.com/...  (MM_Project2/data)'     # <-- EDIT

!pip -q install python-docx
import os, json, pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from google.colab import drive
drive.mount('/content/drive')
R1, R2 = '/content/drive/MyDrive/MM_Project1/results', '/content/drive/MyDrive/MM_Project2/results'
OUT = '/content/drive/MyDrive/DSAI413_Report.docx'
""")

code(r"""
def load(d):
    s = json.load(open(f'{d}/summary.json')) if os.path.exists(f'{d}/summary.json') else None
    t = pd.read_csv(f'{d}/results_table.csv') if os.path.exists(f'{d}/results_table.csv') else None
    return s, t
s1, t1 = load(R1); s2, t2 = load(R2)
print('Project 1 results found:', s1 is not None, '| Project 2 results found:', s2 is not None)

doc = Document()
doc.styles['Normal'].font.name = 'Calibri'; doc.styles['Normal'].font.size = Pt(11)

def para(text, bold=False):
    p = doc.add_paragraph(); r = p.add_run(text); r.bold = bold; return p

def table(df):
    tb = doc.add_table(rows=1, cols=len(df.columns)); tb.style = 'Light Grid Accent 1'
    for i, c in enumerate(df.columns): tb.rows[0].cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = tb.add_row().cells
        for i, v in enumerate(row): cells[i].text = str(v)
    for row in tb.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs: r.font.size = Pt(8)

def image(path, w=6.5):
    if os.path.exists(path): doc.add_picture(path, width=Inches(w))

def pct(x): return f'{100*x:.1f}%'

doc.add_heading('DSAI 413 — Multimedia Project (Two Tasks)', 0)
para('Team: ' + '; '.join(TEAM))
para('Code: Project1_Multimodal_Alzheimer.ipynb, Project2_Multimodal_HAR.ipynb (outputs included in the notebooks).')
para(f'Datasets (Google Drive): Project 1 — {DRIVE_LINK1}; Project 2 — {DRIVE_LINK2}')

# ---------------- Project 1 ----------------
doc.add_heading('Project 1 — Multimodal Alzheimer\'s Disease Classification (MRI + Clinical Text)', 1)
doc.add_heading('1. Dataset and the two data modalities', 2)
para('ADNI requires an access application whose approval exceeds the project deadline, so the open OASIS-1 cross-sectional '
     'dataset was used. Both modalities come from the same patients and are joined on the subject ID.')
if s1:
    cc = s1['class_counts']
    para(f"Cohort: {s1['n_patients']} patients with both an MRI and a clinical rating — CN (CDR 0): {cc['CN']}, "
         f"MCI (CDR 0.5): {cc['MCI']}, AD (CDR ≥ 1): {cc['AD']}.")
para('Modality 1 — MRI: T1-weighted structural brain MRI (1.5 T, atlas-registered). Each patient is represented by '
     '8 evenly spaced axial slices of the first acquisition.')
para('Modality 2 — Clinical text: each patient\'s clinical and cognitive-assessment record (sex, handedness, age, education, '
     'socioeconomic status, MMSE score, intracranial volume eTIV, normalised whole-brain volume nWBV) written as a short clinical note. '
     'The CDR rating is excluded from the text because it defines the label.')
if os.path.exists(f'{R1}/modality_summary.csv'): table(pd.read_csv(f'{R1}/modality_summary.csv'))
image(f'{R1}/mri_examples.png', 5.5)

doc.add_heading('2. Architecture', 2)
para('MRI branch: ResNet-50 (CNN, 2048-d) and ViT-B/16 (768-d), ImageNet-pretrained and frozen; slice features are averaged → FI. '
     'Text branch: TF-IDF (uni+bi-grams, fitted on training patients only, SVD) or BERT (bert-base-uncased, mean-pooled 768-d) → FT. '
     'Each block is standardised and reduced to 64 dimensions with PCA fitted on the training fold, so no modality dominates by size. '
     'Feature-level fusion: F_fusion = [FI, FT] (concatenation). Classifier: fully connected network '
     '(128 → 64 → 3, ReLU, dropout 0.5) with softmax output, class-weighted cross-entropy, AdamW, early stopping on an inner validation split.')

doc.add_heading('3. Evaluation protocol and metrics', 2)
if s1:
    para(f"Patient-level stratified {s1['n_folds']}-fold cross-validation (one scan per patient, so no patient is in both train and test), "
         f"repeated with {len(s1['seeds'])} seeds. All models use exactly the same patient-level test folds. "
         'Metrics: accuracy, balanced accuracy, macro precision/recall/F1, Cohen\'s κ and macro one-vs-rest ROC-AUC (mean ± std over fold-seed runs), '
         'confusion matrices, and an exact McNemar test between the best fusion model and the best single-modality model.')

doc.add_heading('4. Results — image only vs text only vs fusion', 2)
if t1 is not None: table(t1)
image(f'{R1}/confusion_matrices.png')

doc.add_heading('5. Does multimodal fusion improve performance?', 2)
if s1:
    for key, title in [('main', 'Main setting (text includes MMSE)'), ('ablation_no_mmse', 'Ablation (MMSE removed from text)')]:
        m = s1[key]
        para(f"{title}: best single = {m['best_single']}; best fusion = {m['fusion']}; macro-F1 gain = {m['gain_f1']:+.4f}; "
             f"fusion corrects {m['n_fixed']} and breaks {m['n_broken']} patients; McNemar p = {m['mcnemar_p']:.3g}. "
             f"Conclusion: {m['verdict']}")
para('Interpretation: MMSE is a cognitive test that is used, together with CDR, to diagnose dementia, so the text modality is '
     'very strong when MMSE is included. The ablation without MMSE shows how much the MRI adds when the text carries no direct '
     'cognitive score.', )
doc.add_heading('Limitations', 3)
para('Small cohort (especially the AD class), frozen ImageNet feature extractors on 2-D slices rather than a 3-D model trained on MRI, '
     'and CDR 0.5 used as a proxy for MCI. Results are cross-validated estimates, not a held-out external test.')

# ---------------- Project 2 ----------------
doc.add_heading('Project 2 — Human Activity Recognition from Video + Audio + Transcript', 1)
doc.add_heading('1. Dataset', 2)
para('UCF101 (YouTube action clips). Only classes whose clips carry an audio track were kept; the official split 1 is used, '
     'which keeps the same actor groups out of both train and test. UCF101 has no transcripts, so they were generated with Whisper.')
if s2:
    para(f"Classes ({len(s2['classes'])}): {', '.join(s2['classes'])}. Clips — train: {s2.get('n_train','?')}, "
         f"validation: {s2.get('n_val','?')}, test: {s2['n_test']}.")
    if 'speech_fraction' in s2:
        para(f"Clips with detectable speech: {pct(s2['speech_fraction'])}. The remaining clips have no transcript, "
             'which limits the text modality.')
doc.add_heading('2. Features and model', 2)
para('Video (FV): ResNet-50 features of 8 uniformly sampled RGB frames + ResNet-50 features of Farneback optical-flow images '
     '(HSV-encoded), averaged over time (4096-d). Audio (FA): 40 MFCC + Δ + ΔΔ and spectral centroid, bandwidth, roll-off, flatness, '
     'zero-crossing rate, RMS, spectral contrast and chroma, mean/std over time (278-d). Text (FT): Whisper transcript → TF-IDF '
     '(SVD-256) or BERT mean-pooled (768-d). Fusion: [FV, FA, FT] concatenation of standardised blocks → fully connected network '
     '(512 → 256, BatchNorm, ReLU, dropout) with softmax, early stopping on the validation groups.')
doc.add_heading('3. Results', 2)
if t2 is not None: table(t2)
image(f'{R2}/confusion_matrices.png')
doc.add_heading('4. Does fusion improve performance?', 2)
if s2:
    p = s2['mcnemar_p']; g = s2['gain_f1']
    verdict = ('Yes — the improvement is statistically significant (p < 0.05).' if g > 0 and p < 0.05 else
               'Fusion is better on average but the difference is not statistically significant.' if g > 0 else
               'No — fusion did not beat the best single modality.')
    para(f"Best single modality: {s2['best_single']}; best fusion: {s2['fusion']}; macro-F1 gain {g:+.4f}; "
         f"McNemar p = {p:.3g} on {s2['n_test']} test clips. {verdict}")
    if 'ablation_f1' in s2:
        para('Drop-one-modality ablation (macro-F1): ' + ', '.join(f'{k}: {v:.4f}' for k, v in s2['ablation_f1'].items()))
doc.add_heading('Limitations', 3)
para('Transcripts are machine-generated and most activity clips contain little speech; frozen ImageNet CNN features; '
     'a subset of UCF101 classes with audio.')

doc.save(OUT); print('Report written to', OUT)
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "colab": {"provenance": []}}
nbf.write(nb, sys.argv[1])
print('written', len(cells), 'cells')
