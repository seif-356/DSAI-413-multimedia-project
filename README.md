# DSAI 413 — Multimedia Projects (Two Tasks)

Assignment spec: `assignment/ASSIGNMENT.md`. History: `HISTORY.md`.
**Moodle due: Wed 25 Nov 2026 12:00 AM, 1 attempt** (the updated PDF says 28-11-2026 — Moodle is what locks).

## Run order (Google Colab, Runtime → T4 GPU → Run all)
1. `project1_alzheimer/Project1_Multimodal_Alzheimer.ipynb` — ~20–30 min. Outputs → `MyDrive/MM_Project1/`
2. `project2_activity_recognition/Project2_Multimodal_HAR.ipynb` — ~1–1.5 h (6.5 GB download + Whisper). Outputs → `MyDrive/MM_Project2/`
3. `report/Make_Report.ipynb` — edit team names + Drive links in cell 1 → writes `MyDrive/DSAI413_Report.docx` (no GPU needed)

Run 1 and 2 in parallel in two Colab tabs if your account allows two GPU sessions.

## Run on a Mac (Apple silicon) instead
Outputs go to `~/DSAI413/` instead of Drive; upload the dataset folders to Drive afterwards.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone -b claude/festive-hypatia-wn53pf https://github.com/seif-356/DSAI-413-multimedia-project.git ~/DSAI413-code
cd ~/DSAI413-code && uv venv --seed --python 3.12 && source .venv/bin/activate
uv pip install jupyterlab torch torchvision transformers openai-whisper librosa opencv-python-headless scikit-learn scipy pandas seaborn tqdm kagglehub python-docx imageio-ffmpeg
jupyter lab
```

## Submit
1. Both notebooks **with outputs** (File → Download .ipynb after running).
2. Google Drive share links: `MyDrive/MM_Project1/dataset` and `MyDrive/MM_Project2/data`.
3. `DSAI413_Report.docx` — read it and fix anything wrong before submitting.

`build_notebook.py` in each folder regenerates its notebook.
