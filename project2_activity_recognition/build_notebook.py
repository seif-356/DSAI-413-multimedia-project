import nbformat as nbf
import sys

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Project 2 — Multimodal Human Activity Recognition (Video + Audio + Transcript)

**Pipeline**
| Modality | Feature extractor | Vector |
|---|---|---|
| Video | CNN (ResNet-50, ImageNet) on RGB frames **+** CNN on Farneback optical-flow images | `FV` (4096-d) |
| Audio | MFCC (+Δ, ΔΔ) + spectral features (centroid, bandwidth, roll-off, flatness, contrast, ZCR, RMS, chroma) | `FA` (278-d) |
| Text  | Whisper ASR transcript → TF-IDF **and** BERT (`bert-base-uncased`, mean-pooled) | `FT` |
| Fusion | Feature-level concatenation `[FV, FA, FT]` → MLP (FC + Softmax) | — |

**Dataset:** UCF101 (YouTube clips with audio tracks). UCF101 has **no transcripts**, so they are generated with Whisper.
We keep only classes whose clips actually contain an audio stream (checked with `ffprobe`, not assumed).

**Evaluation protocol:** the official UCF101 **split 1**. Its train/test split is by actor group, so clips of the same person/scene never appear on both sides.
Validation is carved from the training groups. **Every model is tested on the same test clips.**
Metrics: Accuracy, macro Precision / Recall / F1, Cohen's κ, confusion matrices. Mean ± std over 3 seeds, plus McNemar's test for *fusion vs best single modality*.

> Run on **Google Colab with a GPU** (Runtime → Change runtime type → T4 GPU). Features are cached to Google Drive, so a disconnect does not lose work.
""")

code(r"""
# 1) Setup
!pip -q install openai-whisper
!apt-get -qq install -y unrar ffmpeg > /dev/null

import os, re, json, glob, random, subprocess, warnings
import numpy as np, pandas as pd
import cv2, librosa, torch, torch.nn as nn
import matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE)

USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = '/content/drive/MyDrive/MM_Project2'
else:
    ROOT = '/content/MM_Project2'
DATA, FEAT, RES = f'{ROOT}/data', f'{ROOT}/features', f'{ROOT}/results'
for d in (DATA, FEAT, RES): os.makedirs(d, exist_ok=True)
""")

code(r"""
# 2) Config
SEEDS         = [0, 1, 2]
N_CLASSES     = 10      # final number of classes kept
N_FRAMES      = 8       # frames sampled uniformly per clip
IMG_SIZE      = 224
SR            = 16000   # audio sample rate
WHISPER_MODEL = 'base'  # 'base' is fast enough for the deadline; 'small' is more accurate
MIN_AUDIO_FRAC = 0.95   # class kept only if >=95% of its clips have audio

# Candidate classes: everyday / sound-producing activities. The final list is chosen from data (audio availability).
CANDIDATES = ['ApplyEyeMakeup','ApplyLipstick','BlowDryHair','BoxingPunchingBag','BrushingTeeth',
              'CuttingInKitchen','Hammering','Haircut','HeadMassage','Knitting','MoppingFloor',
              'PlayingCello','PlayingDaf','PlayingDhol','PlayingFlute','PlayingSitar','ShavingBeard',
              'Typing','WritingOnBoard','Bowling','BasketballDunk','TableTennisShot','SumoWrestling']

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(0)
""")

code(r"""
# 3) Download UCF101 (~6.5 GB) + official splits  (one-time; stored on Drive)
os.chdir(DATA)
if not os.path.exists('UCF101.rar'):
    !wget -q --show-progress https://www.crcv.ucf.edu/data/UCF101/UCF101.rar || wget -q --show-progress --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
if not os.path.exists('ucfTrainTestlist'):
    !wget -q https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip || wget -q --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
    !unzip -q -o UCF101TrainTestSplits-RecognitionTask.zip
# Extract only candidate classes, to local disk (faster than Drive I/O)
VID_DIR = '/content/UCF-101'
for c in CANDIDATES:
    if not os.path.isdir(f'{VID_DIR}/{c}'):
        subprocess.run(['unrar','x','-o+','-inul',f'{DATA}/UCF101.rar',f'UCF-101/{c}/*','/content/'], check=False)
print({c: len(glob.glob(f'{VID_DIR}/{c}/*.avi')) for c in CANDIDATES})
# If the UCF server's certificate fails in your region, download UCF101.rar manually, put it in DATA, and re-run.
""")

md("## 4) Select classes by real audio availability")
code(r"""
def has_audio(path):
    r = subprocess.run(['ffprobe','-v','error','-select_streams','a','-show_entries','stream=index',
                        '-of','csv=p=0',path], capture_output=True, text=True)
    return r.stdout.strip() != ''

audio_stats = {}
for c in tqdm(CANDIDATES):
    vids = sorted(glob.glob(f'{VID_DIR}/{c}/*.avi'))
    audio_stats[c] = np.mean([has_audio(v) for v in vids]) if vids else 0.0
audio_df = pd.Series(audio_stats).sort_values(ascending=False).rename('audio_fraction')
display(audio_df.to_frame())

eligible = [c for c, f in audio_df.items() if f >= MIN_AUDIO_FRAC]
CLASSES = sorted(eligible[:N_CLASSES])
assert len(CLASSES) >= 5, 'Too few classes with audio - lower MIN_AUDIO_FRAC or add candidates'
print('Selected classes:', CLASSES)
""")

md("## 5) Official split 1 (group-disjoint) + group-wise validation split")
code(r"""
def parse_list(fn):
    out = []
    for line in open(fn):
        rel = line.strip().split(' ')[0]
        if rel: out.append(rel)
    return out

train_rel = parse_list(f'{DATA}/ucfTrainTestlist/trainlist01.txt')
test_rel  = parse_list(f'{DATA}/ucfTrainTestlist/testlist01.txt')

def make_df(rels, split):
    rows = []
    for r in rels:
        c = r.split('/')[0]
        if c not in CLASSES: continue
        p = f'{VID_DIR}/{r}'
        if not os.path.exists(p) or not has_audio(p): continue      # drop the few clips without audio (reported below)
        g = int(re.search(r'_g(\d+)_', r).group(1))
        rows.append(dict(path=p, cls=c, label=CLASSES.index(c), group=g, split=split))
    return pd.DataFrame(rows)

df = pd.concat([make_df(train_rel, 'train'), make_df(test_rel, 'test')], ignore_index=True)
# Validation = training groups 8-10 (UCF101 train groups are 8-25; test groups 1-7) -> no actor leakage
df.loc[(df.split == 'train') & (df.group <= 10), 'split'] = 'val'
df['vid'] = df.path.map(lambda p: os.path.splitext(os.path.basename(p))[0])
df.to_csv(f'{RES}/metadata.csv', index=False)
print(df.groupby(['split']).size())
assert not set(df[df.split=='test'].group) & set(df[df.split!='test'].group), 'Group leakage!'
display(pd.crosstab(df.cls, df.split))
""")

md("## 6) Video branch — CNN on RGB frames + CNN on optical-flow images")
code(r"""
import torchvision
from torchvision import transforms as T

cnn = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
cnn.fc = nn.Identity()
cnn = cnn.eval().to(DEVICE)
prep = T.Compose([T.ToPILImage(), T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                  T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def read_frames(path, n=N_FRAMES):
    # Returns n (frame_t, frame_t+1) RGB pairs sampled uniformly; the pair is used for optical flow.
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(total - 2, 0), n).astype(int)
    pairs = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok1, f1 = cap.read(); ok2, f2 = cap.read()
        if not ok1: continue
        if not ok2: f2 = f1
        pairs.append((cv2.cvtColor(f1, cv2.COLOR_BGR2RGB), cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)))
    cap.release()
    return pairs

def flow_image(f1, f2):
    # Dense Farneback optical flow -> HSV encoding (hue = direction, value = magnitude) -> RGB image for the CNN
    g1, g2 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*g1.shape, 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), mag

@torch.no_grad()
def video_features(path):
    pairs = read_frames(path)
    rgb  = torch.stack([prep(a) for a, _ in pairs]).to(DEVICE)
    flws, mags = zip(*[flow_image(a, b) for a, b in pairs])
    flw  = torch.stack([prep(f) for f in flws]).to(DEVICE)
    f_rgb  = cnn(rgb).mean(0).cpu().numpy()          # temporal average pooling -> 2048
    f_flow = cnn(flw).mean(0).cpu().numpy()          # 2048
    return np.concatenate([f_rgb, f_flow]).astype(np.float32)

def cached(name, fn, items):
    fp = f'{FEAT}/{name}.npy'
    if os.path.exists(fp):
        X = np.load(fp)
        if len(X) == len(items): return X
    X = np.stack([fn(x) for x in tqdm(items, desc=name)])
    np.save(fp, X); return X

FV = cached('FV_video', video_features, df.path.tolist())
print('FV', FV.shape)
""")

md("## 7) Audio branch — MFCC + spectral features")
code(r"""
WAV_DIR = '/content/wav'; os.makedirs(WAV_DIR, exist_ok=True)
def to_wav(path):
    out = f'{WAV_DIR}/{os.path.splitext(os.path.basename(path))[0]}.wav'
    if not os.path.exists(out):
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',path,'-vn','-ac','1','-ar',str(SR),out], check=True)
    return out

def stats(m):  # mean and std over time
    return np.concatenate([m.mean(1), m.std(1)])

def audio_features(path):
    y, sr = librosa.load(to_wav(path), sr=SR)
    if len(y) < sr // 4: y = np.pad(y, (0, sr // 4 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feats = [stats(mfcc), stats(librosa.feature.delta(mfcc)), stats(librosa.feature.delta(mfcc, order=2))]  # 240
    spec = np.vstack([librosa.feature.spectral_centroid(y=y, sr=sr),
                      librosa.feature.spectral_bandwidth(y=y, sr=sr),
                      librosa.feature.spectral_rolloff(y=y, sr=sr),
                      librosa.feature.spectral_flatness(y=y),
                      librosa.feature.zero_crossing_rate(y),
                      librosa.feature.rms(y=y)])
    feats += [stats(spec),                                               # 12
              stats(librosa.feature.spectral_contrast(y=y, sr=sr)),      # 14
              librosa.feature.chroma_stft(y=y, sr=sr).mean(1)]           # 12
    return np.nan_to_num(np.concatenate(feats)).astype(np.float32)       # 278

FA = cached('FA_audio', audio_features, df.path.tolist())
print('FA', FA.shape)
""")

md(r"""
## 8) Text branch — Whisper transcripts → TF-IDF / BERT
Whisper tends to invent text on music or silence (e.g. "Thank you for watching"). We drop segments with `no_speech_prob > 0.6`, and clips with no speech get the token `[nospeech]`.
**Expect this modality to be weak.** Most activity clips contain little speech, which is itself a finding for the report.
""")
code(r"""
import whisper
tx_fp = f'{FEAT}/transcripts.json'
transcripts = json.load(open(tx_fp)) if os.path.exists(tx_fp) else {}
todo = [p for p in df.path if os.path.basename(p) not in transcripts]
if todo:
    asr = whisper.load_model(WHISPER_MODEL, device=DEVICE)
    for i, p in enumerate(tqdm(todo, desc='whisper')):
        r = asr.transcribe(to_wav(p), language='en', fp16=(DEVICE=='cuda'), condition_on_previous_text=False)
        txt = ' '.join(s['text'].strip() for s in r['segments'] if s['no_speech_prob'] < 0.6).strip()
        transcripts[os.path.basename(p)] = txt
        if i % 50 == 0: json.dump(transcripts, open(tx_fp, 'w'))
    json.dump(transcripts, open(tx_fp, 'w'))
    del asr; torch.cuda.empty_cache()

df['text'] = df.path.map(lambda p: transcripts[os.path.basename(p)] or '[nospeech]')
df[['vid','cls','text']].to_csv(f'{RES}/transcripts.csv', index=False)
print('Clips with speech: %.1f%%' % (100 * (df.text != '[nospeech]').mean()))
display(df[df.text != '[nospeech]'][['cls','text']].sample(min(10, (df.text!='[nospeech]').sum()), random_state=0))
""")
code(r"""
from transformers import AutoTokenizer, AutoModel
tok  = AutoTokenizer.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained('bert-base-uncased').eval().to(DEVICE)

@torch.no_grad()
def bert_embed(texts, bs=32):
    out = []
    for i in tqdm(range(0, len(texts), bs), desc='bert'):
        enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
        h = bert(**enc).last_hidden_state
        m = enc['attention_mask'].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())   # mean pooling over tokens
    return np.vstack(out).astype(np.float32)

fp = f'{FEAT}/FT_bert.npy'
FT_bert = np.load(fp) if os.path.exists(fp) and len(np.load(fp)) == len(df) else bert_embed(df.text.tolist())
np.save(fp, FT_bert)
print('FT_bert', FT_bert.shape)
# TF-IDF is built inside the experiment loop and fitted on TRAIN ONLY (no vocabulary leakage).
""")

md("## 9) Classifier — fully connected network + Softmax (same model for every experiment)")
code(r"""
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             cohen_kappa_score, confusion_matrix, classification_report)

idx = {s: np.where(df.split.values == s)[0] for s in ('train','val','test')}
y   = df.label.values
C   = len(CLASSES)

def tfidf_block():
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=5000, sublinear_tf=True)
    Xtr = vec.fit_transform(df.text.values[idx['train']])
    Xall = vec.transform(df.text.values)
    print('TF-IDF vocabulary:', Xtr.shape[1])
    if Xtr.shape[1] <= 256:                                       # tiny vocab (little speech) -> use it directly
        return Xall.toarray().astype(np.float32)
    svd = TruncatedSVD(256, random_state=0).fit(Xtr)             # dense 256-d for the MLP
    return svd.transform(Xall).astype(np.float32)

BLOCKS = {'V': FV, 'A': FA, 'T_bert': FT_bert, 'T_tfidf': tfidf_block()}

def build(keys):
    # Standardise each modality block with train statistics, then concatenate: F = [FV, FA, FT]
    parts = []
    for k in keys:
        X = BLOCKS[k]
        sc = StandardScaler().fit(X[idx['train']])
        parts.append(sc.transform(X).astype(np.float32))
    return np.concatenate(parts, 1)

class MLP(nn.Module):
    def __init__(self, d, c, p=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.2),
                                 nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(256, c))       # softmax is applied inside CrossEntropyLoss / at inference
    def forward(self, x): return self.net(x)

def train_eval(X, seed, epochs=200, patience=20, lr=1e-3, wd=1e-3, bs=64):
    set_seed(seed)
    Xt = {s: torch.tensor(X[idx[s]]).to(DEVICE) for s in idx}
    yt = {s: torch.tensor(y[idx[s]]).to(DEVICE) for s in idx}
    model = MLP(X.shape[1], C).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_state, wait = -1, None, 0
    n = len(idx['train'])
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            b = perm[i:i+bs]
            if len(b) < 2: continue
            opt.zero_grad(); lossf(model(Xt['train'][b]), yt['train'][b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vf1 = f1_score(y[idx['val']], model(Xt['val']).argmax(1).cpu().numpy(), average='macro')
        if vf1 > best: best, best_state, wait = vf1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(Xt['test']), 1).cpu().numpy()
    return prob

def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return dict(accuracy=accuracy_score(y_true, y_pred), precision=p, recall=r, f1=f,
                kappa=cohen_kappa_score(y_true, y_pred))
""")

md("## 10) Experiments — single modalities (before fusion) vs fusion, on the identical test set")
code(r"""
EXPERIMENTS = {
    'Video only (CNN + optical flow)':   ['V'],
    'Audio only (MFCC + spectral)':      ['A'],
    'Text only (TF-IDF)':                ['T_tfidf'],
    'Text only (BERT)':                  ['T_bert'],
    'Video + Audio':                     ['V','A'],
    'Video + Text (BERT)':               ['V','T_bert'],
    'Audio + Text (BERT)':               ['A','T_bert'],
    'Fusion [FV, FA, FT] (TF-IDF)':      ['V','A','T_tfidf'],
    'Fusion [FV, FA, FT] (BERT)':        ['V','A','T_bert'],
}
y_test = y[idx['test']]
rows, preds = [], {}
for name, keys in EXPERIMENTS.items():
    X = build(keys)
    runs, probs = [], []
    for s in SEEDS:
        pr = train_eval(X, s); probs.append(pr)
        runs.append(metrics(y_test, pr.argmax(1)))
    preds[name] = np.mean(probs, 0).argmax(1)                # seed-ensemble prediction (for confusion matrix / McNemar)
    m = pd.DataFrame(runs)
    rows.append({'Model': name, 'Dim': X.shape[1],
                 **{f'{k}': f'{m[k].mean():.4f} ± {m[k].std():.4f}' for k in m.columns},
                 '_f1': m.f1.mean()})
    print(f'{name:38s} acc={m.accuracy.mean():.4f}  macroF1={m.f1.mean():.4f}')

results = pd.DataFrame(rows).set_index('Model')
results.drop(columns='_f1').to_csv(f'{RES}/results_table.csv')
display(results.drop(columns='_f1'))
print('Test clips:', len(y_test), '| identical for all models')
""")

code(r"""
# Confusion matrices: each single modality vs full fusion
show = ['Video only (CNN + optical flow)','Audio only (MFCC + spectral)','Text only (BERT)','Fusion [FV, FA, FT] (BERT)']
fig, axes = plt.subplots(1, 4, figsize=(28, 6.5))
for ax, n in zip(axes, show):
    cm = confusion_matrix(y_test, preds[n], normalize='true')
    sns.heatmap(cm, ax=ax, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                xticklabels=CLASSES, yticklabels=CLASSES)
    ax.set_title(n); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout(); plt.savefig(f'{RES}/confusion_matrices.png', dpi=150); plt.show()

print(classification_report(y_test, preds['Fusion [FV, FA, FT] (BERT)'], target_names=CLASSES, digits=4))
""")

md("## 11) Does multimodal fusion improve performance?  (tested, not assumed)")
code(r"""
from scipy.stats import binomtest
single = ['Video only (CNN + optical flow)','Audio only (MFCC + spectral)','Text only (TF-IDF)','Text only (BERT)']
best_single = max(single, key=lambda n: results.loc[n, '_f1'])
fusion = max(['Fusion [FV, FA, FT] (TF-IDF)','Fusion [FV, FA, FT] (BERT)'], key=lambda n: results.loc[n, '_f1'])

a = preds[best_single] == y_test; b = preds[fusion] == y_test
n01, n10 = int((a & ~b).sum()), int((~a & b).sum())        # single right / fusion wrong, and vice versa
p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0   # exact McNemar test

gain = results.loc[fusion, '_f1'] - results.loc[best_single, '_f1']
print(f'Best single modality : {best_single}  (macro-F1 {results.loc[best_single, "_f1"]:.4f})')
print(f'Best fusion          : {fusion}  (macro-F1 {results.loc[fusion, "_f1"]:.4f})')
print(f'Gain                 : {gain:+.4f} macro-F1')
print(f'McNemar: fusion fixes {n10} clips, breaks {n01}; p = {p:.4g}')
print('Conclusion:', 'Fusion SIGNIFICANTLY improves performance (p<0.05).' if (gain > 0 and p < 0.05)
      else 'Improvement is NOT statistically significant on this test set.' if gain > 0
      else 'Fusion does NOT improve over the best single modality.')

# Contribution of each modality (drop-one ablation from full fusion)
abl = {k: results.loc[n, '_f1'] for k, n in [('full', 'Fusion [FV, FA, FT] (BERT)'), ('− text', 'Video + Audio'),
                                             ('− audio', 'Video + Text (BERT)'), ('− video', 'Audio + Text (BERT)')]}
display(pd.Series(abl, name='macro-F1').to_frame())
json.dump({'classes': CLASSES, 'best_single': best_single, 'fusion': fusion, 'gain_f1': gain,
           'mcnemar_p': p, 'n_test': int(len(y_test)), 'n_train': int(len(idx['train'])), 'n_val': int(len(idx['val'])),
           'speech_fraction': float((df.text != '[nospeech]').mean()), 'n_fixed': n10, 'n_broken': n01,
           'ablation_f1': abl}, open(f'{RES}/summary.json', 'w'), indent=2)
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "accelerator": "GPU", "colab": {"provenance": []}}
nbf.write(nb, sys.argv[1])
print('written', len(cells), 'cells')
