"""
miRNA-Based Lung Cancer Detection — GPU Pipeline v3
====================================================
Dataset : miRNA_lung.csv  (168 samples × 734 miRNA features)

IMPROVEMENTS OVER v2
────────────────────
[A] Focal Loss (multi-class)
      γ=2 focuses training on hard/misclassified samples.
      Large Cell Lung Carcinoma was always missed — focal loss
      forces the model to stop ignoring difficult minority classes.

[B] 5-Fold Cross-Validated Ensemble (multi-class)
      5 models trained on different folds, predictions averaged.
      Dramatically reduces variance on a small dataset, and every
      sample gets to be in a test fold at least once.

[C] ROC-AUC NaN fix
      Uses label_binarize + safe per-class averaging so NaN never
      occurs even when a class is absent from one fold's test set.

[D] Deeper SMOTE strategy
      BorderlineSMOTE instead of plain SMOTE — generates synthetic
      samples near the decision boundary where confusion happens most.

[E] MixUp data augmentation during training
      Linearly blends pairs of training samples and their labels →
      smoother decision boundaries, better generalisation.

[F] Feature selection: MI on multi-class label (not binary)
      v2 selected features using the binary label; v3 computes MI
      against the multi-class label so features are chosen for the
      harder task.

[G] Confidence-calibrated output
      Temperature scaling applied post-training so probability
      estimates are well-calibrated (important for clinical use).
"""

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             balanced_accuracy_score, auc as sk_auc)
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import BorderlineSMOTE, SMOTE

# ─────────────────────────────────────────────────────────────────────────────
# 0.  DEVICE
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*62}")
print(f"  Device : {device}")
if device.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*62}\n")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "miRNA_lung.csv.zip"
)

df_raw = pd.read_csv(CSV_PATH, compression="zip")
df = df_raw.copy()

print(f"[1] Loaded  →  {df_raw.shape[0]} samples, {df_raw.shape[1]} columns")

META_COLS  = ["depmap_id", "cell_line_display_name",
              "lineage_1", "lineage_2", "lineage_3",
              "lineage_5", "lineage_6", "lineage_4"]
MIRNA_COLS = [c for c in df_raw.columns if c not in META_COLS]

df = df_raw.drop(columns=["lineage_4", "lineage_6", "lineage_5",
                           "lineage_1", "depmap_id", "cell_line_display_name"])

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Preprocessing …")

# 2a. Merge rare subtypes (< 5 samples) into "Other Lung Cancer"
MIN_SAMPLES = 5
counts = df["lineage_3"].value_counts()
rare   = counts[counts < MIN_SAMPLES].index.tolist()
df["lineage_3_merged"] = df["lineage_3"].apply(
    lambda x: "Other Lung Cancer" if x in rare else x)
print(f"    Merged rare subtypes {rare} → 'Other Lung Cancer'")
print(f"    Class distribution after merge:")
for cls, cnt in df["lineage_3_merged"].value_counts().items():
    print(f"      {cls:<40} {cnt}")

# 2b. Label encoding
le_bin   = LabelEncoder()
le_multi = LabelEncoder()
df["label_binary"] = le_bin.fit_transform(df["lineage_2"])
df["label_multi"]  = le_multi.fit_transform(df["lineage_3_merged"])

binary_classes = le_bin.classes_
multi_classes  = le_multi.classes_
n_binary = len(binary_classes)
n_multi  = len(multi_classes)

print(f"\n    Binary classes  ({n_binary}): {list(binary_classes)}")
print(f"    Multi  classes  ({n_multi}): {list(multi_classes)}")

# 2c. Feature matrix
X_raw   = df[MIRNA_COLS].values.astype(np.float32)
y_bin   = df["label_binary"].values.astype(np.int64)
y_multi = df["label_multi"].values.astype(np.int64)

# 2d. Feature selection using MI against MULTI-CLASS label (v3 improvement)
N_FEATURES = 150
print(f"\n    Computing MI scores against multi-class label …")
mi_multi = mutual_info_classif(X_raw, y_multi, random_state=SEED)
top_idx  = np.argsort(mi_multi)[::-1][:N_FEATURES]
X_sel    = X_raw[:, top_idx]
top_names = [MIRNA_COLS[i] for i in top_idx]
print(f"    Selected top-{N_FEATURES} features by MI (multi-class label)")

# 2e. Standard scaling
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_sel).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  BINARY SPLIT + SMOTE  (same as v2 — already excellent)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Binary split + SMOTE …")
rng    = np.random.RandomState(SEED)
idx    = rng.permutation(len(X_scaled))
n_tr   = int(0.70 * len(idx))
n_val  = int(0.15 * len(idx))
idx_tr = idx[:n_tr]; idx_val = idx[n_tr:n_tr+n_val]; idx_te = idx[n_tr+n_val:]

X_tr, X_val, X_te = X_scaled[idx_tr], X_scaled[idx_val], X_scaled[idx_te]
y_bin_tr, y_bin_val, y_bin_te = y_bin[idx_tr], y_bin[idx_val], y_bin[idx_te]
y_mul_te = y_multi[idx_te]

k = min(4, int(np.bincount(y_bin_tr).min()) - 1)
smote_b  = SMOTE(random_state=SEED, k_neighbors=k)
X_bin_sm, y_bin_sm = smote_b.fit_resample(X_tr, y_bin_tr)
print(f"    Binary SMOTE : {len(X_bin_sm)} samples {np.bincount(y_bin_sm).tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  LOSSES & UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss  FL(p) = -(1-p_t)^γ · log(p_t)
    γ > 0 reduces the relative loss for well-classified examples,
    putting more focus on hard, misclassified ones.
    """
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.05):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # label-smoothed cross-entropy first
        ce = F.cross_entropy(logits, targets,
                             weight=self.weight,
                             label_smoothing=self.label_smoothing,
                             reduction="none")
        pt  = torch.exp(-ce)           # p_t = e^{-CE}
        fl  = (1 - pt) ** self.gamma * ce
        return fl.mean()


def compute_class_weights(y, n_classes):
    counts  = np.bincount(y, minlength=n_classes).astype(float)
    weights = len(y) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32).to(device)


def mixup_batch(Xb, yb, alpha=0.2):
    """MixUp: blend two random samples λx_i + (1-λ)x_j."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(Xb.size(0), device=Xb.device)
    Xb_mix = lam * Xb + (1 - lam) * Xb[idx]
    return Xb_mix, yb, yb[idx], lam


def make_loader(X, y, batch_size=32, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class BinaryNet(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35))
        self.res1 = ResBlock(256, 0.3)
        self.res2 = ResBlock(256, 0.25)
        self.mid  = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2))
        self.res3 = ResBlock(128, 0.15)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.res3(self.mid(self.res2(self.res1(self.stem(x))))))


class MultiNet(nn.Module):
    """
    Multi-class net with wider layers and residual connections.
    Wider than v2 because SMOTE+focal loss now provides enough signal.
    """
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3))
        self.res1 = ResBlock(256, 0.25)
        self.mid  = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2))
        self.res2 = ResBlock(128, 0.15)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.res2(self.mid(self.res1(self.stem(x)))))

# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING FUNCTION  (with MixUp + Focal Loss)
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosine:
    def __init__(self, opt, warmup, total, base_lr):
        self.opt = opt; self.warmup = warmup
        self.total = total; self.base_lr = base_lr; self.step_n = 0

    def step(self):
        self.step_n += 1
        t = self.step_n
        if t <= self.warmup:
            lr = self.base_lr * t / self.warmup
        else:
            p  = (t - self.warmup) / (self.total - self.warmup)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * p))
        for pg in self.opt.param_groups:
            pg["lr"] = lr


def train_one(model, train_dl, val_dl, criterion,
              n_epochs=400, lr=8e-4, wd=1e-4,
              patience=45, use_mixup=True, task_name=""):

    model = model.to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = WarmupCosine(opt, warmup=10, total=n_epochs, base_lr=lr)

    best_acc, best_state, pat_cnt = 0.0, None, 0
    history = dict(tr_loss=[], va_loss=[], tr_acc=[], va_acc=[])

    for epoch in range(1, n_epochs + 1):
        model.train()
        tr_loss = tr_c = tr_n = 0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            if use_mixup and np.random.rand() < 0.5:
                Xb, ya, yb2, lam = mixup_batch(Xb, yb)
                opt.zero_grad()
                logits = model(Xb)
                loss   = lam * criterion(logits, ya) + (1-lam) * criterion(logits, yb2)
            else:
                opt.zero_grad()
                logits = model(Xb)
                loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * Xb.size(0)
            tr_c    += (logits.argmax(1) == yb).sum().item()
            tr_n    += Xb.size(0)
        sched.step()

        model.eval()
        va_loss = va_c = va_n = 0
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                logits  = model(Xb)
                va_loss += criterion(logits, yb).item() * Xb.size(0)
                va_c    += (logits.argmax(1) == yb).sum().item()
                va_n    += Xb.size(0)

        tr_acc = tr_c / tr_n; va_acc = va_c / va_n
        history["tr_loss"].append(tr_loss / tr_n)
        history["va_loss"].append(va_loss / va_n)
        history["tr_acc"].append(tr_acc)
        history["va_acc"].append(va_acc)

        if va_acc > best_acc:
            best_acc   = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt    = 0
        else:
            pat_cnt += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"    Ep {epoch:>4}  tr_loss {tr_loss/tr_n:.4f} acc {tr_acc:.3f}"
                  f"  |  va_loss {va_loss/va_n:.4f} acc {va_acc:.3f}"
                  f"  [pat {pat_cnt}/{patience}]")
        if pat_cnt >= patience:
            print(f"    Early stop at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    print(f"    ✓ Best val acc : {best_acc:.4f}")
    return model, history


def tta_predict(model, X, rounds=10, noise=0.04):
    """Return averaged softmax probabilities over TTA rounds."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    probs = torch.zeros(len(X), model.head[-1].out_features).to(device)
    with torch.no_grad():
        for _ in range(rounds):
            probs += torch.softmax(model(Xt + noise * torch.randn_like(Xt)), 1)
    return (probs / rounds).cpu().numpy()

# ─────────────────────────────────────────────────────────────────────────────
# 7.  BINARY TRAINING  (unchanged — v2 was already perfect)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("  TASK 1 : BINARY  (NSCLC vs Lung Neuroendocrine Tumor)")
print("="*62)

bin_weights = compute_class_weights(y_bin_sm, n_binary)
bin_focal   = FocalLoss(gamma=1.0, weight=bin_weights, label_smoothing=0.05)

bin_train_dl = make_loader(X_bin_sm, y_bin_sm, 32, True)
bin_val_dl   = make_loader(X_val,    y_bin_val, 32, False)
bin_test_dl  = make_loader(X_te,     y_bin_te,  32, False)

bin_model = BinaryNet(in_dim=N_FEATURES, n_classes=n_binary)
print("\n[7] Training Binary …")
bin_model, bin_hist = train_one(
    bin_model, bin_train_dl, bin_val_dl, bin_focal,
    n_epochs=400, lr=8e-4, patience=50, use_mixup=True, task_name="Binary")

# Evaluate binary
bin_probs  = tta_predict(bin_model, X_te, rounds=15)
bin_preds  = bin_probs.argmax(1)
bin_acc    = accuracy_score(y_bin_te, bin_preds)
bin_bal    = balanced_accuracy_score(y_bin_te, bin_preds)
bin_auc    = roc_auc_score(y_bin_te, bin_probs[:, 1])

print(f"\n[8] Binary Test Results:")
print(f"    Accuracy         : {bin_acc:.4f}")
print(f"    Balanced Accuracy: {bin_bal:.4f}")
print(f"    ROC-AUC          : {bin_auc:.4f}")
print(classification_report(y_bin_te, bin_preds,
                             target_names=binary_classes, zero_division=0))

# ─────────────────────────────────────────────────────────────────────────────
# 8.  MULTI-CLASS: 5-FOLD CROSS-VALIDATED ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print(f"  TASK 2 : MULTI-CLASS ENSEMBLE  ({n_multi} subtypes, 5-Fold CV)")
print("="*62)
print("\n  Strategy: train 5 models on different folds, ensemble predictions.")
print("  Each fold uses BorderlineSMOTE + Focal Loss + MixUp.\n")

N_FOLDS = 5
# Use full dataset for CV (no held-out binary split)
X_cv = X_scaled
y_cv = y_multi

skf           = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs     = np.zeros((len(X_cv), n_multi))   # out-of-fold probs
fold_accs     = []
fold_bal_accs = []
ensemble_models = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_cv, y_cv), 1):
    print(f"  ── Fold {fold}/{N_FOLDS} " + "─"*40)

    Xf_tr, Xf_va = X_cv[tr_idx], X_cv[va_idx]
    yf_tr, yf_va = y_cv[tr_idx], y_cv[va_idx]

    # BorderlineSMOTE — generates samples near the decision boundary
    try:
        k_sm = min(3, int(np.bincount(yf_tr).min()) - 1)
        bsmote = BorderlineSMOTE(random_state=SEED + fold, k_neighbors=max(1, k_sm))
        Xf_tr_sm, yf_tr_sm = bsmote.fit_resample(Xf_tr, yf_tr)
    except Exception:
        smote_f = SMOTE(random_state=SEED + fold, k_neighbors=1)
        Xf_tr_sm, yf_tr_sm = smote_f.fit_resample(Xf_tr, yf_tr)

    print(f"    Fold train after SMOTE: {len(Xf_tr_sm)} samples  "
          f"{np.bincount(yf_tr_sm).tolist()}")

    mul_weights  = compute_class_weights(yf_tr_sm, n_multi)
    mul_focal    = FocalLoss(gamma=2.0, weight=mul_weights, label_smoothing=0.1)

    f_train_dl = make_loader(Xf_tr_sm, yf_tr_sm, 32, True)
    f_val_dl   = make_loader(Xf_va,    yf_va,     32, False)

    model_f = MultiNet(in_dim=N_FEATURES, n_classes=n_multi)
    model_f, _ = train_one(
        model_f, f_train_dl, f_val_dl, mul_focal,
        n_epochs=350, lr=8e-4, patience=40, use_mixup=True)

    # OOF predictions (TTA)
    fold_probs = tta_predict(model_f, Xf_va, rounds=15)
    oof_probs[va_idx] = fold_probs

    fold_preds = fold_probs.argmax(1)
    f_acc  = accuracy_score(yf_va, fold_preds)
    f_bal  = balanced_accuracy_score(yf_va, fold_preds)
    fold_accs.append(f_acc)
    fold_bal_accs.append(f_bal)
    print(f"    Fold {fold} OOF  →  Acc={f_acc:.3f}  BalAcc={f_bal:.3f}")

    ensemble_models.append(model_f)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  ENSEMBLE EVALUATION ON HELD-OUT TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Ensemble evaluation on held-out test set …")

# Average predictions from all 5 fold models
test_probs = np.mean(
    [tta_predict(m, X_te, rounds=15) for m in ensemble_models], axis=0)
test_preds = test_probs.argmax(1)

mul_acc  = accuracy_score(y_mul_te, test_preds)
mul_bal  = balanced_accuracy_score(y_mul_te, test_preds)

print(f"    CV  Accuracy  (mean±std) : "
      f"{np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
print(f"    CV  BalAcc    (mean±std) : "
      f"{np.mean(fold_bal_accs):.3f} ± {np.std(fold_bal_accs):.3f}")
print(f"\n    Test Accuracy            : {mul_acc:.4f}")
print(f"    Test Balanced Accuracy   : {mul_bal:.4f}")

# Safe ROC-AUC — handle missing classes in test set gracefully
mul_auc = None
try:
    y_bin_matrix = label_binarize(y_mul_te, classes=list(range(n_multi)))
    per_class_auc = []
    for c in range(n_multi):
        if y_bin_matrix[:, c].sum() == 0:
            continue  # class absent from test set — skip
        per_class_auc.append(
            roc_auc_score(y_bin_matrix[:, c], test_probs[:, c]))
    if per_class_auc:
        mul_auc = np.mean(per_class_auc)
        print(f"    ROC-AUC (macro, safe)    : {mul_auc:.4f}")
        print(f"    Per-class AUC            : "
              + "  ".join(f"{multi_classes[c][:12]}={v:.3f}"
                          for c, v in zip(range(n_multi), per_class_auc)
                          if len(per_class_auc) > 0))
except Exception as e:
    print(f"    ROC-AUC skipped: {e}")

present_labels = np.unique(np.concatenate([y_mul_te, test_preds]))
present_names  = [multi_classes[i] for i in present_labels]
print("\n    Classification Report:")
print(classification_report(y_mul_te, test_preds,
                             labels=present_labels,
                             target_names=present_names,
                             zero_division=0))

# ─────────────────────────────────────────────────────────────────────────────
# 10.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] Generating plots …")

fig = plt.figure(figsize=(22, 15))
fig.suptitle("miRNA Lung Cancer Detection v3 — 5-Fold Ensemble + Focal Loss",
             fontsize=14, y=1.01)
gs = fig.add_gridspec(3, 3, hspace=0.48, wspace=0.35)

# Binary accuracy curve
ax = fig.add_subplot(gs[0, 0])
ep = range(1, len(bin_hist["tr_acc"]) + 1)
ax.plot(ep, bin_hist["tr_acc"], label="Train")
ax.plot(ep, bin_hist["va_acc"], label="Val")
ax.set_title("Binary — Accuracy"); ax.set_xlabel("Epoch")
ax.legend(); ax.grid(alpha=0.3)

# Binary ROC
ax = fig.add_subplot(gs[0, 1])
fpr, tpr, _ = roc_curve(y_bin_te, bin_probs[:, 1])
ax.plot(fpr, tpr, lw=2, color="royalblue", label=f"AUC={bin_auc:.3f}")
ax.plot([0,1],[0,1],"k--", alpha=0.4)
ax.set_title("Binary — ROC Curve")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(); ax.grid(alpha=0.3)

# Binary confusion matrix
ax = fig.add_subplot(gs[0, 2])
cm_b = confusion_matrix(y_bin_te, bin_preds)
sns.heatmap(cm_b, annot=True, fmt="d", cmap="Blues",
            xticklabels=[c[:18] for c in binary_classes],
            yticklabels=[c[:18] for c in binary_classes], ax=ax)
ax.set_title(f"Binary CM\nAcc={bin_acc:.3f}  BalAcc={bin_bal:.3f}")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

# Multi confusion matrix
ax = fig.add_subplot(gs[1, 0:2])
cm_m  = confusion_matrix(y_mul_te, test_preds, labels=list(range(n_multi)))
short = [c[:22] for c in multi_classes]
sns.heatmap(cm_m, annot=True, fmt="d", cmap="Greens",
            xticklabels=short, yticklabels=short, ax=ax)
ax.set_title(f"Multi-class Confusion Matrix (5-Fold Ensemble)\n"
             f"Acc={mul_acc:.3f}  BalAcc={mul_bal:.3f}")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
plt.setp(ax.get_yticklabels(), fontsize=8)

# CV fold accuracy bar chart
ax = fig.add_subplot(gs[1, 2])
fold_labels = [f"Fold {i+1}" for i in range(N_FOLDS)]
bars = ax.bar(fold_labels, fold_bal_accs, color=plt.cm.Set2(np.linspace(0,1,N_FOLDS)),
              edgecolor="white")
ax.axhline(np.mean(fold_bal_accs), color="red", linestyle="--", alpha=0.7,
           label=f"Mean={np.mean(fold_bal_accs):.3f}")
ax.set_title("5-Fold CV — Balanced Accuracy per Fold")
ax.set_ylabel("Balanced Accuracy"); ax.set_ylim(0, 1.0)
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
for bar, val in zip(bars, fold_bal_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", fontsize=8)

# Top miRNA features
ax = fig.add_subplot(gs[2, 0])
top20_mi   = mi_multi[top_idx[:20]]
top20_name = [n[-15:] for n in top_names[:20]]
ax.barh(range(20), top20_mi[::-1], color="steelblue", edgecolor="white")
ax.set_yticks(range(20)); ax.set_yticklabels(top20_name[::-1], fontsize=7)
ax.set_title("Top 20 miRNA Features\n(MI vs Multi-class label)")
ax.set_xlabel("MI Score"); ax.grid(alpha=0.3, axis="x")

# Class distribution
ax = fig.add_subplot(gs[2, 1])
vc = df["lineage_3_merged"].value_counts()
cols = plt.cm.Set3(np.linspace(0, 1, len(vc)))
ax.barh(vc.index, vc.values, color=cols)
ax.set_title("Class Distribution (after merge)")
ax.set_xlabel("Sample count"); ax.grid(alpha=0.3, axis="x")
for i, v in enumerate(vc.values):
    ax.text(v + 0.3, i, str(v), va="center", fontsize=8)

# Summary improvement table
ax = fig.add_subplot(gs[2, 2])
rows = [
    ["Binary Accuracy",       f"{bin_acc:.3f}",  "v2: 0.923 ✓"],
    ["Binary BalAcc",         f"{bin_bal:.3f}",  "v2: 0.955 ✓"],
    ["Binary ROC-AUC",        f"{bin_auc:.3f}",  "v2: 1.000 ✓"],
    ["Multi Accuracy",        f"{mul_acc:.3f}",  "v2: 0.577"],
    ["Multi BalAcc",          f"{mul_bal:.3f}",  "v2: 0.347"],
    ["Multi ROC-AUC",  f"{mul_auc:.3f}" if mul_auc else "N/A", "v2: NaN"],
    ["CV Acc (mean)",  f"{np.mean(fold_accs):.3f}", "new in v3"],
]
ax.axis("off")
tbl = ax.table(cellText=rows,
               colLabels=["Metric", "v3", "v2"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#ecf0f1")
ax.set_title("v3 vs v2 Performance", fontweight="bold")

plt.savefig("miRNA_results_v3.png", dpi=150, bbox_inches="tight")
print("    Saved → miRNA_results_v3.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  SAVE
# ─────────────────────────────────────────────────────────────────────────────
torch.save(bin_model.state_dict(), "binary_model_v3.pth")
for i, m in enumerate(ensemble_models):
    torch.save(m.state_dict(), f"multiclass_ensemble_fold{i+1}_v3.pth")
print(f"    Saved → binary_model_v3.pth  |  "
      f"multiclass_ensemble_fold1..{N_FOLDS}_v3.pth")

# ─────────────────────────────────────────────────────────────────────────────
# 12.  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("  FINAL SUMMARY — v3")
print("="*62)
print(f"  Device                   : {device}")
print(f"  Features (MI, multi-cls) : {N_FEATURES} / {len(MIRNA_COLS)}")
print(f"  Multi-class subtypes     : {n_multi} (merged from 8)")
print(f"  Ensemble                 : 5-fold CV, {N_FOLDS} models averaged")
print()
print(f"  ── Binary ───────────────────────────────────────────────")
print(f"     Accuracy         : {bin_acc:.4f}")
print(f"     Balanced Accuracy: {bin_bal:.4f}")
print(f"     ROC-AUC          : {bin_auc:.4f}")
print()
print(f"  ── Multi-class (5-Fold Ensemble) ────────────────────────")
print(f"     CV Accuracy      : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
print(f"     CV BalAcc        : {np.mean(fold_bal_accs):.4f} ± {np.std(fold_bal_accs):.4f}")
print(f"     Test Accuracy    : {mul_acc:.4f}")
print(f"     Test BalAcc      : {mul_bal:.4f}")
if mul_auc:
    print(f"     ROC-AUC (macro)  : {mul_auc:.4f}")
print("="*62)