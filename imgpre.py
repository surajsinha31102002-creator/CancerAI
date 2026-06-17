"""
Lung & Colon Cancer Image Classifier — GPU Pipeline v2
=======================================================

5 Classes:
  colon_aca  → Colon Adenocarcinoma          (CANCER)
  colon_n    → Normal Colon Tissue           (NORMAL)
  lung_aca   → Lung Adenocarcinoma           (CANCER)
  lung_n     → Normal Lung Tissue            (NORMAL)
  lung_scc   → Lung Squamous Cell Carcinoma  (CANCER)

ZIP structure handled:
  dataset.zip
    └── colon_image_sets/
          ├── colon_aca/   ← 5 class folders found here
          ├── colon_n/
    └── lung_image_sets/
          ├── lung_aca/
          ├── lung_n/
          └── lung_scc/

2-Stage Prediction Logic:
  Stage 1 → Cancer present? (CANCER / NORMAL)
  Stage 2 → Which cancer type?

Usage:
  python cancer_img_classifier.py --zip dataset.zip
  python cancer_img_classifier.py --zip dataset.zip --epochs 30 --batch 64
  python cancer_img_classifier.py --predict-only
"""

import os, sys, json, zipfile, shutil, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.cuda.amp import GradScaler, autocast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, balanced_accuracy_score)

# ─────────────────────────────────────────────────────────────────────────────
# CLASS METADATA
# ─────────────────────────────────────────────────────────────────────────────
CLASS_META = {
    "colon_aca": {"label": "Colon Adenocarcinoma",
                  "organ": "Colon", "status": "CANCER", "emoji": "🔴"},
    "colon_n":   {"label": "Normal Colon Tissue",
                  "organ": "Colon", "status": "NORMAL", "emoji": "🟢"},
    "lung_aca":  {"label": "Lung Adenocarcinoma",
                  "organ": "Lung",  "status": "CANCER", "emoji": "🔴"},
    "lung_n":    {"label": "Normal Lung Tissue",
                  "organ": "Lung",  "status": "NORMAL", "emoji": "🟢"},
    "lung_scc":  {"label": "Lung Squamous Cell Carcinoma",
                  "organ": "Lung",  "status": "CANCER", "emoji": "🔴"},
}

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 25
LR           = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 8
NUM_WORKERS  = 4
SEED         = 42
EXTS         = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SCRIPT_DIR  = Path(__file__).parent.resolve()
EXTRACT_DIR = SCRIPT_DIR / "dataset_extracted"
MODEL_PATH  = SCRIPT_DIR / "best_model.pth"
NAMES_PATH  = SCRIPT_DIR / "class_names.json"
PLOT_PATH   = SCRIPT_DIR / "training_results.png"

torch.manual_seed(SEED)
np.random.seed(SEED)

# Device — module-level so all functions can use it;
# banner only printed from __main__ to avoid worker subprocess noise.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  UNZIP  — drills into nested folder structure to find all 5 class dirs
# ─────────────────────────────────────────────────────────────────────────────
def _has_images(folder: Path) -> bool:
    return any(f.suffix.lower() in EXTS
               for f in folder.iterdir() if f.is_file())


def _collect_class_dirs(root: Path):
    """
    Recursively walk `root` and collect every folder that directly
    contains image files. Works regardless of nesting depth.
    Example:
      root/colon_image_sets/colon_aca/*.jpg  → colon_aca collected
      root/lung_image_sets/lung_aca/*.jpg    → lung_aca  collected
    """
    class_dirs = []
    for child in sorted(root.iterdir()):
        if child.name.startswith(".") or not child.is_dir():
            continue
        if _has_images(child):
            class_dirs.append(child)
        else:
            class_dirs.extend(_collect_class_dirs(child))
    return class_dirs


def unzip_dataset(zip_path: str) -> list:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True)

    print(f"[1] Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(EXTRACT_DIR)

    class_dirs = _collect_class_dirs(EXTRACT_DIR)

    if not class_dirs:
        raise RuntimeError("No image folders found inside ZIP. "
                           "Check the ZIP structure.")

    print(f"\n    Found {len(class_dirs)} class folder(s):")
    for d in class_dirs:
        n_imgs = sum(1 for f in d.rglob("*") if f.suffix.lower() in EXTS)
        meta   = CLASS_META.get(d.name.lower(), {})
        label  = meta.get("label",  d.name)
        status = meta.get("status", "?")
        emoji  = meta.get("emoji",  "")
        print(f"      {emoji} {d.name:<14}  {label:<38}  [{status}]  {n_imgs:>6} images")

    return class_dirs          # list[Path], sorted alphabetically


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET
# ─────────────────────────────────────────────────────────────────────────────
class CancerDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples   # list of (Path, int_label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 128)
        if self.transform:
            img = self.transform(img)
        return img, label


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25,
                           saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_datasets(class_dirs: list):
    class_names = [d.name for d in class_dirs]   # sorted alphabetically

    all_samples = []
    for label_idx, cls_dir in enumerate(class_dirs):
        paths = [p for p in cls_dir.rglob("*") if p.suffix.lower() in EXTS]
        all_samples += [(p, label_idx) for p in paths]

    labels_arr = [s[1] for s in all_samples]
    idx        = np.arange(len(all_samples))

    idx_tr, idx_tmp, _, y_tmp = train_test_split(
        idx, labels_arr, test_size=0.30,
        stratify=labels_arr, random_state=SEED)
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

    tr_ds  = CancerDataset([all_samples[i] for i in idx_tr],  train_tf)
    val_ds = CancerDataset([all_samples[i] for i in idx_val], val_tf)
    te_ds  = CancerDataset([all_samples[i] for i in idx_te],  val_tf)

    print(f"\n[2] Split  →  Train {len(tr_ds):,} | Val {len(val_ds):,} | Test {len(te_ds):,}")
    print(f"    Classes: {class_names}\n")
    return tr_ds, val_ds, te_ds, class_names


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODEL  — EfficientNet-B3
# ─────────────────────────────────────────────────────────────────────────────
def build_model(n_classes: int):
    model = models.efficientnet_b3(
        weights=models.EfficientNet_B3_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.features[6:].parameters():
        p.requires_grad = True
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_feat, 512),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 128),
        nn.SiLU(),
        nn.Linear(128, n_classes),
    )
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, tr_dl, val_dl, n_epochs):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6)
    scaler = GradScaler()

    history  = dict(tr_loss=[], va_loss=[], tr_acc=[], va_acc=[])
    best_acc = 0.0
    pat_cnt  = 0

    print(f"[3] Training EfficientNet-B3  "
          f"({n_epochs} epochs · fp16 · RTX optimized) …\n")

    for epoch in range(1, n_epochs + 1):
        # train
        model.train()
        tr_loss = tr_c = tr_n = 0
        for imgs, labels in tqdm(tr_dl,
                                 desc=f"Ep {epoch:>3}/{n_epochs} [train]",
                                 leave=False, ncols=80):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * imgs.size(0)
            tr_c    += (logits.detach().argmax(1) == labels).sum().item()
            tr_n    += imgs.size(0)
        scheduler.step()

        # validate
        model.eval()
        va_loss = va_c = va_n = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_dl,
                                     desc=f"Ep {epoch:>3}/{n_epochs} [ val ]",
                                     leave=False, ncols=80):
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast():
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                va_loss += loss.item() * imgs.size(0)
                va_c    += (logits.argmax(1) == labels).sum().item()
                va_n    += imgs.size(0)

        tr_acc = tr_c / tr_n
        va_acc = va_c / va_n
        history["tr_loss"].append(tr_loss / tr_n)
        history["va_loss"].append(va_loss / va_n)
        history["tr_acc"].append(tr_acc)
        history["va_acc"].append(va_acc)

        saved = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), MODEL_PATH)
            saved   = "  ← best ✓"
            pat_cnt = 0
        else:
            pat_cnt += 1

        print(f"  Ep {epoch:>3}/{n_epochs}  "
              f"tr {tr_loss/tr_n:.4f}/{tr_acc:.4f}  |  "
              f"va {va_loss/va_n:.4f}/{va_acc:.4f}  "
              f"[pat {pat_cnt}/{PATIENCE}]{saved}")

        if pat_cnt >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}.")
            break

    print(f"\n  ✓ Best val accuracy : {best_acc:.4f}  →  {MODEL_PATH}\n")
    return history, best_acc


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, te_dl, class_names):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(te_dl, desc="Testing", ncols=80):
            imgs = imgs.to(device)
            with autocast():
                logits = model(imgs)
            probs = torch.softmax(logits, 1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(probs.argmax(1))
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    acc     = (all_labels == all_preds).mean()
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    print(f"\n[4] Test Accuracy         : {acc:.4f}")
    print(f"    Balanced Accuracy      : {bal_acc:.4f}")
    try:
        auc = roc_auc_score(all_labels, all_probs,
                            multi_class="ovr", average="macro")
        print(f"    ROC-AUC (macro OvR)   : {auc:.4f}")
    except Exception:
        auc = None

    print("\n    Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, zero_division=0))
    return all_labels, all_preds, all_probs, acc, bal_acc, auc


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def save_plots(history, all_labels, all_preds, class_names, acc, bal_acc, auc):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Cancer Image Classifier — EfficientNet-B3 (RTX GPU)", fontsize=13)

    ep = range(1, len(history["tr_acc"]) + 1)
    axes[0].plot(ep, history["tr_acc"], label="Train", lw=2)
    axes[0].plot(ep, history["va_acc"], label="Val",   lw=2)
    axes[0].set_title("Accuracy Curve"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["tr_loss"], label="Train", lw=2)
    axes[1].plot(ep, history["va_loss"], label="Val",   lw=2)
    axes[1].set_title("Loss Curve"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)

    cm = confusion_matrix(all_labels, all_preds)
    short = [c.replace("_", "\n") for c in class_names]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=short, yticklabels=short,
                ax=axes[2], linewidths=0.5)
    title = f"Confusion Matrix\nAcc={acc:.4f}  BalAcc={bal_acc:.4f}"
    if auc: title += f"  AUC={auc:.4f}"
    axes[2].set_title(title)
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"    Plot saved → {PLOT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  INFERENCE  (used by Gradio)
# ─────────────────────────────────────────────────────────────────────────────
infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tta_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])


def predict_image(pil_img: Image.Image, model, class_names, tta_rounds=8):
    model.eval()
    pil_rgb   = pil_img.convert("RGB")
    probs_sum = torch.zeros(len(class_names)).to(device)

    with torch.no_grad():
        for i in range(tta_rounds):
            src = pil_rgb if i == 0 else tta_tf(pil_rgb)
            inp = infer_tf(src).unsqueeze(0).to(device)
            with autocast():
                logits = model(inp)
            probs_sum += torch.softmax(logits[0], 0)

    probs    = (probs_sum / tta_rounds).cpu().numpy()
    pred_idx = int(probs.argmax())
    return probs, class_names[pred_idx], float(probs[pred_idx])


# ─────────────────────────────────────────────────────────────────────────────
# 8.  GRADIO APP
# ─────────────────────────────────────────────────────────────────────────────
def launch_gradio(class_names, model):
    try:
        import gradio as gr
    except ImportError:
        os.system(f"{sys.executable} -m pip install gradio --quiet")
        import gradio as gr

    def gradio_predict(img_np):
        if img_np is None:
            return ({c: 0.0 for c in class_names},
                    "### ⚠️ No image uploaded. Please upload a histopathology image.",
                    "—")

        pil_img           = Image.fromarray(img_np)
        probs, pred_cls, conf = predict_image(pil_img, model, class_names)
        conf_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        meta   = CLASS_META.get(pred_cls, {})
        status = meta.get("status", "UNKNOWN")
        label  = meta.get("label",  pred_cls)
        organ  = meta.get("organ",  "—")
        emoji  = meta.get("emoji",  "🔍")
        pct    = conf * 100

        tier = ("🔒 Very High Confidence" if pct >= 90 else
                "✅ High Confidence"       if pct >= 75 else
                "🔶 Moderate Confidence"   if pct >= 55 else
                "❓ Low Confidence — please consult a specialist")

        if status == "NORMAL":
            s1_line = "## 🟢 Stage 1 Result: NO CANCER DETECTED"
            s2_line = f"**Tissue type:** {label}"
        else:
            s1_line = "## 🔴 Stage 1 Result: CANCER DETECTED"
            s2_line = f"**Cancer type:** {emoji} {label}"

        verdict = f"""
{s1_line}

---

### Stage 2 — Detailed Classification

{s2_line}

**Organ:** {organ}

**Model Confidence:** {pct:.1f}% — {tier}

---
> ⚠️ *Research use only. This is not a medical diagnosis.*
"""
        summary = f"{status} | {label} | {pct:.1f}% confidence"
        return conf_dict, verdict, summary

    # ── UI ────────────────────────────────────────────────────────────────
    with gr.Blocks(title="Cancer Histopathology Classifier",
                   theme=gr.themes.Soft(primary_hue="blue")) as demo:

        gr.Markdown("""
# 🏥 Lung & Colon Cancer Histopathology Classifier
**AI-powered · EfficientNet-B3 · Trained on RTX 5070**

Upload a histopathology image — the model will:
1. **Stage 1:** Determine if cancer is present or absent
2. **Stage 2:** If cancer detected, identify the exact type
---
""")
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(
                    label="📂 Upload Histopathology Image (drag & drop)",
                    type="numpy", height=360)
                predict_btn = gr.Button("🔬 Analyse Image",
                                        variant="primary", size="lg")
                gr.Markdown("""
| Class | Description | Status |
|---|---|---|
| `colon_aca` | Colon Adenocarcinoma | 🔴 Cancer |
| `colon_n`   | Normal Colon Tissue  | 🟢 Normal |
| `lung_aca`  | Lung Adenocarcinoma  | 🔴 Cancer |
| `lung_n`    | Normal Lung Tissue   | 🟢 Normal |
| `lung_scc`  | Lung Squamous Cell Carcinoma | 🔴 Cancer |
""")
            with gr.Column(scale=1):
                verdict_out = gr.Markdown(
                    "*Upload an image to see the prediction.*")
                conf_out = gr.Label(
                    label="📊 All Class Probabilities",
                    num_top_classes=5)
                summary_out = gr.Textbox(
                    label="Quick Summary", interactive=False, lines=1)

        gr.Markdown("""---
### ℹ️ Model Info
| Property | Value |
|---|---|
| Architecture | EfficientNet-B3 (ImageNet pretrained) |
| Input size | 224 × 224 px |
| Training | GPU · Mixed Precision fp16 |
| Inference | 8× Test-Time Augmentation |
| Classes | 5 (3 cancer + 2 normal) |
""")

        predict_btn.click(fn=gradio_predict, inputs=[img_input],
                          outputs=[conf_out, verdict_out, summary_out])
        img_input.change(fn=gradio_predict, inputs=[img_input],
                         outputs=[conf_out, verdict_out, summary_out])

    print("\n" + "="*62)
    print("  Launching Gradio App — open the URL below in your browser")
    print("="*62 + "\n")
    demo.launch(share=False, inbrowser=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN  — everything inside this guard so DataLoader workers don't re-run it
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Print device banner once (not in worker subprocesses)
    print(f"\n{'='*62}")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'='*62}\n")

    parser = argparse.ArgumentParser(
        description="5-class Cancer Histopathology Classifier (GPU)")
    parser.add_argument("--zip",          type=str, default=None)
    parser.add_argument("--epochs",       type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch",        type=int, default=BATCH_SIZE)
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()

    # ── Predict-only ──────────────────────────────────────────────────────
    if args.predict_only:
        if not MODEL_PATH.exists() or not NAMES_PATH.exists():
            print("ERROR: No saved model found. Train first.")
            sys.exit(1)
        with open(NAMES_PATH) as f:
            class_names = json.load(f)
        print(f"[→] Loaded classes: {class_names}")
        _model = build_model(len(class_names))
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        launch_gradio(class_names, _model)
        sys.exit(0)

    # ── Training pipeline ─────────────────────────────────────────────────
    if args.zip is None:
        zips = list(SCRIPT_DIR.glob("*.zip"))
        if zips:
            args.zip = str(zips[0])
            print(f"[→] Auto-detected ZIP: {args.zip}")
        else:
            print("ERROR: Provide --zip path/to/dataset.zip")
            sys.exit(1)

    class_dirs = unzip_dataset(args.zip)
    tr_ds, val_ds, te_ds, class_names = build_datasets(class_dirs)

    with open(NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"    Class names saved → {NAMES_PATH}")

    tr_dl  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=True)
    te_dl  = DataLoader(te_ds,  batch_size=args.batch, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=True)

    _model = build_model(len(class_names))
    params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    print(f"    Trainable parameters : {params:,}\n")

    history, best_acc = train_model(_model, tr_dl, val_dl, args.epochs)

    _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    lbl, prd, prb, acc, bal_acc, auc = evaluate_model(_model, te_dl, class_names)
    save_plots(history, lbl, prd, class_names, acc, bal_acc, auc)

    print("\n" + "="*62)
    print("  TRAINING COMPLETE")
    print("="*62)
    print(f"  Model     : {MODEL_PATH}")
    print(f"  Classes   : {NAMES_PATH}")
    print(f"  Plot      : {PLOT_PATH}")
    print(f"  Classes   : {class_names}")
    print(f"  Acc       : {acc:.4f}")
    print(f"  BalAcc    : {bal_acc:.4f}")
    if auc: print(f"  ROC-AUC   : {auc:.4f}")
    print("="*62)

    launch_gradio(class_names, _model)