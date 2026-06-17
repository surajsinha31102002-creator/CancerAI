from pdf_report import create_report
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import models
import numpy as np
import pandas as pd
import joblib
import json
import io
import os
import math
import re

# ══════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="Lung + Colon Cancer AI API",
    description="Hybrid AI using Histopathology + miRNA",
    version="2.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
CLASS_META = {
    "colon_aca": {"label": "Colon Adenocarcinoma",         "organ": "Colon", "status": "CANCER"},
    "colon_n":   {"label": "Normal Colon Tissue",          "organ": "Colon", "status": "NORMAL"},
    "lung_aca":  {"label": "Lung Adenocarcinoma",          "organ": "Lung",  "status": "CANCER"},
    "lung_n":    {"label": "Normal Lung Tissue",           "organ": "Lung",  "status": "NORMAL"},
    "lung_scc":  {"label": "Lung Squamous Cell Carcinoma", "organ": "Lung",  "status": "CANCER"},
}

def safe_float(x, default=0.0):
    try:
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else round(v, 4)
    except Exception:
        return default

def convert_european_decimal(value):
    """Convert European decimal format (1.234,56) to standard (1234.56)"""
    if isinstance(value, str):
        # Check if it has European format (dot as thousand separator, comma as decimal)
        if ',' in value and value.count(',') == 1 and '.' in value:
            # Remove thousand separators (dots) and replace comma with dot
            value = value.replace('.', '').replace(',', '.')
        elif ',' in value and '.' not in value:
            # Simple comma as decimal
            value = value.replace(',', '.')
    return value

def read_csv_robust(csv_bytes: bytes) -> pd.DataFrame:
    """Handle Excel files, CSVs with headers, and CSVs without headers (raw data)."""
    # Excel detection
    if csv_bytes[:4] == b'PK\x03\x04' or csv_bytes[:2] == b'\xd0\xcf':
        try:
            df = pd.read_excel(io.BytesIO(csv_bytes), header=None)
            # If first row looks like column names (strings), use header=0
            if df.iloc[0].dtype == object:
                df = pd.read_excel(io.BytesIO(csv_bytes))
            return df
        except Exception as e:
            raise ValueError(f"File looks like Excel but failed to read: {e}")

    # Try CSV encodings
    for enc in ["utf-8-sig", "utf-8", "utf-16", "latin-1", "cp1252"]:
        try:
            # First try with header (normal CSV)
            df = pd.read_csv(io.BytesIO(csv_bytes), encoding=enc)
            
            # If 0 rows — the data row was mistaken for header (no-header CSV)
            # Detect: column names are numeric strings like '14.735'
            if len(df) == 0:
                try:
                    float(str(df.columns[0]))  # column name is a number
                    # Re-read without header
                    df = pd.read_csv(io.BytesIO(csv_bytes), encoding=enc, header=None)
                except (ValueError, IndexError):
                    pass
            
            return df
        except Exception:
            continue

    raise ValueError(
        "Could not parse the file. Upload a CSV or Excel file with miRNA expression columns."
    )

def clean_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare CSV data for prediction.
    Handles string numbers, European decimals, and ensures proper numeric conversion.
    """
    # First, attempt to convert all columns to numeric
    # For each column, try to convert string representations to numbers
    for col in df.columns:
        if df[col].dtype == 'object':
            # Apply European decimal conversion to string values
            df[col] = df[col].apply(convert_european_decimal)
            # Convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Keep only numeric columns (drop any remaining non-numeric)
    df = df.select_dtypes(include=[np.number])
    
    # Drop columns that are completely NaN
    df = df.dropna(axis=1, how='all')
    
    # Drop any explicitly named label columns if present
    label_columns = [c for c in ["label", "target", "diagnosis", "class", "Label", "Target", "Diagnosis", "Class"]
                     if c in df.columns]
    if label_columns:
        df = df.drop(columns=label_columns, errors="ignore")
    
    # If DataFrame is empty, raise error
    if df.empty:
        raise ValueError("No numeric columns found in the CSV file. Please ensure the file contains numeric expression values.")
    
    # Return only the first row (first patient's data)
    # If there are multiple rows, take the first one
    return df.iloc[[0]]

def validate_and_prepare_features(df: pd.DataFrame, selected_features: np.ndarray) -> np.ndarray:
    """
    Validate feature count and prepare the feature matrix for prediction.
    """
    X = df.values.astype(np.float32)
    min_cols_needed = int(np.max(selected_features)) + 1
    
    if X.shape[1] < min_cols_needed:
        raise ValueError(
            f"CSV has {X.shape[1]} numeric columns, need at least {min_cols_needed}. "
            f"Please ensure your miRNA expression data has {min_cols_needed} features."
        )
    
    # Select only the required features
    X_selected = X[:, selected_features]
    
    # Check for any remaining NaN or Inf values
    if np.any(np.isnan(X_selected)) or np.any(np.isinf(X_selected)):
        print(f"Warning: Found {np.sum(np.isnan(X_selected))} NaN values, replacing with 0")
        X_selected = np.nan_to_num(X_selected, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_selected

# ══════════════════════════════════════════════════════════════
# IMAGE MODEL  — EfficientNet-B3
# ══════════════════════════════════════════════════════════════
print("Loading image model...")

with open(os.path.join(BASE_DIR, "class_names.json")) as f:
    image_classes = json.load(f)

image_model = models.efficientnet_b3(weights=None)
in_features = image_model.classifier[1].in_features
image_model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, len(image_classes)),
)
image_model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "best_model.pth"), map_location=device)
)
image_model.to(device).eval()
print(f"✓ Image model loaded  ({len(image_classes)} classes)")

# TTA transforms
TTA_AUGS = [
    lambda x: x,
    lambda x: TF.hflip(x),
    lambda x: TF.vflip(x),
    lambda x: TF.rotate(x, 90),
    lambda x: TF.rotate(x, 180),
    lambda x: TF.rotate(x, 270),
    lambda x: TF.hflip(TF.vflip(x)),
    lambda x: TF.adjust_brightness(x, 1.1),
]

IMG_TF = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_image_tta(pil_image) -> torch.Tensor:
    """Run TTA and return averaged softmax probabilities."""
    prob_sum = None
    with torch.no_grad():
        for aug in TTA_AUGS:
            tensor = IMG_TF(aug(pil_image)).unsqueeze(0).to(device)
            probs  = torch.softmax(image_model(tensor), dim=1)
            prob_sum = probs if prob_sum is None else prob_sum + probs
    return (prob_sum / len(TTA_AUGS)).squeeze(0)

# ══════════════════════════════════════════════════════════════
# TABULAR PREPROCESSING
# ══════════════════════════════════════════════════════════════
print("Loading preprocessing files...")
scaler            = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
selected_features = joblib.load(os.path.join(BASE_DIR, "selected_features.pkl"))
binary_encoder    = joblib.load(os.path.join(BASE_DIR, "binary_label_encoder.pkl"))
multi_encoder     = joblib.load(os.path.join(BASE_DIR, "multi_label_encoder.pkl"))

# Ensure selected_features is always a numpy array
selected_features = np.array(selected_features)
print(f"✓ Preprocessing loaded (using {len(selected_features)} features)")

# ══════════════════════════════════════════════════════════════
# TABULAR MODELS
# ══════════════════════════════════════════════════════════════
print("Loading tabular models...")

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class BinaryMiRNANet(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.mid  = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.res3 = ResidualBlock(128)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.res3(self.mid(self.res2(self.res1(self.stem(x))))))


class MultiMiRNANet(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.res1 = ResidualBlock(256)
        self.mid  = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.res2 = ResidualBlock(128)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.res2(self.mid(self.res1(self.stem(x)))))


binary_model = BinaryMiRNANet(input_dim=150, n_classes=2)
binary_model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "binary_model_v3.pth"), map_location=device))
binary_model.to(device).eval()

ensemble_models = []
n_multi = len(multi_encoder.classes_)
for i in range(1, 6):
    m = MultiMiRNANet(input_dim=150, n_classes=n_multi)
    m.load_state_dict(
        torch.load(os.path.join(BASE_DIR, f"multiclass_ensemble_fold{i}_v3.pth"),
                   map_location=device))
    m.to(device).eval()
    ensemble_models.append(m)

print(f"✓ Tabular models loaded  (1 binary + {len(ensemble_models)}-fold ensemble)")


def predict_tabular_ensemble(X_scaled: np.ndarray):
    """Run binary + ensemble multi-class prediction on scaled feature array."""
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        bin_probs = torch.softmax(binary_model(X_t), dim=1)
        bin_pred  = int(torch.argmax(bin_probs, dim=1).item())

        mul_avg  = torch.mean(
            torch.stack([torch.softmax(m(X_t), dim=1) for m in ensemble_models]), dim=0)
        mul_pred = int(torch.argmax(mul_avg, dim=1).item())
        mul_conf = safe_float(mul_avg.max().item() * 100)

    return (
        binary_encoder.inverse_transform([bin_pred])[0],
        multi_encoder.inverse_transform([mul_pred])[0],
        mul_conf,
    )


# ══════════════════════════════════════════════════════════════
# FAVICON / DEVTOOLS — suppress noisy 404s in logs
# ══════════════════════════════════════════════════════════════

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def devtools():
    return Response(status_code=204)


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


# ── DEBUG ENDPOINT (helps diagnose CSV issues) ────────────────
@app.post("/debug-csv")
async def debug_csv(csv_file: UploadFile = File(...)):
    """Debug endpoint to inspect CSV structure and data types."""
    try:
        contents = await csv_file.read()
        df = read_csv_robust(contents)
        
        # Show original structure
        debug_info = {
            "original_shape": df.shape,
            "original_columns": list(df.columns[:10]) if len(df.columns) > 10 else list(df.columns),
            "original_dtypes": df.dtypes.astype(str).to_dict(),
            "first_row_original": df.iloc[0].head(10).to_dict() if len(df.columns) > 0 else {},
        }
        
        # Show cleaned structure
        df_cleaned = clean_csv(df)
        debug_info["cleaned_shape"] = df_cleaned.shape
        debug_info["cleaned_dtypes"] = df_cleaned.dtypes.astype(str).to_dict()
        debug_info["cleaned_first_row"] = df_cleaned.iloc[0].head(10).to_dict() if len(df_cleaned.columns) > 0 else {}
        
        # Check feature requirements
        min_cols_needed = int(np.max(selected_features)) + 1
        debug_info["min_columns_required"] = min_cols_needed
        debug_info["has_enough_columns"] = df_cleaned.shape[1] >= min_cols_needed
        
        return debug_info
    except Exception as e:
        return {"error": str(e)}


# ── /predict-image ─────────────────────────────────────────────
@app.post("/predict-image")
async def predict_image_endpoint(image_file: UploadFile = File(...)):
    try:
        image    = Image.open(io.BytesIO(await image_file.read())).convert("RGB")
        avg      = predict_image_tta(image)
        pred_idx = int(torch.argmax(avg).item())
        pred_cls = image_classes[pred_idx]
        conf     = safe_float(avg[pred_idx].item() * 100)
        meta     = CLASS_META.get(pred_cls, {"label": pred_cls, "organ": "?", "status": "UNKNOWN"})
        probs    = {image_classes[i]: safe_float(avg[i].item())
                    for i in range(len(image_classes))}

        return {
            "predicted_class": pred_cls,
            "label":           meta["label"],
            "status":          meta["status"],
            "organ":           meta["organ"],
            "confidence":      conf,
            "probabilities":   probs,
        }
    except Exception as e:
        return {"error": str(e)}


# ── /predict-tabular ────────────────────────────────────────────
@app.post("/predict-tabular")
async def predict_tabular(csv_file: UploadFile = File(...)):
    try:
        # Read and clean the CSV
        contents = await csv_file.read()
        df = read_csv_robust(contents)
        df = clean_csv(df)
        
        # Validate and prepare features
        X = validate_and_prepare_features(df, selected_features)
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Run prediction
        binary_label, multi_label, conf = predict_tabular_ensemble(X_scaled)

        return {
            "binary_prediction":  binary_label,
            "subtype_prediction": multi_label,
            "confidence":         conf,
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


# ── /predict-combined ───────────────────────────────────────────
@app.post("/predict-combined")
async def predict_combined(
    image_file: UploadFile = File(...),
    csv_file: UploadFile = File(...),
):
    try:
        # ── Image Prediction ──────────────────────────────────
        image = Image.open(io.BytesIO(await image_file.read())).convert("RGB")
        avg = predict_image_tta(image)
        img_pred_idx = int(torch.argmax(avg).item())
        img_cls = image_classes[img_pred_idx]
        img_conf = safe_float(avg[img_pred_idx].item() * 100)
        img_meta = CLASS_META.get(img_cls, {"label": img_cls, "organ": "?", "status": "UNKNOWN"})
        img_probs = {image_classes[i]: safe_float(avg[i].item())
                     for i in range(len(image_classes))}

        # ── miRNA Prediction ──────────────────────────────────
        mol_subtype = "N/A"
        mol_conf = 0.0
        mol_binary = "N/A"
        mol_error = None
        
        try:
            df = read_csv_robust(await csv_file.read())
            df = clean_csv(df)
            X = df.values.astype(np.float32)
            min_cols_needed = int(np.max(selected_features)) + 1
            
            if X.shape[1] >= min_cols_needed:
                X = scaler.transform(X[:, selected_features])
                mol_binary, mol_subtype, mol_conf = predict_tabular_ensemble(X)
            else:
                mol_error = f"CSV has {X.shape[1]} columns, need {min_cols_needed}"
        except Exception as e:
            mol_error = str(e)
            print(f"miRNA prediction error: {mol_error}")

        # ── Detect Conflict ──────────────────────────────────
        conflict = False
        conflict_message = None
        
        if mol_subtype != "N/A" and mol_subtype != "Unknown":
            mol_is_cancer = any(word in mol_subtype.lower() for word in ['cancer', 'carcinoma', 'tumor'])
            img_is_cancer = img_meta["status"] == "CANCER"
            
            if mol_is_cancer != img_is_cancer:
                conflict = True
                img_status = "CANCER" if img_is_cancer else "NORMAL"
                mol_status = "CANCER" if mol_is_cancer else "NORMAL"
                conflict_message = (
                    f"⚠️ IMAGE and miRNA results differ! "
                    f"Image: {img_status} ({img_conf:.1f}%), "
                    f"miRNA: {mol_status} ({mol_conf:.1f}%). "
                    f"Manual review recommended."
                )

        # ── Combined Confidence ──────────────────────────────
        combined_conf = safe_float(
            (img_conf + mol_conf) / 2 if mol_subtype != "N/A" and mol_conf > 0 else img_conf
        )

        # ── Response with BOTH results ──────────────────────
        return {
            # Overall result (prioritizes image)
            "predicted_class": img_cls,
            "label": img_meta["label"],
            "status": img_meta["status"],
            "organ": img_meta["organ"],
            "confidence": combined_conf,
            
            # Individual results
            "image": {
                "prediction": img_cls,
                "label": img_meta["label"],
                "status": img_meta["status"],
                "confidence": img_conf,
                "probabilities": img_probs,
                "organ": img_meta["organ"],
            },
            "mirna": {
                "binary": mol_binary,
                "subtype": mol_subtype,
                "confidence": mol_conf,
                "error": mol_error,
            },
            
            # Metadata
            "conflict": conflict,
            "conflict_message": conflict_message,
            "mode": "Combined (Image + miRNA)",
        }
        
    except Exception as e:
        import traceback
        print(f"Combined prediction error: {traceback.format_exc()}")
        return {"error": str(e)}


# ── /generate-report ────────────────────────────────────────────
# ── /generate-report ────────────────────────────────────────────
@app.post("/generate-report")
async def generate_report(
    label:         str = Form(""),
    confidence:    str = Form("0"),
    status:        str = Form("UNKNOWN"),
    organ:         str = Form(""),
    class_id:      str = Form(""),
    model_used:    str = Form(""),
    probabilities: str = Form(""),
):
    try:
        from pdf_report import create_report
        import os
        import time
        from datetime import datetime
        
        # Parse probabilities
        probs = json.loads(probabilities) if probabilities else None
        conf_f = safe_float(confidence)
        
        # Generate a unique filename to avoid caching issues
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f"CancerAI_Report_{timestamp}.pdf"
        output_path = os.path.join(temp_dir, filename)
        
        # Generate the report
        out_path = create_report(
            label=label,
            confidence=conf_f,
            status=status,
            organ=organ,
            class_id=class_id,
            model_used=model_used or "EfficientNet-B3 + miRNA Ensemble",
            probabilities=probs,
            patient_id="N/A",
            output_path=output_path
        )
        
        # Small delay to ensure file is fully written
        time.sleep(0.1)
        
        # Verify file exists and has content
        if not os.path.exists(out_path):
            return {"error": "PDF file was not created"}
        
        file_size = os.path.getsize(out_path)
        if file_size == 0:
            return {"error": "PDF file is empty (0 bytes)"}
        
        print(f"✅ PDF generated: {out_path} ({file_size} bytes)")
        
        # Return the file with proper headers
        return FileResponse(
            path=out_path,
            media_type="application/pdf",
            filename=os.path.basename(out_path),
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(out_path)}",
                "Content-Type": "application/pdf",
                "Content-Length": str(file_size),
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )
        
    except Exception as e:
        import traceback
        print(f"❌ PDF generation error: {e}")
        print(traceback.format_exc())
        return {"error": f"PDF generation failed: {str(e)}"}


# ── /download-report (legacy) ───────────────────────────────────
@app.get("/download-report")
def download_report():
    path = os.path.join(BASE_DIR, "report.pdf")
    if not os.path.exists(path):
        return {"error": "No report generated yet. Run an analysis first."}
    return FileResponse(
        path=path,
        media_type="application/pdf",
        filename="CancerAI_Report.pdf",
    )


# ── Health check endpoint ──────────────────────────────────────
@app.get("/health")
async def health_check():
    """Check if the API is running and models are loaded."""
    return {
        "status": "healthy",
        "device": str(device),
        "image_model_loaded": True,
        "tabular_models_loaded": True,
        "num_features": len(selected_features),
    }