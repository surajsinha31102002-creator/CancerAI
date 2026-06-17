import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif

# -----------------------------
# Load dataset
# -----------------------------
CSV_PATH = "miRNA_lung.csv.zip"

df = pd.read_csv(CSV_PATH, compression="zip")

# -----------------------------
# Merge rare classes
# -----------------------------
rare_classes = [
    "Giant Cell Carcinoma of the Lung",
    "Mucoepidermoid Carcinoma of the Lung"
]

df["lineage_3_merged"] = df["lineage_3"].replace(
    rare_classes,
    "Other Lung Cancer"
)

# -----------------------------
# Label encoders
# -----------------------------
le_bin = LabelEncoder()
le_multi = LabelEncoder()

df["label_binary"] = le_bin.fit_transform(
    df["lineage_2"]
)

df["label_multi"] = le_multi.fit_transform(
    df["lineage_3_merged"]
)

# -----------------------------
# Feature columns
# -----------------------------
META_COLS = [
    "depmap_id",
    "cell_line_display_name",
    "lineage_1",
    "lineage_2",
    "lineage_3",
    "lineage_4",
    "lineage_5",
    "lineage_6",
    "lineage_3_merged",
    "label_binary",
    "label_multi"
]

MIRNA_COLS = [
    c for c in df.columns
    if c not in META_COLS
]

X = df[MIRNA_COLS].values.astype(np.float32)

# -----------------------------
# Feature selection
# -----------------------------
y_multi = df["label_multi"]

mi_scores = mutual_info_classif(
    X,
    y_multi,
    random_state=42
)

top_idx = np.argsort(mi_scores)[::-1][:150]

X_selected = X[:, top_idx]

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
scaler.fit(X_selected)

# -----------------------------
# Save preprocessing artifacts
# -----------------------------
joblib.dump(scaler, "scaler.pkl")
joblib.dump(top_idx, "selected_features.pkl")

joblib.dump(
    le_bin,
    "binary_label_encoder.pkl"
)

joblib.dump(
    le_multi,
    "multi_label_encoder.pkl"
)

print("✓ Saved preprocessing files")