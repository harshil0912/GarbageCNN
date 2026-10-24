
"""
Generate GitHub-ready evaluation reports for Smart Waste Classification V1.

IMPORTANT:
- Does not retrain or modify the model.
- Does not overwrite existing V1 reports.
- Reconstructs the stratified 70/15/15 split with seed 42.
- Verify the split logic matches your original train.py before trusting metrics.
"""

from pathlib import Path
import json
import csv
import hashlib
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    ROOT / "data" / "raw" / "Garbage classification"
    / "Garbage classification"
)

MODEL_PATH = ROOT / "models" / "garbage_cnn.keras"

OUTPUT_DIR = ROOT / "results" / "v1_github_report"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

IMAGE_SIZE = (128, 128)
SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---------------------------------------------------------
# SAFETY CHECKS
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"V1 model not found: {MODEL_PATH}"
    )

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATA_DIR}"
    )

# Never overwrite a previous report run.
if OUTPUT_DIR.exists():
    run_number = 2
    while (ROOT / "results" /
           f"v1_github_report_run_{run_number}").exists():
        run_number += 1

    OUTPUT_DIR = (
        ROOT / "results" /
        f"v1_github_report_run_{run_number}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

print(f"Report output: {OUTPUT_DIR}")

# ---------------------------------------------------------
# COLLECT DATASET PATHS
# ---------------------------------------------------------

image_paths = []
labels = []

for class_id, class_name in enumerate(CLASSES):
    class_dir = DATA_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Missing class folder: {class_dir}"
        )

    files = sorted(
        p for p in class_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    print(f"{class_name}: {len(files)} images")

    image_paths.extend(str(p) for p in files)
    labels.extend([class_id] * len(files))

image_paths = np.array(image_paths)
labels = np.array(labels, dtype=np.int32)

if len(image_paths) == 0:
    raise RuntimeError("No images were found.")

# Dataset inventory report
inventory = pd.DataFrame({
    "class_name": CLASSES,
    "class_id": range(len(CLASSES)),
    "image_count": [
        int(np.sum(labels == i))
        for i in range(len(CLASSES))
    ],
})

inventory["percentage"] = (
    inventory["image_count"] / len(labels) * 100
)

inventory.to_csv(
    OUTPUT_DIR / "dataset_class_counts.csv",
    index=False,
)

# ---------------------------------------------------------
# RECREATE THE SPLIT
# ---------------------------------------------------------
# This must match the original train.py exactly.
# First: 70% train / 30% temporary.
# Second: split temporary data equally into validation/test.

all_indices = np.arange(len(image_paths))

train_idx, temp_idx = train_test_split(
    all_indices,
    test_size=0.30,
    random_state=SEED,
    stratify=labels,
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=labels[temp_idx],
)

# Save split manifest so the evaluated images can be audited.
split_rows = []

for split_name, indices in [
    ("train", train_idx),
    ("validation", val_idx),
    ("test", test_idx),
]:
    for idx in indices:
        split_rows.append({
            "split": split_name,
            "relative_path": str(
                Path(image_paths[idx]).relative_to(ROOT)
            ),
            "true_class": CLASSES[labels[idx]],
        })

pd.DataFrame(split_rows).to_csv(
    OUTPUT_DIR / "split_manifest.csv",
    index=False,
)

print(f"Train images: {len(train_idx)}")
print(f"Validation images: {len(val_idx)}")
print(f"Test images: {len(test_idx)}")

# ---------------------------------------------------------
# LOAD V1 MODEL
# ---------------------------------------------------------

print("Loading V1 model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("Model loaded.")

# ---------------------------------------------------------
# PREPROCESS TEST IMAGES
# ---------------------------------------------------------

def load_image(path):
    """Load an image using the V1 resize/normalization pipeline."""
    raw = tf.io.read_file(path)

    image = tf.io.decode_image(
        raw,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image.numpy()


X_test = np.stack([
    load_image(image_paths[i])
    for i in test_idx
])

y_true = labels[test_idx]

print(f"Test tensor shape: {X_test.shape}")
print("Predicting test images...")

probabilities = model.predict(
    X_test,
    batch_size=32,
    verbose=1,
)

y_pred = np.argmax(probabilities, axis=1)

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

accuracy = float(accuracy_score(y_true, y_pred))

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        zero_division=0,
    )
)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    output_dict=True,
    zero_division=0,
)

report_text = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    digits=4,
    zero_division=0,
)

(OUTPUT_DIR / "classification_report.txt").write_text(
    report_text,
    encoding="utf-8",
)

pd.DataFrame(report_dict).transpose().to_csv(
    OUTPUT_DIR / "classification_report.csv",
    index_label="class_or_average",
)

# Per-class metrics
per_class = pd.DataFrame({
    "class_name": CLASSES,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "support": support,
})

per_class.to_csv(
    OUTPUT_DIR / "per_class_metrics.csv",
    index=False,
)

# Overall metrics
overall_metrics = {
    "model": "Smart Waste Classification V1",
    "evaluation_timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "test_accuracy": accuracy,
    "test_correct": int(np.sum(y_true == y_pred)),
    "test_incorrect": int(np.sum(y_true != y_pred)),
    "test_samples": int(len(y_true)),
    "macro_precision": float(
        report_dict["macro avg"]["precision"]
    ),
    "macro_recall": float(
        report_dict["macro avg"]["recall"]
    ),
    "macro_f1": float(
        report_dict["macro avg"]["f1-score"]
    ),
    "weighted_precision": float(
        report_dict["weighted avg"]["precision"]
    ),
    "weighted_recall": float(
        report_dict["weighted avg"]["recall"]
    ),
    "weighted_f1": float(
        report_dict["weighted avg"]["f1-score"]
    ),
    "classes": CLASSES,
    "image_size": list(IMAGE_SIZE),
    "split_seed": SEED,
    "python_version": platform.python_version(),
    "tensorflow_version": tf.__version__,
    "note": (
        "Metrics are valid for the reconstructed split only "
        "if it exactly matches the original V1 split."
    ),
}

with open(
    OUTPUT_DIR / "overall_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(overall_metrics, f, indent=4)

pd.DataFrame([overall_metrics]).to_csv(
    OUTPUT_DIR / "overall_metrics.csv",
    index=False,
)

# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASSES))),
)

pd.DataFrame(
    cm,
    index=CLASSES,
    columns=CLASSES,
).rename_axis("actual_class").to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)

fig, ax = plt.subplots(figsize=(10, 8))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES,
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=35,
    colorbar=False,
)

ax.set_title("V1 CNN — Test Confusion Matrix")
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# ---------------------------------------------------------
# MISCLASSIFIED IMAGES
# ---------------------------------------------------------

errors = []

for position, dataset_idx in enumerate(test_idx):
    true_id = int(y_true[position])
    pred_id = int(y_pred[position])

    if true_id == pred_id:
        continue

    errors.append({
        "relative_path": str(
            Path(image_paths[dataset_idx]).relative_to(ROOT)
        ),
        "true_class": CLASSES[true_id],
        "predicted_class": CLASSES[pred_id],
        "predicted_score": float(
            probabilities[position, pred_id]
        ),
        "true_class_score": float(
            probabilities[position, true_id]
        ),
    })

pd.DataFrame(
    errors,
    columns=[
        "relative_path",
        "true_class",
        "predicted_class",
        "predicted_score",
        "true_class_score",
    ],
).to_csv(
    OUTPUT_DIR / "misclassified_images.csv",
    index=False,
)

# ---------------------------------------------------------
# PREDICTIONS FOR AUDITING
# ---------------------------------------------------------

pred_rows = []

for position, dataset_idx in enumerate(test_idx):
    pred_id = int(y_pred[position])

    row = {
        "relative_path": str(
            Path(image_paths[dataset_idx]).relative_to(ROOT)
        ),
        "true_class": CLASSES[int(y_true[position])],
        "predicted_class": CLASSES[pred_id],
        "correct": bool(y_true[position] == y_pred[position]),
    }

    for class_id, class_name in enumerate(CLASSES):
        row[f"score_{class_name}"] = float(
            probabilities[position, class_id]
        )

    pred_rows.append(row)

pd.DataFrame(pred_rows).to_csv(
    OUTPUT_DIR / "all_test_predictions.csv",
    index=False,
)

# ---------------------------------------------------------
# REPORT README
# ---------------------------------------------------------

readme = f"""# Smart Waste Classification — V1 Evaluation

## Summary

- Test samples: {len(y_true)}
- Test accuracy on this run: {accuracy:.4%}
- Correct predictions: {int(np.sum(y_true == y_pred))}
- Incorrect predictions: {int(np.sum(y_true != y_pred))}
- Classes: {", ".join(CLASSES)}

## Files

- `overall_metrics.json` — overall evaluation metrics
- `overall_metrics.csv` — overall metrics in tabular form
- `classification_report.txt` — readable classification report
- `classification_report.csv` — full metrics table
- `per_class_metrics.csv` — precision, recall, F1 and support
- `confusion_matrix.png` — confusion matrix visualization
- `confusion_matrix.csv` — raw confusion matrix counts
- `dataset_class_counts.csv` — class counts
- `split_manifest.csv` — images assigned to each split
- `all_test_predictions.csv` — predictions and scores for every test image
- `misclassified_images.csv` — incorrect predictions for error analysis

## Important reproducibility warning

This script reconstructs a stratified split using seed {SEED}.
Before treating these numbers as the original V1 results, verify
that this split logic, file ordering, dataset contents, image
decoding, and preprocessing match the original training script.

If they do not match, these are metrics for a reconstructed split,
not proof of the original V1 test result.

Scores from the softmax output are not necessarily calibrated
probabilities.

## Model

Saved model: `models/garbage_cnn.keras`
Input size: {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}
TensorFlow: {tf.__version__}
"""

(OUTPUT_DIR / "README.md").write_text(
    readme,
    encoding="utf-8",
)

print("\nREPORT GENERATION COMPLETE")
print(f"Accuracy: {accuracy:.4%}")
print(f"Correct: {np.sum(y_true == y_pred)} / {len(y_true)}")
print(f"Reports saved in: {OUTPUT_DIR}")
print("\nPer-class results:")
print(per_class.to_string(index=False))
