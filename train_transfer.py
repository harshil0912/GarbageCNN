
"""
train_share.py
MobileNetV2 transfer learning for Smart Waste Classification.

Keeps the original train.py (V1) untouched.
Dataset split: 70% train, 15% validation, 15% test.
"""

from pathlib import Path
import os
import random
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "Garbage classification"
    / "Garbage classification"
)

MODEL_DIR = PROJECT_DIR / "models" / "transfer_learning"
RESULTS_DIR = PROJECT_DIR / "results" / "transfer_learning"

CLASS_NAMES = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

HEAD_EPOCHS = 25
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LAST_N_LAYERS = 30

HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("Dataset:", DATA_DIR)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATA_DIR}\n"
        "Check the DATA_DIR path at the top of this script."
    )

# ============================================================
# 2. FIND IMAGES AND CREATE LABELS
# ============================================================

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_paths = []
image_labels = []

for class_id, class_name in enumerate(CLASS_NAMES):
    class_dir = DATA_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Missing class folder: {class_dir}")

    files = sorted(
        p for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not files:
        raise ValueError(f"No images found in {class_dir}")

    image_paths.extend(str(p) for p in files)
    image_labels.extend([class_id] * len(files))

image_paths = np.asarray(image_paths, dtype=str)
image_labels = np.asarray(image_labels, dtype=np.int32)

print("\nClass counts:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:12s}: {np.sum(image_labels == i)}")

print("Total images:", len(image_paths))

# ============================================================
# 3. STRATIFIED 70/15/15 SPLIT
# ============================================================

train_val_paths, test_paths, train_val_labels, test_labels = (
    train_test_split(
        image_paths,
        image_labels,
        test_size=0.15,
        random_state=SEED,
        stratify=image_labels,
    )
)

train_paths, val_paths, train_labels, val_labels = (
    train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=0.15 / 0.85,
        random_state=SEED,
        stratify=train_val_labels,
    )
)

print("\nSplit sizes:")
print(f"Training:   {len(train_paths)}")
print(f"Validation: {len(val_paths)}")
print(f"Testing:    {len(test_paths)}")

split_df = pd.DataFrame({
    "image_path": np.concatenate(
        [train_paths, val_paths, test_paths]
    ),
    "label_index": np.concatenate(
        [train_labels, val_labels, test_labels]
    ),
    "split": (
        ["train"] * len(train_paths)
        + ["validation"] * len(val_paths)
        + ["test"] * len(test_paths)
    ),
})

split_df["class_name"] = split_df["label_index"].map(
    lambda i: CLASS_NAMES[int(i)]
)
split_df.to_csv(RESULTS_DIR / "data_splits.csv", index=False)

# ============================================================
# 4. IMAGE PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, label):
    """Decode RGB image and resize it, keeping pixel range 0-255."""
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(
        image, IMAGE_SIZE, antialias=True
    )
    image = tf.cast(image, tf.float32)

    return image, tf.cast(label, tf.int32)


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

# Class weights are calculated from the training split only.
classes = np.unique(train_labels)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_labels,
)
class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("\nClass weights:")
for i, weight in class_weights.items():
    print(f"{CLASS_NAMES[i]:12s}: {weight:.4f}")

with open(RESULTS_DIR / "class_weights.json", "w") as f:
    json.dump(
        {
            CLASS_NAMES[i]: weight
            for i, weight in class_weights.items()
        },
        f,
        indent=2,
    )

# ============================================================
# 5. BUILD MOBILENETV2 TRANSFER-LEARNING MODEL
# ============================================================

print("\nLoading pretrained MobileNetV2 weights...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)

# Phase 1: freeze the pretrained feature extractor.
base_model.trainable = False

# Augmentation receives RGB pixel values in the 0-255 range.
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=SEED),
    tf.keras.layers.RandomRotation(0.08, seed=SEED + 1),
    tf.keras.layers.RandomZoom(0.10, seed=SEED + 2),
    tf.keras.layers.RandomContrast(0.10, seed=SEED + 3),
], name="data_augmentation")

inputs = tf.keras.Input(
    shape=(*IMAGE_SIZE, 3),
    name="rgb_image_0_to_255",
)

x = augmentation(inputs)

# MobileNetV2 expects values scaled approximately to [-1, 1].
x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1.0,
    name="mobilenetv2_preprocessing",
)(x)

# Keep BatchNormalization statistics in inference mode.
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)

outputs = tf.keras.layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
    name="class_predictions",
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="garbage_mobilenetv2_transfer",
)

model.summary()

# ============================================================
# 6. CALLBACKS
# ============================================================


def make_callbacks(checkpoint_path, csv_path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=7,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_path)),
    ]


# ============================================================
# 7. PHASE 1: TRAIN CLASSIFICATION HEAD
# ============================================================

phase1_path = MODEL_DIR / "mobilenetv2_phase1_best.keras"
phase1_log = RESULTS_DIR / "phase1_training_history.csv"

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=HEAD_LR
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n" + "=" * 60)
print("PHASE 1: TRAIN CLASSIFICATION HEAD")
print("=" * 60)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(phase1_path, phase1_log),
)

# Load the best checkpoint from phase 1.
phase1_model = tf.keras.models.load_model(phase1_path)

phase1_val_loss, phase1_val_acc = phase1_model.evaluate(
    val_ds, verbose=0
)

print(
    f"\nPhase 1 validation loss: {phase1_val_loss:.4f}"
    f"\nPhase 1 validation accuracy: {phase1_val_acc:.4f}"
)

# ============================================================
# 8. PHASE 2: FINE-TUNE UPPER BACKBONE LAYERS
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2: FINE-TUNE MOBILENETV2")
print("=" * 60)

# Start fine-tuning from the best phase-1 model.
model = phase1_model
backbone = model.get_layer("mobilenetv2_1.00_224")
backbone.trainable = True

# Freeze everything, then unfreeze only the final selected layers.
for layer in backbone.layers:
    layer.trainable = False

for layer in backbone.layers[-FINE_TUNE_LAST_N_LAYERS:]:
    # Keep BatchNormalization layers frozen for stability.
    if not isinstance(
        layer, tf.keras.layers.BatchNormalization
    ):
        layer.trainable = True

print(
    "Trainable backbone layers:",
    sum(layer.trainable for layer in backbone.layers),
)

# Recompile after changing trainable layers.
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=FINE_TUNE_LR
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

phase2_path = MODEL_DIR / "mobilenetv2_finetuned_best.keras"
phase2_log = RESULTS_DIR / "phase2_training_history.csv"

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(phase2_path, phase2_log),
)

phase2_model = tf.keras.models.load_model(phase2_path)

phase2_val_loss, phase2_val_acc = phase2_model.evaluate(
    val_ds, verbose=0
)

print(
    f"\nPhase 2 validation loss: {phase2_val_loss:.4f}"
    f"\nPhase 2 validation accuracy: {phase2_val_acc:.4f}"
)

# ============================================================
# 9. SELECT THE FINAL MODEL USING VALIDATION LOSS
# ============================================================

if phase2_val_loss < phase1_val_loss:
    final_model = phase2_model
    selected_phase = "phase2_finetuned"
    selected_val_loss = phase2_val_loss
    selected_val_acc = phase2_val_acc
else:
    final_model = phase1_model
    selected_phase = "phase1_frozen_backbone"
    selected_val_loss = phase1_val_loss
    selected_val_acc = phase1_val_acc

final_model_path = MODEL_DIR / "mobilenetv2_best.keras"
final_model.save(final_model_path)

selection_info = {
    "selected_phase": selected_phase,
    "validation_loss": float(selected_val_loss),
    "validation_accuracy": float(selected_val_acc),
    "phase1_validation_loss": float(phase1_val_loss),
    "phase1_validation_accuracy": float(phase1_val_acc),
    "phase2_validation_loss": float(phase2_val_loss),
    "phase2_validation_accuracy": float(phase2_val_acc),
    "image_size": list(IMAGE_SIZE),
    "class_names": CLASS_NAMES,
    "seed": SEED,
    "batch_size": BATCH_SIZE,
}

with open(RESULTS_DIR / "model_selection.json", "w") as f:
    json.dump(selection_info, f, indent=2)

print("\nSelected model:", selected_phase)
print("Saved model:", final_model_path)

# ============================================================
# 10. FINAL TEST EVALUATION
# ============================================================

# Test data is used only after model selection.
print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_loss, test_acc = final_model.evaluate(test_ds, verbose=1)

probabilities = final_model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)

prediction_accuracy = accuracy_score(
    test_labels, predicted_labels
)

report = classification_report(
    test_labels,
    predicted_labels,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print("\nClassification report:")
print(report)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

with open(RESULTS_DIR / "classification_report.txt", "w") as f:
    f.write("MobileNetV2 Transfer Learning\n")
    f.write(f"Selected phase: {selected_phase}\n")
    f.write(f"Test loss: {test_loss:.6f}\n")
    f.write(f"Test accuracy: {test_acc:.6f}\n\n")
    f.write(report)

report_dict = classification_report(
    test_labels,
    predicted_labels,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report_dict).transpose().to_csv(
    RESULTS_DIR / "classification_report.csv"
)

with open(RESULTS_DIR / "test_metrics.json", "w") as f:
    json.dump({
        "selected_phase": selected_phase,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "prediction_accuracy": float(prediction_accuracy),
        "test_image_count": int(len(test_labels)),
    }, f, indent=2)

# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    predicted_labels,
    labels=list(range(len(CLASS_NAMES))),
)

pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
).to_csv(RESULTS_DIR / "confusion_matrix.csv")

fig, ax = plt.subplots(figsize=(9, 8))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
).plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    colorbar=False,
)

ax.set_title("MobileNetV2 Transfer Learning - Test Confusion Matrix")
fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 12. SAVE TRAINING HISTORY AND PLOTS
# ============================================================

history1_df = pd.read_csv(phase1_log)
history2_df = pd.read_csv(phase2_log)

history1_df["phase"] = "phase1_head"
history2_df["phase"] = "phase2_finetune"

combined_history = pd.concat(
    [history1_df, history2_df],
    ignore_index=True,
)

combined_history.to_csv(
    RESULTS_DIR / "training_history.csv",
    index=False,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for df, label, style in [
    (history1_df, "Phase 1", "-"),
    (history2_df, "Phase 2", "--"),
]:
    epochs = range(1, len(df) + 1)

    axes[0].plot(
        epochs, df["accuracy"],
        linestyle=style, label=f"{label} train",
    )
    axes[0].plot(
        epochs, df["val_accuracy"],
        linestyle=style, label=f"{label} validation",
    )

    axes[1].plot(
        epochs, df["loss"],
        linestyle=style, label=f"{label} train",
    )
    axes[1].plot(
        epochs, df["val_loss"],
        linestyle=style, label=f"{label} validation",
    )

axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch within phase")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch within phase")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle("MobileNetV2 Transfer Learning Training Curves")
fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "training_curves.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 13. COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TRANSFER-LEARNING EXPERIMENT COMPLETE")
print("=" * 60)
print(f"Selected model: {final_model_path}")
print(f"Results folder: {RESULTS_DIR}")
print(f"Test accuracy: {test_acc:.2%}")
print("\nCompare this result with V1's test accuracy and per-class F1.")
print("Do not claim an improvement unless the measured results support it.")


