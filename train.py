
import os
import random
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

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

SEED = 42
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 120
NUM_CLASSES = 6

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent

# Works from Windows and WSL, provided the project is in the
# same folder and the dataset has the expected structure.
DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "Garbage classification"
    / "Garbage classification"
)

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "garbage_cnn.keras"


# ============================================================
# 2. CPU / GPU SETUP
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("\nGPU detected:")
    for gpu in gpus:
        print(" ", gpu.name)

    # Let TensorFlow allocate GPU memory as it needs it.
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print("Could not set GPU memory growth:", error)

    DEVICE = "/GPU:0"
    print("Training device: GPU")
else:
    DEVICE = "/CPU:0"
    print("\nNo GPU detected. Training on CPU.")

print("TensorFlow version:", tf.__version__)
print("Project directory:", PROJECT_DIR)
print("Dataset directory:", DATA_DIR)

if not DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"\nDataset folder not found:\n{DATA_DIR}\n"
        "Check DATA_DIR and your dataset folder structure."
    )


# ============================================================
# 3. LOAD IMAGE PATHS AND LABELS
# ============================================================

image_paths = []
labels = []

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

for class_id, class_name in enumerate(CLASS_NAMES):
    class_dir = DATA_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Missing class folder: {class_dir}"
        )

    files = sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in valid_extensions
    )

    print(f"{class_name:12s}: {len(files)} images")

    image_paths.extend(str(path) for path in files)
    labels.extend([class_id] * len(files))

image_paths = np.array(image_paths)
labels = np.array(labels, dtype=np.int32)

if len(image_paths) == 0:
    raise ValueError("No images found in the dataset.")

if len(np.unique(labels)) != NUM_CLASSES:
    raise ValueError("Expected images from all six classes.")

print("\nTotal images:", len(image_paths))


# ============================================================
# 4. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# 70% training, 15% validation, 15% testing.
train_paths, temp_paths, train_labels, temp_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )
)

val_paths, test_paths, val_labels, test_labels = (
    train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels,
    )
)

print("\nDataset split:")
print(f"Training:   {len(train_paths)}")
print(f"Validation: {len(val_paths)}")
print(f"Testing:    {len(test_paths)}")


# ============================================================
# 5. IMAGE LOADING PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(label, tf.int32)

    return image, label


def make_dataset(paths, targets, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, targets)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_ds = make_dataset(
    train_paths, train_labels, training=True
)

val_ds = make_dataset(
    val_paths, val_labels
)

test_ds = make_dataset(
    test_paths, test_labels
)


# ============================================================
# 6. CLASS WEIGHTS FOR IMBALANCED DATA
# ============================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels,
)

class_weights = {
    class_id: float(weight)
    for class_id, weight in enumerate(weights)
}

print("\nClass weights:")
for name, weight in zip(CLASS_NAMES, weights):
    print(f"{name:12s}: {weight:.3f}")


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="data_augmentation",
)


# ============================================================
# 8. CNN MODEL
# ============================================================

def convolution_block(x, filters, dropout_rate):
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    return x


inputs = tf.keras.Input(
    shape=(*IMAGE_SIZE, 3),
    name="input_image",
)

x = augmentation(inputs)

x = convolution_block(x, 32, 0.15)
x = convolution_block(x, 64, 0.20)
x = convolution_block(x, 128, 0.25)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.40)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="class_predictions",
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ============================================================
# 9. TRAINING CALLBACKS
# ============================================================

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),

    tf.keras.callbacks.CSVLogger(
        str(RESULTS_DIR / "training_history.csv")
    ),
]


# ============================================================
# 10. TRAIN
# ============================================================

print("\nStarting training...")
print("Maximum epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Device:", DEVICE)

with tf.device(DEVICE):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

print("\nTraining completed.")


# ============================================================
# 11. LOAD BEST CHECKPOINT AND EVALUATE TEST SET
# ============================================================

best_model = tf.keras.models.load_model(MODEL_PATH)

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1,
)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

probabilities = best_model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)

# Keep labels aligned with the test dataset.
true_labels = np.concatenate(
    [labels.numpy() for _, labels in test_ds],
    axis=0,
)

report = classification_report(
    true_labels,
    predicted_labels,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASS_NAMES,
    zero_division=0,
)

print("\nClassification report:\n")
print(report)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(report)

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=np.arange(NUM_CLASSES),
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)

fig, ax = plt.subplots(figsize=(9, 9))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
ax.set_title("Garbage CNN — Test Confusion Matrix")
fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=200,
)
plt.close(fig)


# ============================================================
# 12. SAVE TRAINING CURVES
# ============================================================

history_df = pd.DataFrame(history.history)
history_df.to_csv(
    RESULTS_DIR / "training_history_full.csv",
    index=False,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Training")
axes[0].plot(
    history.history["val_accuracy"],
    label="Validation",
)
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history["loss"], label="Training")
axes[1].plot(
    history.history["val_loss"],
    label="Validation",
)
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "training_curves.png",
    dpi=200,
)
plt.close(fig)

print("\nAll training and evaluation steps completed!")
print("Best model:", MODEL_PATH)
print("Results folder:", RESULTS_DIR)
