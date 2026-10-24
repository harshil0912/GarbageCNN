
"""
Smart Waste Classification
MobileNetV2 Transfer Learning - Streamlit App

Model:
models/transfer_learning/mobilenetv2_best.keras

Expected model input:
224 x 224 RGB images, float32 values in the 0-255 range.
The saved model handles preprocessing internally.
"""

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "transfer_learning"
    / "mobilenetv2_best.keras"
)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

IMAGE_SIZE = (224, 224)


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #9CA3AF;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .result-card {
            padding: 1.4rem;
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-top: 0.7rem;
            margin-bottom: 1rem;
        }

        .result-label {
            font-size: 0.9rem;
            color: #9CA3AF;
            margin-bottom: 0.35rem;
        }

        .result-class {
            font-size: 1.8rem;
            font-weight: 800;
            text-transform: capitalize;
        }

        .result-confidence {
            font-size: 1.3rem;
            font-weight: 700;
        }

        .section-heading {
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.7rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            padding: 1rem;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path_string):
    """Load and cache the trained Keras model."""
    return tf.keras.models.load_model(model_path_string)


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict_image(image, model):
    """
    Prepare the uploaded image and predict its waste category.

    The model includes its own Rescaling layer.
    Therefore, do NOT divide pixel values by 255 here.
    """

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)

    # Add batch dimension:
    # (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    return predicted_class, confidence, predictions


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:
    st.title("♻️ Smart Waste AI")

    st.markdown("---")

    st.subheader("About the model")

    st.write(
        "This application uses MobileNetV2 transfer learning "
        "to classify waste images into six categories."
    )

    st.markdown(
        """
        **Supported categories**
        - 📦 Cardboard
        - 🪟 Glass
        - 🔩 Metal
        - 📄 Paper
        - 🧴 Plastic
        - 🗑️ Trash
        """
    )

    st.markdown("---")

    st.caption("Model: MobileNetV2")
    st.caption("Input size: 224 × 224")
    st.caption("Classes: 6")
    st.caption("Test accuracy: 87.37%")

    st.info(
        "The reported test accuracy comes from the saved "
        "model's evaluation on the project's test split. "
        "Real-world performance may differ."
    )


# ============================================================
# 7. MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">♻️ Smart Waste Classification</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Upload an image and let the trained AI model predict
        its waste category.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# ============================================================
# 8. CHECK MODEL FILE
# ============================================================

if not MODEL_PATH.exists():
    st.error("The trained MobileNetV2 model could not be found.")

    st.write("Expected model location:")

    st.code(str(MODEL_PATH), language="text")

    st.warning(
        "Make sure you have trained the model and that "
        "mobilenetv2_best.keras exists inside "
        "models/transfer_learning/."
    )

    st.stop()


# ============================================================
# 9. LOAD MODEL SAFELY
# ============================================================

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error("Failed to load the trained model.")
    st.exception(error)
    st.stop()


# ============================================================
# 10. IMAGE UPLOADER
# ============================================================

st.markdown(
    '<div class="section-heading">Upload a waste image</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose an image to classify",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Upload a clear image of cardboard, glass, metal, paper, plastic, or trash.",
)


# ============================================================
# 11. IMAGE PREVIEW AND PREDICTION
# ============================================================

if uploaded_file is None:
    st.info(
        "Upload an image to get started. "
        "For better predictions, use a clear image with the waste item visible."
    )

    st.markdown("### How to use")
    st.markdown(
        """
        1. Upload an image using the uploader above.
        2. Preview the uploaded image.
        3. Click **Classify Image**.
        4. View the predicted category and confidence scores.
        """
    )

else:
    try:
        image = Image.open(uploaded_file)
        image.load()
        image = image.convert("RGB")

        left_column, right_column = st.columns(
            [1, 1],
            gap="large",
        )

        with left_column:
            st.subheader("Uploaded image")

            st.image(
                image,
                caption="Image submitted for classification",
                use_container_width=True,
            )

            st.caption(
                f"Original image dimensions: "
                f"{image.width} × {image.height}"
            )

        with right_column:
            st.subheader("AI prediction")

            st.write(
                "Click the button below to classify the uploaded image."
            )

            classify_button = st.button(
                "🔍 Classify Image",
                type="primary",
                use_container_width=True,
            )

            if classify_button:
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, predictions = (
                        predict_image(image, model)
                    )

                st.session_state["prediction_result"] = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "predictions": predictions.tolist(),
                }

        # Show the latest prediction if one exists.
        result = st.session_state.get("prediction_result")

        if result is not None:
            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            predictions = np.asarray(
                result["predictions"],
                dtype=np.float32,
            )

            st.markdown("---")

            st.subheader("Classification result")

            result_column, confidence_column = st.columns(2)

            with result_column:
                st.metric(
                    label="Predicted category",
                    value=predicted_class.title(),
                )

            with confidence_column:
                st.metric(
                    label="Model confidence",
                    value=f"{confidence:.2%}",
                )

            st.progress(
                float(np.clip(confidence, 0.0, 1.0)),
                text=f"Confidence: {confidence:.2%}",
            )

            st.markdown("---")

            st.subheader("Class probability scores")

            st.caption(
                "Scores show the model's relative predicted "
                "probabilities for the six trained categories. "
                "They are not guaranteed to be calibrated probabilities."
            )

            sorted_indices = np.argsort(predictions)[::-1]

            for index in sorted_indices:
                class_name = CLASS_NAMES[int(index)]
                score = float(predictions[int(index)])

                label_column, score_column = st.columns(
                    [3, 1]
                )

                with label_column:
                    if class_name == predicted_class:
                        st.markdown(
                            f"**✅ {class_name.title()}**"
                        )
                    else:
                        st.write(class_name.title())

                with score_column:
                    st.markdown(
                        f"<div style='text-align:right;font-weight:600'>"
                        f"{score:.2%}</div>",
                        unsafe_allow_html=True,
                    )

                st.progress(
                    float(np.clip(score, 0.0, 1.0))
                )

            st.markdown("---")

            st.caption(
                "This is an image classification model, not a "
                "waste-sorting authority. Check local disposal "
                "guidelines before making disposal decisions."
            )

    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "The uploaded file could not be read as a valid image. "
            "Please try another JPG, PNG, or WebP image."
        )

    except Exception as error:
        st.error("Something went wrong while processing the image.")
        st.exception(error)


# ============================================================
# 12. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#9CA3AF;font-size:0.85rem;">
        Smart Waste Classification · MobileNetV2 Transfer Learning
    </div>
    """,
    unsafe_allow_html=True,
)
