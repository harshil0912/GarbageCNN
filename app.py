
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "garbage_cnn.keras"

IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

CLASS_EMOJIS = {
    "cardboard": "📦",
    "glass": "🫙",
    "metal": "🥫",
    "paper": "📄",
    "plastic": "🧴",
    "trash": "🗑️",
}

st.set_page_config(
    page_title="GarbageVision | AI Waste Classifier",
    page_icon="♻️",
    layout="wide",
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1220;
        color: #f8fafc;
    }

    .hero {
        padding: 28px 10px 18px 10px;
        text-align: center;
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 5px;
        color: #4ade80;
    }

    .hero p {
        font-size: 1.05rem;
        color: #a7b4c8;
    }

    .panel {
        background: #111c2f;
        border: 1px solid #26364c;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 14px;
    }

    .prediction {
        font-size: 2.1rem;
        font-weight: 750;
        color: #4ade80;
        margin-bottom: 4px;
    }

    .muted {
        color: #a7b4c8;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}\n"
            "Make sure your V1 model exists in the models folder."
        )

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# --------------------------------------------------
# PREPROCESS IMAGE
# --------------------------------------------------

def preprocess_image(uploaded_image):
    image = ImageOps.exif_transpose(uploaded_image).convert("RGB")

    # Match V1 preprocessing: resize, then scale pixels to [0, 1].
    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image, image_array


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>♻️ GarbageVision</h1>
        <p>Drop an image. Let a CNN classify your waste.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="panel">
        <strong>How it works</strong><br>
        <span class="muted">
        Upload a waste image and our CNN, trained from scratch,
        will predict one of six categories.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Drop a waste image here, or browse your files",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="For best results, use a clear image of one main waste item.",
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file)
        display_image, model_input = preprocess_image(original_image)

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("### Your image")
            st.image(
                display_image,
                caption=uploaded_file.name,
                use_container_width=True,
            )

            st.caption(
                "The model resizes the image to 128 × 128 pixels."
            )

        with right:
            st.markdown("### CNN prediction")

            with st.spinner("Analysing image..."):
                model = load_model()
                predictions = model.predict(
                    model_input,
                    verbose=0,
                )[0]

            if len(predictions) != len(CLASS_NAMES):
                st.error(
                    "The model output does not match the six expected classes."
                )
                st.stop()

            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(predictions[predicted_index])

            st.markdown(
                f"""
                <div class="panel">
                    <div class="prediction">
                        {CLASS_EMOJIS[predicted_class]}
                        {predicted_class.title()}
                    </div>
                    <div class="muted">Top predicted category</div>
                    <h2>{confidence:.1%}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Scores for all categories")

            for index in np.argsort(predictions)[::-1]:
                class_name = CLASS_NAMES[int(index)]
                score = float(predictions[index])

                st.write(
                    f"{CLASS_EMOJIS[class_name]} "
                    f"**{class_name.title()}** — {score:.1%}"
                )

                st.progress(
                    float(np.clip(score, 0.0, 1.0))
                )

            st.caption(
                "These are model scores, not guaranteed probabilities "
                "of being correct."
            )

    except Exception as error:
        st.error(f"Could not classify this image: {error}")


else:
    st.info(
        "👆 Upload an image to see the prediction and class scores."
    )


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

st.divider()

st.markdown("### About this project")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Waste categories", "6")

with col2:
    st.metric("Test accuracy", "76.32%")

with col3:
    st.metric("Input size", "128 × 128")

st.markdown(
    """
    <div class="panel">
        <strong>Built with TensorFlow and Streamlit</strong>
        <p class="muted">
        This is an educational image-classification demo.
        The model can make mistakes, especially for visually similar
        materials such as glass, plastic, and metal.
        Do not use its prediction as the sole basis for real-world
        waste disposal or recycling decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
