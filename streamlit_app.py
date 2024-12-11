from tensorflow.keras.models import model_from_json
import joblib
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
import cv2


def load_model(model_name: str, model_type: str):
    """
    Load a pre-trained model.

    Args:
        model_name (str): Name of the model file to load.
        model_type (str): The type of model (KNN, ANN, SVM, or CNN).

    Returns:
        Model: The loaded model.
    """
    if model_type == "CNN":
        # Load model architecture and weights
        model_data = joblib.load(model_name)
        if 'model_architecture' not in model_data or 'model_weights' not in model_data:
            raise KeyError("The .pkl file does not contain the expected keys ('model_architecture' and 'model_weights').")
        model_json = model_data['model_architecture']
        model_weights = model_data['model_weights']
        model = model_from_json(model_json)
        model.set_weights(model_weights)
        return model
    else:
        return joblib.load(model_name)  # Load models for KNN, ANN, SVM


# Function to preprocess the image for CNN
def preprocess_image_for_cnn(img) -> np.ndarray:
    """
    Preprocess the image for CNN input.

    Args:
        img (PIL.Image): The input image.

    Returns:
        np.ndarray: Preprocessed image suitable for CNN.
    """
    image_cv = np.array(img.resize((128, 128)))  # Resize to match CNN input
    image_preprocessed = image_cv / 255.0  # Normalize pixel values
    return np.expand_dims(image_preprocessed, axis=0)  # Add batch dimension

# Streamlit app
st.title("Bone Structure Analysis")
st.write("Upload an X-ray or bone scan image to analyze the structure.")

# Upload image
image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Model selection
model_type = st.selectbox("Choose a model:", ["KNN", "ANN", "SVM", "CNN"])

# Load the selected model
try:
    model_files = {
        "KNN": "knn_classifier.pkl",
        "ANN": "ann_classifier.pkl",
        "SVM": "svm_classifier.pkl",
        "CNN": "small_cnn_with_dropout.pkl" 
    }
    selected_model_file = model_files[model_type]

    # Load the model
    model = load_model(selected_model_file, model_type)

    # Print CNN model summary if selected
    if model_type == "CNN":
        st.text("CNN Model Summary:")
        model.summary(print_fn=lambda x: st.text(x))  # Display the model's architecture
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    pred_button = st.button("Analyze Bone Structure")

    if pred_button:
        # Perform image classification
        try:
            image = Image.open(image_file).convert("RGB")
            features = preprocess_image_for_cnn(image)
            probabilities = model.predict(features)[0][0]  # Get probability for "Fractured"
            prediction = "Fractured" if probabilities >= 0.5 else "Not Fractured"

            # Display prediction
            st.success(f'Predicted Structure: **{prediction}** '
                       f'Confidence: {probabilities:.2%}')
        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
