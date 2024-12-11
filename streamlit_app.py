from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import joblib
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
import cv2

# Function to load a model
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
        return load_keras_model(model_name)
    else:
        return joblib.load(model_name)

# Function to preprocess the image for CNN
def preprocess_image_for_cnn(img, use_preprocess_input=False) -> np.ndarray:
    """
    Preprocess the image for CNN input.

    Args:
        img (PIL.Image): The input image.
        use_preprocess_input (bool): Whether to use `preprocess_input` for preprocessing.

    Returns:
        np.ndarray: Preprocessed image suitable for CNN.
    """
    image_cv = np.array(img.resize((128, 128)))  # Resize to match CNN input
    if use_preprocess_input:
        image_preprocessed = preprocess_input(image_cv)  # Use preprocess_input for pretrained models
    else:
        image_preprocessed = image_cv / 255.0  # Normalize pixel values for custom-trained models
    return np.expand_dims(image_preprocessed, axis=0)  # Add batch dimension

# Function to extract SIFT features
def extract_features(img) -> np.ndarray:
    """
    Extract features from the image using SIFT.

    Args:
        img (PIL.Image): The input image.

    Returns:
        np.ndarray: Feature vector of fixed size (128).
    """
    image_cv = np.array(img)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_cv, None)

    if descriptors is not None:
        return descriptors.flatten()[:128]  # Truncate/pad to fixed size
    else:
        return np.zeros(128)  # Zero vector if no features are found

# Function to classify an image
def classify_image(img: bytes, model, model_type: str) -> pd.DataFrame:
    """
    Classify the given image using the selected model and return predictions.

    Args:
        img (bytes): The image file to classify.
        model: The pre-trained model.
        model_type (str): The type of model (KNN, ANN, SVM, or CNN).

    Returns:
        pd.DataFrame: A DataFrame containing predictions and their probabilities.
    """
    try:
        image = Image.open(img).convert("RGB")

        # Extract features or preprocess image based on model type
        if model_type == "CNN":
            features = preprocess_image_for_cnn(image)
            st.text(f"Input shape for CNN: {features.shape}")
            probabilities = model.predict(features)[0]
            st.text(f"Full Predicted Probabilities: {probabilities}")
            prediction = np.argmax(probabilities)  # Get class with highest probability
        else:
            features = extract_features(image)
            if model_type in ["KNN", "SVM"]:
                prediction = model.predict([features])[0]
                probabilities = model.predict_proba([features])[0]
            elif model_type == "ANN":
                probabilities = model.predict([features])[0]
                prediction = np.argmax(probabilities)

        # Dynamic label mapping
        LABEL_MAPPING = {
            0: "Not Fractured",
            1: "Fractured"
        }
        class_labels = [LABEL_MAPPING[i] for i in range(len(probabilities))]

        # Create a DataFrame to store predictions and probabilities
        prediction_df = pd.DataFrame({
            "Class": class_labels,
            "Probability": probabilities
        })
        return prediction_df.sort_values("Probability", ascending=False), LABEL_MAPPING[prediction]

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        return pd.DataFrame(), None

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
        "CNN": "small_cnn_with_dropout.h5"
    }
    selected_model_file = model_files[model_type]

    # Load the model
    model = load_model(selected_model_file, model_type)

    # Print CNN model summary if selected
    if model_type == "CNN":
        st.text("CNN Model Summary:")
        st.text(model.summary())
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    pred_button = st.button("Analyze Bone Structure")

    if pred_button:
        # Perform image classification
        predictions_df, top_prediction = classify_image(image_file, model, model_type)

        if not predictions_df.empty:
            # Display top prediction
            st.success(f'Predicted Structure: **{top_prediction}** '
                       f'Confidence: {predictions_df.iloc[0]["Probability"]:.2%}')

            # Display all predictions
            st.write("Detailed Predictions:")
            st.table(predictions_df)
        else:
            st.error("Failed to classify the image.")
