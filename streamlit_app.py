import joblib
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
import cv2  # For SIFT feature extraction
from tensorflow.keras.models import load_model as load_keras_model  # For loading CNN
from tensorflow.keras.applications.resnet50 import preprocess_input

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
    
    return joblib.load(model_name)

# Function to preprocess the image for CNN
def preprocess_image_for_cnn(img) -> np.ndarray:
    """
    Preprocess the image for CNN input.

    Args:
        img (PIL.Image): The input image.

    Returns:
        np.ndarray: Preprocessed image suitable for CNN.
    """
    image_cv = np.array(img.resize((224, 224)))  # Resize to match CNN input
    image_preprocessed = preprocess_input(image_cv)  # Apply ResNet50 preprocessing
    return np.expand_dims(image_preprocessed, axis=0)  # Add batch dimension

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
            probabilities = model.predict(features)[0]
            prediction = [np.argmax(probabilities)]  # Get class with highest probability
        else:
            features = extract_features(image)
            if model_type in ["KNN", "SVM"]:
                prediction = model.predict([features])
                probabilities = model.predict_proba([features])[0]
            elif model_type == "ANN":
                probabilities = model.predict_proba([features])[0]
                prediction = [np.argmax(probabilities)]

        # Map numeric predictions to descriptive labels
        LABEL_MAPPING = {
            0: "Not Fractured",
            1: "Fractured"
        }
        class_labels = [LABEL_MAPPING[cls] for cls in range(len(probabilities))]

        # Create a DataFrame to store predictions and probabilities
        prediction_df = pd.DataFrame({
            "Class": class_labels,
            "Probability": probabilities
        })
        return prediction_df.sort_values("Probability", ascending=False), LABEL_MAPPING[prediction[0]]

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
        "CNN": "small_cnn_with_dropout.pkl"  # CNN model file in H5 format
    }
    selected_model_file = model_files[model_type]
    model = load_model(selected_model_file, model_type)
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
