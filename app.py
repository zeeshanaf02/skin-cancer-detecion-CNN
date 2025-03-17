import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Cache the model loading for better performance
@st.cache_resource
def load_skin_cancer_model():
    # Path to the TensorFlow model
    model_path = "C:/Zeeshan/trained images/converted_keras/keras_model.h5"
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error("Failed to load the model. Ensure the model path is correct.")
        st.stop()

# Load the model
model = load_skin_cancer_model()
model.summary()

# Streamlit app starts here
st.title("Skin Cancer Detection App")
st.header("Upload an image to predict skin cancer")

# Instructions for the user
st.info(
    """
    **Instructions:**
    1. Upload an image of the skin lesion (JPG, PNG, or JPEG formats).
    2. The app will preprocess the image and provide a prediction.
    3. This tool is for educational purposes only and not a substitute for medical advice.
    """
)

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Load and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for the model
        image = image.resize((224, 224))  # Resize to model's expected input
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make a prediction
        st.write("Predicting...")
        prediction = model.predict(image_array)

        # Display probabilities
        classes = ["Benign", "Malignant"]
        probabilities = {cls: round(prob * 100, 2) for cls, prob in zip(classes, prediction[0])}
        st.subheader("Prediction Probabilities:")
        st.write(probabilities)

        # Display the final class prediction
        predicted_class = classes[np.argmax(prediction)]
        st.success(f"The model predicts: **{predicted_class}**")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
else:
    st.warning("Please upload an image to proceed.")