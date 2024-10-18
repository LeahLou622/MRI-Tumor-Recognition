import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Function to preprocess a single image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Adjust the size based on your model's input size
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = image_array.reshape(1, 224, 224, 3)  # Reshape to match model input
    return image_array

# Load the saved model (VGG16 or EfficientNetB0)
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to make a prediction
def run_prediction(model, preprocessed_image):
    prediction = model.predict(preprocessed_image)
    return prediction

# Function to convert prediction to JSON format
def format_results(prediction):
    result = {"predictions": prediction.tolist()}
    return json.dumps(result)

# Streamlit App
def main():
    st.title('MRI Tumor Classification')

    # Model selection dropdown
    model_option = st.selectbox(
        'Select the model to use for prediction:',
        ('VGG16', 'EfficientNetB0')  # Updated to reflect the models being used
    )

    # Upload button
    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(uploaded_file)

        # Load the selected model
        if model_option == 'VGG16':
            model = load_model('vgg16_best_model.keras')
        else:
            model = load_model('efficientnetB0.keras')

        # Run prediction
        prediction = run_prediction(model, preprocessed_image)

        # Display result
        st.write(f"Prediction: {format_results(prediction)}")

if __name__ == '__main__':
    main()

