import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import boto3

# Load the saved model from S3
def load_model(model_path, bucket_name):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, model_path, '/tmp/model.keras')
    model = tf.keras.models.load_model('/tmp/model.keras')
    return model

# Function to preprocess a single image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Adjust the size based on your model's input size
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = image_array.reshape(1, 224, 224, 3)  # Reshape to match model input
    return image_array

# # Load the saved model from the local disk
# def load_model(model_path):
#     model = tf.keras.models.load_model(model_path)
#     return model

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
        ('VGG16', 'EfficientNetB0')
    )

    # Upload button
    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(uploaded_file)

        try:
            # Load the selected model from S3
            if model_option == 'VGG16':
                model = load_model('vgg16_best_model.keras', 'tumor-image-rec')
            else:
                model = load_model('efficientnetB0.keras', 'tumor-image-rec')
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

        try:
            # Run prediction
            prediction = run_prediction(model, preprocessed_image)
            st.write(f"Prediction: {format_results(prediction)}")
        except Exception as e:
            st.error(f"Error running prediction: {str(e)}")

if __name__ == '__main__':
    main()