import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import boto3
import os
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# load_dotenv()
# aws_access_key_id = os.getenv('aws_access_key_id')
# aws_secret_access_key = os.getenv('aws_secret_access_key')
# aws_default_region = os.getenv('aws_default_region')

# # Create an S3 client using the hardcoded credentials
# s3 = boto3.client(
#     's3',
#     aws_access_key_id=aws_access_key_id,
#     aws_secret_access_key=aws_secret_access_key,
#     region_name=aws_default_region
# )

def get_secret():

    secret_name = "streamlit_app_credentials"
    region_name = "us-east-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = get_secret_value_response['SecretString']

    # Parse the secret and return as dictionary
    secret = json.loads(get_secret_value_response['SecretString'])
    return secret

# Retrieve secrets and set up S3 client
secrets = get_secret()
s3 = boto3.client(
    's3',
    aws_access_key_id=secrets['aws_access_key_id'],
    aws_secret_access_key=secrets['aws_secret_access_key'],
    region_name=secrets['region_name']
)

# Load the saved VGG model from S3
def load_model(model_path, bucket_name):
    try:
        # Check if the model file exists in S3
        s3.head_object(Bucket=bucket_name, Key=model_path)

        # If it exists, download the model
        temp_model_path = '/tmp/vgg_model.h5'
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)  # Remove existing model if it exists

        s3.download_file(bucket_name, model_path, temp_model_path)

        # Load the model directly from the downloaded file
        model = tf.keras.models.load_model(temp_model_path, compile=False)
        return model
    except s3.exceptions.NoSuchKey:
        raise FileNotFoundError(f"The model file {model_path} does not exist in bucket {bucket_name}.")
    except Exception as e:
        raise Exception(f"Error loading model from S3: {str(e)}")

# Function to preprocess a single image
def preprocess_image(image):
    image = image.resize((150, 150))  # Adjust the size based on your model's input size
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = image_array.reshape(1, 150, 150, 3)  # Reshape to match model input
    return image_array

# Function to make a prediction
def run_prediction(model, preprocessed_image):
    prediction = model.predict(preprocessed_image)
    return prediction

# Streamlit App
def main():
    st.title('MRI Tumor Classification')

    # Upload button
    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)  # Pass the PIL image directly

        model = None  # Initialize model variable

        try:
            # Load the VGG model from S3
            model = load_model('vgg_model.h5', 'tumor-image-rec')
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return  # Exit if model loading fails

        if model is not None:  # Ensure model is loaded before predicting
            try:
                # Run prediction
                prediction = run_prediction(model, preprocessed_image)

                # Define tumor types based on prediction indices
                tumor_types = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

                # Get the predicted class index and corresponding probability
                predicted_class_index = np.argmax(prediction[0])
                predicted_probability = prediction[0][predicted_class_index]

                # Display the results
                st.write(f"Predicted Type of Tumor: {tumor_types[predicted_class_index]}")
                st.write(f"Probability: {predicted_probability:.2f}")

                # Provide explanations for each tumor type
                st.subheader("Tumor Type Explanations")

                if predicted_class_index == 0:  # Glioma
                    st.markdown("""
                    ### Glioma Tumor:
                    Originating in the brain or spine's glial cells, gliomas represent 30% of all brain and central nervous system tumors. 
                    They are mostly malignant, with symptoms that vary depending on location, frequently resulting in seizures, cognitive impairments, or motor deficits.
                    """)
                elif predicted_class_index == 1:  # Meningioma
                    st.markdown("""
                    ### Meningioma Tumor:
                    Meningiomas, arising from the meninges enveloping the brain and spinal cord, are slow-growing tumors. 
                    While they typically are asymptomatic, they can cause seizures, cognitive decline, vision disturbances, or motor deficits depending on their location.
                    """)
                elif predicted_class_index == 3:  # Pituitary
                    st.markdown("""
                    ### Pituitary Tumor:
                    Pituitary adenomas, mostly benign, occur in the pituitary gland, accounting for 10-25% of intracranial neoplasms. 
                    They may cause hormonal imbalances, resulting in a variety of symptoms including headaches, vision changes, or hormonal dysfunctions.
                    """)
                else:  # No Tumor
                    st.markdown("""
                    ### No Tumor:
                    The prediction indicates that there are no tumors detected in the scanned MRI image.
                    """)

            except Exception as e:
                st.error(f"Error running prediction: {str(e)}")

if __name__ == '__main__':
    main()
