import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import keras

# Load TensorFlow SavedModel as an inference-only layer
model = keras.layers.TFSMLayer('v3_pred_cott_dis.h5py', call_endpoint='serving_default')

# Define Streamlit app
def main():
    # Set app title
    st.title('Cotton Plant Disease Detection')
    # Set app description
    st.write('This app helps you to detect whether a cotton plant is healthy or diseased.')
    st.write('NOTE- This model only works on Cotton Plant. (With appropriate Image)')
    # Add file uploader for input image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    # If file uploaded, display it and make prediction
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        # Display image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Preprocess image
        image = preprocess_image(image)
        # Make prediction
        prediction = model.predict(image)
        # Convert prediction to label and confidence score
        label = np.argmax(prediction)
        confidence = prediction[0][label]
        label_text = 'diseased' if label == 1 else 'healthy'
        # Display prediction
        st.write('Prediction: {} (confidence score: {:.2%})'.format(label_text, confidence))

# Define function to preprocess input image
def preprocess_image(image):
    # Resize image
    image = image.resize((150, 150))
    # Convert image to numpy array
    image = np.array(image)
    # Scale pixel values to range [0, 1]
    image = image / 255.0
    # Expand dimensions to create batch of size 1
    image = np.expand_dims(image, axis=0)
    return image

# Run Streamlit app
if __name__ == '__main__':
    main()
