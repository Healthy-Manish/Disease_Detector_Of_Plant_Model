import os
import json
from io import BytesIO

from PIL import  Image

import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_model.h5"
# Load model
model = tf.keras.models.load_model(model_path)

class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# function to load and preprocess the image using pillow

def load_and_preprocess_image(image_path,target_size=(224,224)):
    # load the image
    img = Image.open(image_path)
    #resize the image
    img = img.resize(target_size)
    #     convert the image to a numpy array
    img_array = np.array(img)
    #     add batch dimension
    img_array = np.expand_dims(img_array,axis=0)
    #     scale the image values to [0,1]
    img_array = img_array.astype('float32')/255.
    return img_array


# function to predict the class of an image
def predict_image_class(model,image_path,class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index= np.argmax(predictions,axis = 1)[0]
    predicted_class_name= class_indices[str(predicted_class_index)]
    return predicted_class_name

st.title(' Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...",type=['jpg','jpeg','png'])
if uploaded_image is not None:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_image)
        resized_img = image.resize((150, 150), Image.Resampling.LANCZOS)
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, BytesIO(uploaded_image.read()), class_indices)
            st.success(f'Prediction: {str(prediction)}')