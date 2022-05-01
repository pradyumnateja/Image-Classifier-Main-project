from utils import *
import tensorflow as tf
model = tf.keras.models.load_model('mobilenet_model.h5')

import streamlit as st
st.write("""
         # Image Classifier
         """
         )
st.write("My model can classify images from 8 categories!")
st.write("Airplanes | Cars | Cats | Dogs | Flowers | Fruits | Motorbikes | Person")
file = st.file_uploader("Upload an image file to classify", type=["jpg", "png"])


import cv2
from PIL import Image, ImageOps
import numpy as np

class_labels = {0: 'Airplane',
                1: 'Car',
                2: 'Cat',
                3: 'Dog',
                4: 'Flower',
                5: 'Fruit',
                6: 'Motorbike',
                7: 'Person'}

def import_and_predict(image_data, model):

        size = (224,224)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC))/255.

        img_reshape = img_resize[np.newaxis,...]

        prediction = model.predict(img_reshape)
        prediction_class = prediction.argmax(axis=1)
        predicted_class = class_labels[int(prediction_class)]

        return predicted_class, 100*np.max(prediction)

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=False)
    predicted_class, probability = import_and_predict(image, model)
    st.write('Predicted Class: ', predicted_class)
    st.write('Probability: ', probability)
