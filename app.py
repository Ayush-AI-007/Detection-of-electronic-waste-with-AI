import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

st.header('Image Classification Model')

model = load_model('D:\image_Classification\Image_classify.keras')
data_cat = ['Battery','Keyboard','Microwave','Mobile','Mouse','PCB','Player','Printer','Television','Washing Machine']
img_height = 180
img_width = 180
image = st.text_input(['Enter Image name','Keyboard.jpg'])

image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image)
st.write("these are " + str(np.max(score)*100))