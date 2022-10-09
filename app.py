
import cv2
import os
import numpy as np
from skimage.filters import gaussian
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('model/cat_dog_CNN_SVM.h5')



st.title('Pet Classifier')

st.sidebar.title('Pet Classifier')
st.sidebar.subheader('CNN + Binary Classification')




img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])
from tempfile import NamedTemporaryFile

#if file uploader is used---------
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    demo_image = img_file_buffer
    #display image
    st.image(demo_image,use_column_width = True)

    
    temp_file = NamedTemporaryFile(delete=False)
    temp_file.write(img_file_buffer.getvalue())
    demo_image = temp_file.name

#set a default image when no image is uploaded
else:
    demo_image = "image/1.jpg"
    #image = np.array(Image.open(demo_image))


    #display image
    st.image(demo_image,use_column_width = True)


#predict
test_pic = tf.keras.preprocessing.image.load_img(demo_image, target_size=(64,64))
test_pic = tf.keras.preprocessing.image.img_to_array(test_pic)
test_pic = test_pic/255 #normalization
test_pic = np.expand_dims(test_pic, axis=0)
prediction = model.predict(test_pic)
print(prediction)


# Cat < 0 >= Dog
if prediction[0] < 0:
    st.markdown("Prediction: Cat")
else:
    st.markdown("Prediction: Dog")




