import streamlit as st
import keras 
import tensorflow as tf


@st.cache(allow_output_mutation=True)
def load_models():
  model=keras.models.load_model('image_recognition_model.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_models()

st.write("""
         # Image Recognition
         """

         )

results={
   0:'aeroplane',
   1:'automobile',
   2:'bird',
   3:'cat',
   4:'deer',
   5:'dog',
   6:'frog',
   7:'horse',
   8:'ship',
   9:'truck'
} 
file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
    size = (32,32)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
    img_reshape = img[np.newaxis,...]
    
    prediction = model.predict(img_reshape)
        
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    
    arg=np.argmax(score)
    print('Its a '+ str(results[arg]))
    
    
    st.success(results[arg])
    
