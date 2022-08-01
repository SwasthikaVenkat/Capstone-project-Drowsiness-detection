import streamlit as st
import tensorflow as tf
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model(r'C:\Users\VENKATARAMAN S R\Downloads\CAPSTONE PROJECT- drowsiness detection\newmodelepoch7.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Drowsiness Detection
         """
         )

file = st.file_uploader("Please upload a file", type=["jpg", "png","jpeg"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, width= 500)
    predictions = import_and_predict(image, model)
   
    predIdxs = np.argmax(predictions, axis=1)
    print(predIdxs)
    if predIdxs == 0 :
        st.title("Eyes are closed")

    else :
        st.title("Eyes are open")

