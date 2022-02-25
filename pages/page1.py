from pandas.core.frame import DataFrame
import streamlit as st
from utils import prediction, prediction2
import requests
import PIL.Image
import matplotlib.pyplot as plt
from IPython.display import Image
from imutils import url_to_image
import pylab
import numpy as np
#import cv2
import os



pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def app():


    #st.markdown("# Face detector. \n ## Predict Age range, Gender and Emotion")

    st.write("""
            This web app contains a series of 3 separate Convolutional Neural Networks to detect
            faces, and predict the age among different ranges, gender (male or female) and emotions (Happy, Angry/sad or Neutral).
            """)
    st.write("""
            * Upload the image you want to scan.
            * Try to use images with frontfaces and no sunglasses or mask.
            * This model can detect individual photos or group photos
            """)


    direction = st.radio('Select image', ('Upload image', 'Generate a random image'))
    st.write(direction)

    if direction == 'Upload image':
        ######################## image uploader ############################
        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader("Choose a JPG or PNG file", type=['jpg','png'])
        if uploaded_file is not None:
            image = uploaded_file
            st.image(image, caption='Image loaded', use_column_width=True)

            img = PIL.Image.open(uploaded_file)
            img = img.save('img.jpg')


            st.write(prediction('./img.jpg'))
    if direction == 'Generate a random image':

        if st.button('Press here'):
            # print is visible in the server output, not in the page
            print('button clicked!')
            st.write('Fake human generated')


            url = 'https://randomuser.me/api/'
            response = requests.get(url).json()
            img_url = response['results'][0]['picture']['large']
            pil_image = Image(img_url)
            url_ar = url_to_image(img_url)

            st.image(img_url)

            #st.image(prediction2(url_ar))
            st.write(prediction2(url_ar))

        else:
            st.write('Not clicked yet')


    # if st.button('Detect'):

    #     print(prediction2(url_ar))

    # else:
    #     print('Not detected')
