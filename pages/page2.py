import streamlit as st
import matplotlib.pyplot as plt

def app():
    st.markdown("""
                # Documentation
                """)
    st.write("""
             Te current web application uses 3 pre-trained neural networks. Trained with the UTK dataset for age gender detection (https://susanqq.github.io/UTKFace/),
             CK+48 dataset for emotion detection (https://www.kaggle.com/gauravsharma99/ck48-5-emotions) and a combination of UTK and other images, with augmentation for age detection.
             """)
    st.write("""
            For face detection it uses the Haar Cascade algorithm from OpenCV.
            The random face generator utilizes the random user generator API (https://randomuser.me/)
             """)
