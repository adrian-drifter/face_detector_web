import pwd
from tensorflow.keras.models import load_model
from PIL import Image
#from IPython.display import Image
from tensorflow.keras.preprocessing.image import load_img
import streamlit as st
import matplotlib.pyplot as plt
import pylab
import numpy as np
import cv2
import os
from io import BytesIO

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

age_model = load_model('./models/age_model_pretrained.h5')
gender_model = load_model('./models/gender_model_pretrained.h5')
emotion_model = load_model('./models/emotion_model_pretrained.h5')

age_labels = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
gender_labels = ['male', 'female']
emotion_labels = ['Happy', 'Angry/Sad', 'Neutral']

def prediction(img):

    test_image = cv2.imread(img)
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./models/cv2_cascade_classifier/haarcascade_frontalface_default.xml')
    #glass_cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces)==0:
        return 'Face Not Detected \nSometimes glasses or tilted faces are a limitation of the model'
    else:
        i = 0
        subjects = []
        for (x,y,w,h) in faces:
            i = i+1
            cv2.rectangle(test_image,(x,y),(x+w,y+h),(203,12,255),2)

            img_gray= gray[y:y+h,x:x+w]

            #emotion prediction
            emotion_img = cv2.resize(img_gray, (48, 48), interpolation = cv2.INTER_AREA)
            emotion_img_array = np.array(emotion_img)
            emotion_input = np.expand_dims(emotion_img_array, axis=0)
            output_emotion = emotion_labels[np.argmax(emotion_model.predict(emotion_input, verbose=0))]

            #gender prediction
            gender_img= cv2.resize(img_gray, (100,100), interpolation = cv2.INTER_AREA)
            gender_img_array = np.array(gender_img)
            gender_input = np.expand_dims(gender_img_array, axis=0)
            output_gender = gender_labels[np.argmax(gender_model.predict(gender_input, verbose=0))]

            #age prediction
            age_img = cv2.resize(img_gray, (200,200), interpolation = cv2.INTER_AREA)
            age_input = age_img.reshape(-1, 200, 200, 1)
            output_age = age_labels[np.argmax(age_model.predict(age_input, verbose=0))]

            output_str = f'\nSubject: {str(i)} \nGender: {output_gender} \nAge range: {output_age} \nEmotion: {output_emotion}\n'
            #print(output_str)

            info= {'Subject': i, 'Gender': output_gender, 'Age': output_age, 'Emotion': output_emotion}
            subjects.append(info)
            col = (0,255,0)
            cv2.putText(test_image, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

        #plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        st.image(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    return subjects


def prediction2(img):

    test_image = img
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./models/cv2_cascade_classifier/haarcascade_frontalface_default.xml')
    #glass_cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return 'Face Not Detected \nSometimes glasses or tilted faces are a limitation of the model'
    else:
        i = 0
        subjects = []
        for (x, y, w, h) in faces:
            i = i + 1
            cv2.rectangle(test_image, (x, y), (x + w, y + h), (203, 12, 255),
                          2)

            img_gray = gray[y:y + h, x:x + w]

            #emotion prediction
            emotion_img = cv2.resize(img_gray, (48, 48),
                                     interpolation=cv2.INTER_AREA)
            emotion_img_array = np.array(emotion_img)
            emotion_input = np.expand_dims(emotion_img_array, axis=0)
            output_emotion = emotion_labels[np.argmax(
                emotion_model.predict(emotion_input, verbose=0))]

            #gender prediction
            gender_img = cv2.resize(img_gray, (100, 100),
                                    interpolation=cv2.INTER_AREA)
            gender_img_array = np.array(gender_img)
            gender_input = np.expand_dims(gender_img_array, axis=0)
            output_gender = gender_labels[np.argmax(
                gender_model.predict(gender_input, verbose=0))]

            #age prediction
            age_img = cv2.resize(img_gray, (200, 200),
                                 interpolation=cv2.INTER_AREA)
            age_input = age_img.reshape(-1, 200, 200, 1)
            output_age = age_labels[np.argmax(
                age_model.predict(age_input, verbose=0))]

            output_str = f'\nSubject: {str(i)} \nGender: {output_gender} \nAge range: {output_age} \nEmotion: {output_emotion}\n'
            #print(output_str)

            info = {
                'Subject': i,
                'Gender': output_gender,
                'Age': output_age,
                'Emotion': output_emotion
            }
            subjects.append(info)
            col = (0, 255, 0)
            cv2.putText(test_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
        proc_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        # plt.imshow(proc_image)
        # plt.show()
        #st.image(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    return subjects


def prediction3(img):

    test_image = img
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        './models/cv2_cascade_classifier/haarcascade_frontalface_default.xml')
    #glass_cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return 'Face Not Detected \nSometimes glasses or tilted faces are a limitation of the model'
    else:
        i = 0
        subjects = []
        for (x, y, w, h) in faces:
            i = i + 1
            cv2.rectangle(test_image, (x, y), (x + w, y + h), (203, 12, 255),
                          2)

            img_gray = gray[y:y + h, x:x + w]

            #emotion prediction
            emotion_img = cv2.resize(img_gray, (48, 48),
                                     interpolation=cv2.INTER_AREA)
            emotion_img_array = np.array(emotion_img)
            emotion_input = np.expand_dims(emotion_img_array, axis=0)
            output_emotion = emotion_labels[np.argmax(
                emotion_model.predict(emotion_input, verbose=0))]

            #gender prediction
            gender_img = cv2.resize(img_gray, (100, 100),
                                    interpolation=cv2.INTER_AREA)
            gender_img_array = np.array(gender_img)
            gender_input = np.expand_dims(gender_img_array, axis=0)
            output_gender = gender_labels[np.argmax(
                gender_model.predict(gender_input, verbose=0))]

            #age prediction
            age_img = cv2.resize(img_gray, (200, 200),
                                 interpolation=cv2.INTER_AREA)
            age_input = age_img.reshape(-1, 200, 200, 1)
            output_age = age_labels[np.argmax(
                age_model.predict(age_input, verbose=0))]

            output_str = f'\nSubject: {str(i)} \nGender: {output_gender} \nAge range: {output_age} \nEmotion: {output_emotion}\n'
            #print(output_str)

            info = {
                'Subject': i,
                'Gender': output_gender,
                'Age': output_age,
                'Emotion': output_emotion
            }
            subjects.append(info)
            col = (0, 255, 0)
            cv2.putText(test_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, col, 2)
        #plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        img = Image.open(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    return img



def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image
