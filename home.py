import streamlit as st
import tensorflow
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

keras_model_hindhi = tensorflow.keras.models.load_model('hindhi1.h5')
keras_model_hindhi.summary()
keras_model_bengali=tensorflow.keras.models.load_model('bengali1.h5')
keras_model_bengali.summary()
keras_model_cedar=tensorflow.keras.models.load_model('cedar3.h5')
keras_model_cedar.summary()
keras_model=tensorflow.keras.models.load_model('overall3.h5')
keras_model.summary()
cedar_real_path = r'C:\Users\shamr\Desktop\CEDAR\original'
cedar_forged_path = r'C:\Users\shamr\Desktop\CEDAR\forged'
bengali_real_path = r'C:\Users\shamr\Desktop\sign1'
bengali_forged_path = r'C:\Users\shamr\Desktop\signf'
hindhi_real_path = r'C:\Users\shamr\Desktop\Hindhi\original'
hindhi_forged_path = r'C:\Users\shamr\Desktop\Hindhi\forged'
#c1,c2,c3=st.columns(3)
st.title("Overall image recognition")
x3=st.text_input("Give the path of any image from hindhi or bengali or cedar")
b4=st.button("Predict")
if b4:
    if x3 is not None:
        #st.write(os.path.join(hindhi_forged_path,x2.name))
        #image=cv2.imread(os.path.join(hindhi_forged_path,x2.name))
        image=cv2.imread(x3)
        #st.write(image)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (224, 224))
            #images.append(image)
            #images = np.array(image)

            m=keras_model_hindhi.predict(image.reshape((-1, 224, 224, 1)))
            st.write(m)
            if m[0]<0.5:
                st.write("original sign")
            else:
                st.write("forged sign")


st.title("Cedar image recognition")
#x=st.file_uploader('Upload a cedar image', type=['jpg', 'jpeg', 'png'])
x=st.text_input("Give the path of a Cedar image")
b6=st.button("Predict Cedar")
if b6:
    if x is not None:
        #st.write(os.path.join(cedar_forged_path,x.name))
        #image=cv2.imread(os.path.join(cedar_forged_path,x.name))
        image=cv2.imread(x)
        #st.write(image)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (224, 224))
            #images.append(image)
            #images = np.array(image)

            m=keras_model_cedar.predict(image.reshape((-1, 224, 224, 1)))
            st.write(m)
            if m[0]<0.5:
                st.write("original sign")
            else:
                st.write("forged sign")
#x1=st.file_uploader('Upload a bengali image', type=['jpg', 'jpeg', 'png','ttf'])
st.title("Bengali image recognition")
x1=st.text_input("Give the path of a Bengali image")
b3=st.button("Predict Bengali")

if b3:
    
    if x1 is not None:
        #st.write(os.path.join(bengali_forged_path,x1.name))
        #image=cv2.imread(os.path.join(bengali_forged_path,x1.name))
        image=cv2.imread(x1)
        #st.write(image)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (224, 224))
            #images.append(image)
            #images = np.array(image)

            m=keras_model_bengali.predict(image.reshape((-1, 224, 224, 1)))
            #m=m.values
            st.write(m)
            if m[0]<0.5:
                st.write("original sign")
            else:
                st.write("forged sign")
#x2=st.file_uploader('Upload a hindhi image', type=['jpg', 'jpeg', 'png'])
st.title("Hindhi image recognition")
x2=st.text_input("Give the path of a hindhi image")
b5=st.button("Predict Hindhi")
if b5:
    if x2 is not None:
        #st.write(os.path.join(hindhi_forged_path,x2.name))
        #image=cv2.imread(os.path.join(hindhi_forged_path,x2.name))
        image=cv2.imread(x2)
        #st.write(image)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (224, 224))
            #images.append(image)
            #images = np.array(image)

            m=keras_model_hindhi.predict(image.reshape((-1, 224, 224, 1)))
            st.write(m)
            if m[0]<0.5:
                st.write("original sign")
            else:
                st.write("forged sign")
