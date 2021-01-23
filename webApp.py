import cv2
import streamlit as st
import tensorflow as tf
import numpy as np




model = tf.keras.models.load_model('model_simple_v2.hdf5',compile = False)
category = ["without mask", "with mask"]
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
size = 4

st.title("Webcam Live Feed")
videoCapture_input = 0
run = st.checkbox('Run')
if run:
    parameter = st.text_input("Enter the video URL")
    if parameter:
        videoCapture_input = parameter + "/video"
        
    
    frame_window = st.image([])
    camera = cv2.VideoCapture(videoCapture_input)

    if st.button('Run Camera', key = 'start'):
        while True:
            _, frame = camera.read()
            frame = cv2.flip(frame,1,1)
            shrink = cv2.resize(frame,(frame.shape[1]//size, frame.shape[0]//size))
            faces = face_classifier.detectMultiScale(shrink)
            for f in faces:
                (x,y,width,height) = [v*size for v in f]
                saved_faces = frame[y:y+height, x:x+width]
                resize_faces = cv2.resize(saved_faces,(225,225))
                reshape_faces = np.reshape(resize_faces,(1,225,225,3))
                prediction = model.predict(reshape_faces)
                rounding_prediction = np.round(prediction[0])
                
                cv2.rectangle(frame,(x,y),(x+width,y+height),(0,0,255),2)
                cv2.putText(frame,category[int(rounding_prediction[0])],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
            
            
            frame_window.image(frame)
            
            
                

    
    
    