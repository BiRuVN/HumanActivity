import os
import cv2
import numpy as np
import random
import mediapipe as mp
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *

WIDTH = 360
HEIGHT = 240
timestep = 10
class_list = ['JumpRope', 'Lunges', 'PushUps']
selected_landmarks = [0,11,12,13,14,15,16,23,24,25,26,27,28]
holistic = mp.solutions.holistic.Holistic(model_complexity=2)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

model = tf.keras.models.load_model('./models/lm13-lstm-0908-200-0.2201.h5')
model.summary()

cam = cv2.VideoCapture(0)

seq = deque(maxlen = timestep)
while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame.flags.writeable = False
    results = holistic.process(frame)

    if not results.pose_landmarks:
        cv2.imshow('frame', frame)
        continue

    frame.flags.writeable = True
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    cv2.imshow('frame', frame)

    landmarks = [lm.visibility*np.sqrt(((lm.x*WIDTH)**2 + (lm.y*HEIGHT)**2)/(WIDTH**2 + HEIGHT**2)) for i, lm in enumerate(results.pose_landmarks.landmark) if i in selected_landmarks]
    # landmarks = [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]

    # print(landmarks)
    # if count_skip_frame < skip_frame:
    #     count_skip_frame += 1
    #     continue
    # count_skip_frame = 0
    # frame = cv2.resize(frame, (HEIGHT, WIDTH))/255.0
    
    if len(seq) < timestep:
        seq.append(landmarks)
        continue
    seq.append(landmarks)

    ip = np.expand_dims(seq, 0)
    pred = model.predict(ip)[0]

    if np.max(pred) > 0.95:
        pred_id = np.argmax(pred, axis=0)
        print(f"{class_list[pred_id]} with {np.max(pred)}")
    else:
        print(f"Break with {np.max(pred)}")

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()