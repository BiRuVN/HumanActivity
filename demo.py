import os
import cv2
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *

WIDTH = 64
HEIGHT = 64
CHANNEL = 3
classes_list = ['PushUps', 'JumpRope', 'Lunges']
total_avg_frame = 25
skip_frame = 2
count_skip_frame = 0

model = tf.keras.models.load_model('./models/cnnavg-0408.h5')
model.summary()

cam = cv2.VideoCapture(0)

seq = deque(maxlen = total_avg_frame)
while True:
    ret, frame = cam.read()
    if not ret:
        break

    cv2.imshow('frame', frame)

    if count_skip_frame < skip_frame:
        count_skip_frame += 1
        continue
    count_skip_frame = 0
    frame = cv2.resize(frame, (HEIGHT, WIDTH))/255.0

    ip = np.expand_dims(frame, 0)
    pred = model.predict(ip)[0]
    
    if len(seq) < total_avg_frame:
        seq.append(pred)
        continue
    seq.append(pred)
    seq_np = np.array(seq)
    seq_mean = seq_np.mean(axis=0)
    cl = classes_list[np.argmax(seq_mean)]

    if np.max(seq_mean) > 0.99:
        print(f"{cl} with {np.max(seq_mean)}")
    else:
        print(f"Break with {np.max(seq_mean)}")

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()