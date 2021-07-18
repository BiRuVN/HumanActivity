import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *

WIDTH = 128
HEIGHT = 128
CHANNEL = 3
nb_frame = 8

model = tf.keras.models.load_model('./models/convnet-lstm-1807-20-0.0267.h5')
model.summary()

cap = cv2.VideoCapture(0)

cld = {0: 'PullUps', 1: 'PushUps', 2: 'JumpRope'}
seq = []
step_frame = 2
step = 0
while True:
	ret, frame = cap.read()
	if not ret:
		break

	cv2.imshow('cam', frame)

	if frame is None:
		print("frame is None")
		break

	if step != step_frame:
		step += 1
		continue

	step = 0
	frame = cv2.resize(frame, (WIDTH, HEIGHT))

	if len(seq) < nb_frame:
		seq.append(frame)
		continue

	seq = seq[1:] + [frame]

	ip = np.expand_dims(np.array(seq), axis=0)

	pred = model.predict(ip)[0]
	pred_id = np.argmax(pred, axis=1)
	c = max(list(pred_id), key=list(pred_id).count)
	# if pred[pred_id] < 0.8:
	# 	print(f"Break with confidence {pred[pred_id]}")
	# else:
	# 	print(cld[pred_id], f" with confidence {pred[pred_id]}")
	
	if np.sum(pred_id==c) < 6:
		print(f"Break with confidence {np.sum(pred_id==c)}")
	else:
		print(cld[c], f" with confidence {np.sum(pred_id==c)}")

	if cv2.waitKey(1) == 27:
		break
cap.release()
cv2.destroyAllWindows()