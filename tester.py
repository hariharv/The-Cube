import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import random

import keras
import numpy as np

xEval = np.loadtxt("data/val/val.csv", delimiter=",", skiprows=1)

random.shuffle(xEval)

yEval = xEval[:,0]
xEval = xEval[:, 1:]

model = keras.models.load_model('32-64-32.h5')

y_pred_probs = model.predict(xEval)
for i in range(10):
    print(f'Actual: {yEval[i]}, Predicted: {y_pred_probs[i]}')