import random
import os

import csv

import keras
import numpy as np
from sklearn.preprocessing import StandardScaler

xData = np.loadtxt("./data/train.csv", delimiter=",", skiprows=1)
yData = xData[:,0]
xData = xData[:, 1:]
xEval = np.loadtxt("./data/val.csv", delimiter=",", skiprows=1)

random.shuffle(xEval)

yEval = xEval[:,0]
xEval = xEval[:, 1:]

scaler = StandardScaler()
scaler = scaler.fit(xData)
xData = scaler.transform(xData)
xEval = scaler.transform(xEval)

for file in os.listdir("./"):
    if file.endswith(".h5"):
        model = keras.models.load_model(file)

y_pred_probs = model.predict(xEval)

with open("results.csv", mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['Actual', 'Predicted'])

    for item1, item2 in zip(yEval[:100], y_pred_probs[:100]):
        writer.writerow([item1, item2])