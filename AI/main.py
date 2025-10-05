import numpy as np
import tensorflow as tf
import keras
import os
from keras import layers
from keras import ops
if not os.path.exists('ExooModel.h5'):
    inputs = keras.Input(shape=(7,))
    layer1 = layers.Dense(10, activation='relu')
    layer2 = layers.Dense(4, activation='relu')
    layerOut = layers.Dense(1, activation='sigmoid')
    output = layerOut(layer2(layer1(inputs)))
    model = keras.Model(inputs=inputs, outputs=output, name = 'ExoModel')
    model.save("ExoModel.h5")
    del model

model = keras.models.load_model("ExoModel.h5")
model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.75)])
xData = np.loadtxt("train1.csv", delimiter=",", skiprows=1)
yData = xData[:,0]
xData = xData[:, 1:]
xEval = np.loadtxt("val1.csv", delimiter=",", skiprows=1)
yEval = xEval[:,0]
xEval = xEval[:, 1:]


fin = model.fit(xData, yData, batch_size=32, epochs=100, verbose=1)
evals = model.evaluate(xEval, yEval, verbose=1)
print("Loss, Accuracy: ", evals)
model.save("ExoModel.h5")
del model