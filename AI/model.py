import numpy as np
<<<<<<< HEAD
=======
from sklearn.preprocessing import StandardScaler
>>>>>>> repoB/main
import tensorflow as tf
import keras
import os
from keras import layers
<<<<<<< HEAD
from keras import ops
if not os.path.exists('ExoModel.h5'):
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
xData = np.loadtxt("remaining_70.csv", delimiter=",", skiprows=1)
yData = xData[:,0]
xData = xData[:, 1:]
xEval = np.loadtxt("sample_30.csv", delimiter=",", skiprows=1)
yEval = xEval[:,0]
xEval = xEval[:, 1:]


fin = model.fit(xData, yData, batch_size=32, epochs=100, verbose=1)
evals = model.evaluate(xEval, yEval, verbose=1)
print("Loss, Accuracy: ", evals)
model.save("ExoModel.h5")
del model
=======
import matplotlib.pyplot as plt
import pandas as pd
from keras import ops
#if not os.path.exists('Mod8-16-8.h5'):
inputs = keras.Input(shape=(8,))
layer1 = layers.Dense(16, activation='relu')
layer2 = layers.Dense(32, activation='relu')
layer3 = layers.Dense(16, activation='relu')
layerOut = layers.Dense(1, activation='sigmoid')
output = layerOut(layer3(layer2(layer1(inputs))))
model = keras.Model(inputs=inputs, outputs=output, name = 'Mod8-16-8')
keras.saving.save_model(model, "Mod8-16-8.h5")
del model

model = keras.models.load_model("Mod8-16-8.h5")
model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

df = pd.read_csv("./data/merged.csv")
sample_df = df.sample(frac=0.3, random_state=42)
remaining_df = df.drop(sample_df.index)
sample_df.to_csv("./data/val.csv", index=False)
remaining_df.to_csv("./data/train.csv", index=False)

xData = np.loadtxt("./data/train.csv", delimiter=",", skiprows=1)
yData = xData[:,0]
xData = xData[:, 1:]
xEval = np.loadtxt("./data/val.csv", delimiter=",", skiprows=1)
yEval = xEval[:,0]
xEval = xEval[:, 1:]

scaler = StandardScaler()
scaler = scaler.fit(xData)
xData = scaler.transform(xData)
xEval = scaler.transform(xEval)

history = model.fit(xData, yData, validation_data=(xEval, yEval), batch_size=64, epochs=300, verbose=1)
evals = model.evaluate(xEval, yEval, verbose=1)
print("Loss, Accuracy: ", evals)
model.save("Mod8-16-8.h5")
del model

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
>>>>>>> repoB/main
