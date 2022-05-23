import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.lite.python.interpreter import Interpreter
import glob
import cv2
from tensorflow.keras.applications import EfficientNetB0

img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

#Loading images from DataSet
train_path = "DataSet\Train"
valid_path = "DataSet\Validate"
test_path = "DataSet\Test"

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

#Training The Model
inputs = layers.Input(shape=(224, 224, 3))
x = img_augmentation(inputs)

#Using the imported version of EfficientNet
model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

# Freeze the pretrained weights
model.trainable = False

# Rebuild top
x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
x = layers.BatchNormalization()(x)

top_dropout_rate = 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
outputs = layers.Dense(3, activation="softmax", name="pred")(x)

# Compile
model = tf.keras.Model(inputs, outputs, name="EfficientNet")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

#Unfreeze the model
for layer in model.layers[-20:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

#Fitting
epochs = 5
start = time.time()
hist = model.fit(train_batches, epochs=epochs, validation_data=valid_batches, verbose=1)
print("Total time for training: ",time.time()-start)

# Save the tensorflow Model
model.save("model_01", save_format="h5")

#Graph
history_df = pd.DataFrame(hist.history)
print(history_df)
history_df.plot()
plt.show()
plt.savefig("validate_evaluation.png")

#Evaluation on Testing
test_labels = test_batches.classes
print("Test Labels",test_labels)
print(test_batches.class_indices)

predictions = model.predict(test_batches,steps=len(test_batches),verbose=0)

acc = 0
for i in range(len(test_labels)):
    actual_class = test_labels[i]
    if predictions[i][actual_class] > 0.7 : 
        acc += 1
print("Accuarcy:",(acc/len(test_labels))*100,"%")

