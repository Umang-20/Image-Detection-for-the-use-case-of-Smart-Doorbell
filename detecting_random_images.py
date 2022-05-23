
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

model = load_model("model_01")

categories = {0: "Cat", 1: "Dog",2: "Face"}
image_path = "Random_Images"

images =[]

for img in os.listdir(image_path):
    img = os.path.join(image_path,img)
    
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    images.append(img)


for i in images:
    result = model.predict(i)
    label = np.argmax(result)
    print(categories[label])