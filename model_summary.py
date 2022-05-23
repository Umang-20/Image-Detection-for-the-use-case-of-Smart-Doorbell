import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model("model_01")
print(model.summary())