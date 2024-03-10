import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


class MyModels:
    def __init__(self):
        self.densenet = load_model('models\inceptionV3.h5')
        self.inception = load_model('models\densenet169.h5')
        self.mobilenet = load_model('models\mobilenetV2.h5')

    def densenet_model(self, path):
        try:
            print(path)
            img = load_img(path, target_size=(224, 224))
            img_tensor = img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.0
            pred = self.densenet.predict(img_tensor)
            if pred[0][1] > 0.5:
                return "Yes pneumonia"
            else:
                return "No pneumonia"
        except Exception as e:
            print("Error in densenet function:", e)

    def inception_model(self, path):
        try:
            print(path)
            img = load_img(path, target_size=(224, 224))
            img_tensor = img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.0
            pred = self.inception.predict(img_tensor)
            if pred[0][1] > 0.5:
                return "Yes pneumonia"
            else:
                return "No pneumonia"
        except Exception as e:
            print("Error in inception function:", e)

    def mobilenet_model(self, path):
        try:
            print(path)
            img = load_img(path, target_size=(224, 224))
            img_tensor = img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.0
            pred = self.mobilenet.predict(img_tensor)
            if pred[0][1] > 0.5:
                return "Yes pneumonia"
            else:
                return "No pneumonia"
        except Exception as e:
            print("Error in inception function:", e)
