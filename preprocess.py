import time
import cv2
import mtcnn
import numpy as np
from sklearn.preprocessing import Normalizer
from PIL import Image
from keras_facenet import FaceNet
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def face_extract():
    print('Face The Camera')
    camera = cv2.VideoCapture(0)
    time.sleep(1)

    return_value, image = camera.read()
    del camera

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = np.asarray(image)
    detector = mtcnn.MTCNN()
    results = detector.detect_faces(pixels)

    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize((160, 160))
    face_array = np.asarray(image)
    face_array = np.expand_dims(face_array, axis=0)
    return face_array


def training(x):
    model = FaceNet()
    emb = model.embeddings(x)
    normalizer = Normalizer(norm='l2')
    normalizer.transform(emb)
    return emb
