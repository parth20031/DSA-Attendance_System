import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize

studentImg = os.listdir("student_details1")

imgList = []
RollList = []

for image in studentImg:
    imgList.append(cv2.imread(os.path.join("student_details1", image)))
    RollList.append(image[:-4])

def get_face_embeddings(imgList, model):
    embeddings = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img)
        img = resize(img, (160, 160))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        embedding = model.predict(img)
        embeddings.append(embedding[0])

    return embeddings

# Load the FaceNet model (replace with the actual path to your model)
model = load_model('./openface-model/openface/models/openface')

embeddings = get_face_embeddings(imgList, model)

encodes = [embeddings, RollList]
with open("facenet_encodings.p", "wb") as file:
    pickle.dump(encodes, file)

print("Done")
