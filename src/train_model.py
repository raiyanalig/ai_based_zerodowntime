import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      'haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    image_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            image_paths.append(os.path.join(root, file))

    face_samples = []
    ids = []

    for imagePath in image_paths:
        gray_img = Image.open(imagePath).convert('L')
        img_np = np.array(gray_img, 'uint8')
        user_id = int(imagePath.split("_")[1].split("/")[0])
        faces = face_detector.detectMultiScale(img_np)

        for (x,y,w,h) in faces:
            face_samples.append(img_np[y:y+h,x:x+w])
            ids.append(user_id)

    return face_samples, ids

faces, ids = get_images_and_labels("dataset")
recognizer.train(faces, np.array(ids))
os.makedirs("trainer", exist_ok=True)
recognizer.save("trainer/trainer.yml")

print("Model training completed")
