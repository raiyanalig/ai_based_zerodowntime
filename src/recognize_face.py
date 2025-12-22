import cv2
import numpy as np
import pandas as pd
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      'haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if confidence < 80:
            now = datetime.now()
            df = pd.DataFrame([[id_, now.date(), now.time()]],
                              columns=["ID","Date","Time"])
            df.to_csv("attendance.csv", mode='a', header=False, index=False)

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Attendance", img)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
