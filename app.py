
from cv2 import COLOR_BGR2RGB
from flask import Flask,request,render_template,Response,url_for,redirect
import numpy as np
import mediapipe as mp


import cv2


app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

def generator():
    video = cv2.VideoCapture(0)
    while True:
        success,frame=video.read()
        if not success:
            break
        else:
            cascade = cv2.CascadeClassifier(r'C:\Users\prakash\Desktop\camera accessing using flask\haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            RGB = cv2.cvtColor(frame,COLOR_BGR2RGB)
            a= mp.solutions.drawing_utils
            b = mp.solutions.pose
            c = b.Pose(min_detection_confidence = 0.5,min_tracking_confidence = 0.5)
            d = c.process(RGB)
            a.draw_landmarks(frame,d.pose_landmarks,b.POSE_CONNECTIONS)

            values = cascade.detectMultiScale(gray, 1.1,3)
            for (x,y,w,h) in values:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            

            ret, buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()
        

        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    

@app.route('/video')
def video():
    return Response(generator(),mimetype= 'multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug = True)