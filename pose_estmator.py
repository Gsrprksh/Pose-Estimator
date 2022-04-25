import cv2
import numpy as np
import mediapipe as mp


draw = mp.solutions.drawing_utils
pose = mp.solutions.pose
pose1 = pose.Pose(min_detection_confidence= 0.5,min_tracking_confidence = 0.5)

video =cv2.VideoCapture(0)
cv2.namedWindow('surya')
count = 0

while True:
    has_frame,frame = video.read()
    if not has_frame:
        print("not able to capture an image")
        break
    try:
        #converting the image to gray scale
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #lets apply the poses to the frame

        poses = pose1.process(gray)

        draw.draw_landmarks(frame,poses.pose_land_marks,pose.POSE_CONNECTIONS)

        cv2.imshow('surya',frame)

    except:
        break
    key = cv2.waitkey(1)
    if key%256 == 27:
        print('escape key pressed')
        break
    elif key%256 == 32:
        print('screen shot taken')
        name = 'surya_{}.png'.format(count)
        cv2.imwrite(name,frame)
    count+=1

video.release()
cv2.destroyWindow('surya')



    
