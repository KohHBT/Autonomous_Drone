import numpy as np 
import cv2
face_cascade = cv2.CascadeClassifier("/Users/cuongphan/Desktop/Spring 2020/COSC4332/Autonomous-Drone-code/Face-Recognition-Img/haarcascade_frontalface_default.xml")
capture = cv2.VideoCapture(0)
a = 1
while(True):
    #Capture frame by frame
    a = a + 1
    check, frame = capture.read()
    print(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    #create a for loop and call the method to create a rectangular 
    for x,y,w,h in faces:
        frame = cv2.rectangle(frame,(x,y), (x+w,y+h),(0,255,0),3)
    #Generate a new fram every miliseconds 
    #Display the resulting frame 
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
print(a) #This print the number of frames have captured
capture.release()
cv2.destroyAllWindows()