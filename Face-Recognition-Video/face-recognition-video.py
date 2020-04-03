import numpy as np 
import cv2
# Loading frontface cascade from openCV
face_cascade = cv2.CascadeClassifier("/Users/cuongphan/Desktop/Spring 2020/COSC4332/Autonomous-Drone-code/Face-Recognition-Img/haarcascade_frontalface_default.xml")
# With parameter 0, this trigger off the computer webcam camera
capture = cv2.VideoCapture(0)
#Variable to keep track of number of frame 
a = 1
#Videos is a collection of frame (picture) rendered continuously in the loop 
while(True):
    #Capture frame by frame
    a = a + 1
    #cap.read() return each image (frame) it captures
    check, frame = capture.read()
    print(frame)
    #Convert each colored frame (picture) into gray /black and white
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Call face cascade function to detect human face inside each rendered frame/photo 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    #create a for loop and call the method to create a rectangular around detected human face
    for x,y,w,h in faces:
        frame = cv2.rectangle(frame,(x,y), (x+w,y+h),(0,255,0),3)
    #Display the resulting frame 
    cv2.imshow('frame',frame)
    #keep openCV running and quit by press key 'q' 
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
print(a) #This print the number of frames have captured
capture.release() #release the camera 
#close openCV window 
cv2.destroyAllWindows()