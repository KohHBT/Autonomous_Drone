import cv2
#Create a CasadeClassificer Object 
face_cascade = cv2.CascadeClassifier("/Users/cuongphan/Desktop/Spring 2020/COSC4332/Autonomous-Drone-code/Face-Recognition-Img/haarcascade_frontalface_default.xml")
#Read img as it is 
img = cv2.imread("t_n_avatar.jpeg")
#Read the image as gray scale image 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
#Search the face rectangle coordinate of the image 
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
print(type(faces))
print(faces)
#create a for loop and call the method to create a rectangular 
for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y), (x+w,y+h),(0,255,0),3)
cv2.imshow("TeamMember",img)
cv2.imwrite("/Users/cuongphan/Desktop/Spring 2020/COSC4332/Autonomous-Drone-code/tn-face-recognized.jpg",img)
cv2.waitKey(0)
# if k == 27:
#     cv2.destroyAllWindows()