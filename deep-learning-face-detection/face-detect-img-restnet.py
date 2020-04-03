import numpy as np
import argparse
import cv2
#Loading pre-trained model & training layers
restNet = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt","res.caffemodel")
#load the input image
inputImage = cv2.imread("t_n_avatar.jpeg")
#Get the height and width of the input image
(height_input_image, weight_intput_image) = inputImage.shape[:2]
#Construct an input blob for the image by resizing
#a fixed 300x300 pixes and then normalizing it
blob = cv2.dnn.blobFromImage(cv2.resize(inputImage, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
#Pass the blob to the network 
restNet.setInput(blob)
#obtain a prediction / detection
#detections are in form of numpy array
detections = restNet.forward()
#The second index return the dimension of the detection inside the image
detections_dimensions = detections.shape[2]
#loop over detections
for i  in range(0, detections_dimensions):
    #extract each index 2 in the detections numpy array 
    #it's probability in the prediction 
    probability = detections[0,0,i,2]
    #Determine the threshold. If the probability is smaller then 0.5
    #Then filter out wrong detections
    if probability <0.5: 
    #if the probability meets this threshold => draw the rectangle 
    #along with probability of detection
        continue
    #Compute the (x,y) coordinate of bounding box
    bounding_box = detections[0,0,i,3:7] * np.array([weight_intput_image,height_input_image,weight_intput_image,height_input_image])
    #Coordinates of the bounding box 
    # X_bounding_box : the start of bounding box on x-axis
    # Y_bounding_box: the start of bounding box on y-axis
    # X_End_bounding_box: the end of bounding box on x-axis
    # Y_End_bounding_box: the end of bounding box on y-axis 
    (X_bounding_box, Y_bounding_box, X_End_bounding_box, Y_End_bounding_box) = bounding_box.astype("int")
    #drawing the box with the associate probability 
    #calculate the probablity 
    probability_detection = "{:.2f}%".format(probability * 100)
    #In case we go off the image probability detection
    #shift it down by 5 px in y-coordinates
    y_coordinate = bounding_box.astype("int")[1] - 5 if bounding_box.astype("int")[1] - 5 > 5 else bounding_box.astype("int")[1] +5
    #drawing a rectangular among the detected face 
    cv2.rectangle(inputImage, (X_bounding_box, Y_bounding_box), (X_End_bounding_box, Y_End_bounding_box), (0,0,255),2)
    cv2.putText(inputImage, probability_detection, (X_bounding_box, y_coordinate),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
# show the output image
cv2.imshow("Output", inputImage)
#write the result image to output
cv2.imwrite("/Users/cuongphan/Desktop/Spring 2020/COSC4332/Autonomous-Drone-code/tn-face-restnet-recognized.jpg",inputImage)
cv2.waitKey(0)



