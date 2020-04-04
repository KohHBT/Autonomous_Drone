#Import libraries 
import sys
import traceback
import tellopy #Tello manufacturer library to help control through terminal
import av # to get the live stream video from drone
import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy
import time
import os
import datetime
import imutils #to resize each frame of videos sent from drone
import numpy as np
import argparse
# Call dnn in ResNet with pre-trained dataset and architecture
resNet = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt","res.caffemodel")
# This function save video recorded from drone to computer
def SaveVideFromDrone(event, sender, data):
    global date_fmt
    # Create a file in ~/Pictures/ to receive image data from the drone.
    path = '%s/tello-%s.jpeg' % (os.getenv('HOMEPATH'),#Changed from Home to Homepath
    datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    with open(path, 'wb') as fd:
        fd.write(data)
    #print('Saved photo to ',path)
drone = tellopy.Tello()
#Save videos from drone
video = cv2.VideoWriter_fourcc(*'XVID')
#output vide is named output.avi 400x300 dimensions 
out = cv2.VideoWriter('output.avi',video,20.0,(400,300))
# looking for tello drone connection
try:
    #try if to connect to the drone 
    drone.connect()
    drone.wait_for_connection(60.0) 
    #get livestream video from the drone 
    livestream_video = av.open(drone.get_video_stream())
    #Call the function save videos from drone
    up, down, takeoff = False , False, False #control up and down movement for the drone
    forward, backward = False, False #control forward and backward movement of the drone
    drone.subscribe(drone.EVEN_FILE_RECEIVED, SaveVideFromDrone)
    #use a loop to go through each frame (picture) from drone 
    # video is just a really fast loop of picture
    # skip the first 200 frames 
    skip_frame_value = 200
    while True:
        try:
            for each_frame in livestream_video.decode(video=0):
                if 0 < skip_frame_value:
                    #update skip frame value 
                    skip_frame_value = skip_frame_value -1 
                    continue
                #take off the drone 
                drone.takeoff()
                takeoff = True
                #get the time the drone is on
                start_time_drone_on = time.time()
                input_img_per_frame = cv2.cvtColor(numpy.array(each_frame()),cv2.COLOR_RGB2BGR)
                #resize the image per frame
                input_img_per_frame = imutils.reize(input_img_per_frame, width=400)
                (heigh_img_per_frame, width_img_per_frame) = input_img_per_frame.shape[:2]
                #Construct an input blob for the image by resizing
                #a fixed 300x300 pixes and then normalizing it
                blob = cv2.dnn.blobFromImage(cv2.resize(input_img_per_frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
                #Pass the input blob to neural network model
                resNet.setInput(blob)
                #obtain a prediction/ detection 
                detections = resNet.forward()
                face_dict = {}
                #loop through 
                for i in range(0, detections.shape[2]):
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
                    bounding_box = detections[0,0,i,3:7] * np.array([width_img_per_frame,heigh_img_per_frame,width_img_per_frame,heigh_img_per_frame])
                    #Coordinates of the bounding box 
                    # X_bounding_box : the start of bounding box on x-axis
                    # Y_bounding_box: the start of bounding box on y-axis
                    # X_End_bounding_box: the end of bounding box on x-axis
                    # Y_End_bounding_box: the end of bounding box on y-axis 
                    (X_bounding_box, Y_bounding_box, X_End_bounding_box, Y_End_bounding_box) = bounding_box.astype("int")
                    #drawing the box with the associate probability 
                    #calculate the probablity 
                    probability_detection = "{:.2f}%".format(probability * 100)
                    face_dict[probability_detection] = bounding_box
                    # go with the face with the highest confidence
                    try: 
                        distTolerance = 0.05 * np.linalg.norm(np.array((0, 0))- np.array((width_img_per_frame, heigh_img_per_frame)))
                        bounding_box = face_dict[sorted(face_dict.keys())[0]]
                        #In case we go off the image/frame probability detection
                        #shift it down by 5 px in y-coordinates
                        y_coordinate = bounding_box.astype("int")[1] - 5 if bounding_box.astype("int")[1] - 5 > 5 else bounding_box.astype("int")[1] +5
                         #drawing a rectangle among the detected face area
                        cv2.rectangle(input_img_per_frame, (X_bounding_box, Y_bounding_box), (X_End_bounding_box, Y_End_bounding_box), (0,0,255),2)
                        #distance to the drawn rectangle
                        distance = np.linalg.norm(np.array((X_bounding_box,Y_bounding_box))-np.array((X_End_bounding_box,Y_End_bounding_box)))
                        #Make sure the drone keep a distance from the face
                        #move the drone up and down in y-coordinates
                        if int((Y_bounding_box+Y_End_bounding_box)/2) < heigh_img_per_frame - distTolerance:
                            drone.up(30)
                            up = True 
                        elif int((Y_bounding_box + Y_End_bounding_box)/2) > heigh_img_per_frame/2 + distTolerance:
                            drone.down(30)
                            down = True 
                        # update up and down again 
                        #to prevent the drone from keep moving up / down 
                        else: 
                            if up: 
                                up = False
                                #Stop moving up 
                                drone.up(0)
                            if down:
                                down = False 
                                #Stop moving down
                                drone.down(0)
                        #Keep a distance from human face by moving forward and backward
                        if int(distance) < 120 - distTolerance:
                            forward = True 
                            #send command forward to drone 
                            drone.forward(30)
                        elif int(distance) > 120 + distTolerance:
                            drone.backward(30)
                            backward = True 
                        #Stop the drone from moving forward/ backward right away after a move
                        else: 
                            #if moving backward => stop backward
                            if backward:
                                backward = False 
                                drone.backward(0)
                            #if moving forward => stop forward
                            if forward:
                                forward = False 
                                drone.forward(0)
                    except Exception as e:
                        print(e) 
                        break 
            #show livestream video on computer
            cv2.imshow("Video", input_img_per_frame)
            #caluclate skipping frame in each videos
            if each_frame.time_base < 1.0/60:
                time_base = 1.0/60
            else:
                time_base = each_frame.time_base
            skip_frame_value = int((time.time()-start_time_drone_on)/time_base)
                # wait for user to press a key
            keycode = cv2.waitKey(1)   
            # if key q is pressed => stop land the drone   
            if keycode == ord('q'):
                drone.land()
                takeoff = False
                #raise an exception to trigger off save video function
                raise Exception('Quit')
        except Exception as ex:
            print(ex)
#trigger save video in final exception
finally: 
    out.release()
    drone.quit()
    cv2.destroyAllWindows()