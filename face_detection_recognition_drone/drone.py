import sys
import traceback
import tellopy
import av
import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy
import time
import os
import datetime
import imutils
import numpy as np
import argparse
import face_recognition


ap = argparse.ArgumentParser()
#If you would like to save the video, pass the argument s in the command line when running the program
ap.add_argument("-s", "--save",  action='store_true',
    help="save the video")
args = vars(ap.parse_args())

def saveFileReceived(event, sender, data):
    global date_fmt
    # Create a file in ~/Pictures/ to receive image data from the drone.
    path = '%s/tello-%s.jpeg' % (
        os.getenv('HOMEPATH'),                              #Changed from Home to Homepath
        datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    with open(path, 'wb') as fd:
        fd.write(data)
    #print('Saved photo to ',path)

if args["save"]:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (400,300))
# Call dnn in ResNet with pre-trained dataset and architecture
resNet = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

def main():
    myTelloDrone = tellopy.Tello()
    landed = True
    speed = 30
    up,down,left,right,forw,back,clock,ctclock = False,False,False,False,False,False,False,False
    ai = True
    pic360 = False
    currentPic = 0
    move360 = False

    # add image of person to follow to library
    target_picture = face_recognition.load_image_file("target.jpg")
    target_face_encoding = face_recognition.face_encodings(target_)[0]

    try:
      #We first try to connect the drone 
        myTelloDrone.connect()
        myTelloDrone.wait_for_connection(60.0)
			#After connected, get the video stream into a stream_video_frome_drone to process
        stream_video_from_drone = av.open(myTelloDrone.get_video_stream())
        myTelloDrone.subscribe(myTelloDrone.EVENT_FILE_RECEIVED, saveFileReceived)
        # skip first 200 frames
        frame_skip = 200
        while True:
            try:
                for frame in stream_video_from_drone.decode(video=0): 
                    if 0 < frame_skip:
                        frame_skip = frame_skip - 1
                        continue
                    start_time = time.time() #set start time = current time
                    image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR) #image from frame

                    image = imutils.resize(image, width=400) #resize image width=400px

                    #The shape of an image is accessed by img.shape. It returns a tuple of the number of rows, columns, and channels (if the image is color)
                    #If an image is grayscale, the tuple returned contains only the number of rows and columns
                    #Depends on the numbers of rows and columns we know what shape the picture is in
                    (h, w) = image.shape[:2] 
                    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                    (300, 300), (104.0, 177.0, 123.0))
                    resNet.setInput(blob)
                    detections = resNet.forward()

                    face_dict = {}

                    for i in range(0, detections.shape[2]):
                        # extract the confidence (i.e., probability) associated with the
                        # prediction
                        confidence = detections[0, 0, i, 2]

                        # filter out weak detections by ensuring the `confidence` is
                        # greater than the minimum confidence
                        if confidence < 0.5:
                            continue

                        # compute the (x, y)-coordinates of the bounding box for the
                        # object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # draw the bounding box of the face along with the associated
                        # probability
                        text = "{:.2f}%".format(confidence * 100)
                        face_dict[text]=box

                    # Will go to face with the highest confidence
                    try:    
                        cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR) #image from frame

                        image = imutils.resize(image, width=400) #resize image width=400px
                    
                    #The shape of an image is accessed by img.shape. It returns a tuple of the number of rows, columns, and channels (if the image is color)
                    #If an image is grayscale, the tuple returned contains only the number of rows and columns
                    #Depends on the numbers of rows and columns we know what shape the picture is in
                        (h, w) = image.shape[:2] 
                        H,W,_ = image.shape
                        distTolerance = 0.05 * np.linalg.norm(np.array((0, 0))- np.array((w, h)))

                        box = face_dict[sorted(face_dict.keys())[0]]
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        
                        
                        image_encoding = face_recognition.face_encodings(image)[0]
                        compare = face_recognition.compare_faces([target_face_encoding], image_encoding)
                        if compare[0] == True: 
                            image = cv2.rectangle(image, (startX, startY), (endX, endY),
                            (0, 0, 255), 2)                      
                            cv2.putText(image, 'TARGET', (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0,0,225), 3)

                        distance = np.linalg.norm(np.array((startX,startY))-np.array((endX,endY)))

                        if int((startX+endX)/2) < W/2-distTolerance :
                            #print('CounterClock')
                            myTelloDrone.counter_clockwise(30)
                            ctclock = True
                        elif int((startX+endX)/2) > W/2+distTolerance:
                            #print('Clock') 
                            myTelloDrone.clockwise(30)
                            clock = True
                        else:
                            if ctclock:
                                myTelloDrone.counter_clockwise(0)
                                ctclock = False
                                #print('CTClock 0')
                            if clock:
                                myTelloDrone.clockwise(0)
                                clock = False
                                #print('Clock 0')
                        
                        if int((startY+endY)/2) < H/2-distTolerance :
                            myTelloDrone.up(30)
                            #print('Up')
                            up = True
                        elif int((startY+endY)/2) > H/2+distTolerance :
                            myTelloDrone.down(30)
                            #print('Down')
                            down = True
                        else:
                            if up:
                                up = False
                                #print('Up 0')
                                myTelloDrone.up(0)

                            if down:
                                down = False
                                #print('Down 0')
                                myTelloDrone.down(0)

                        #print(int(distance))

                        if int(distance) < 110-distTolerance  :
                            forw = True
                            #print('Forward')
                            myTelloDrone.forward(30)
                        elif int(distance) > 110+distTolerance :
                            myTelloDrone.backward(30)
                            #print('Backward')
                            back = True
                        else :
                            if back:
                                back = False
                                #print('Backward 0')
                                myTelloDrone.backward(0)
                            if forw:
                                forw = False
                                #print('Forward 0')
                                myTelloDrone.forward(0)
                            

                    except Exception as e:
                        #print(e)
                        None

                    if args["save"]:
                            out.write(image)

                    cv2.imshow('Original', image)

                    #cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                    if frame.time_base < 1.0/60:
                        time_base = 1.0/60
                    else:
                        time_base = frame.time_base
                    frame_skip = int((time.time() - start_time)/time_base)
                    keycode = cv2.waitKey(1)
                    #Enter "T" for the drone to take off
                    if keycode == 84 :
                        if landed:
                            myTelloDrone.takeoff()
                            myTelloDrone = False
                        else:
                            myTelloDrone.land()
                            landed = True
										# If the ESCAPE is enterred, the drone stops
                    if keycode == 27 :
                        raise Exception('Quit')
										# Enter "D" for the drone to take picture
                    if keycode == 13 :
                        myTelloDrone.take_picture()
                        time.sleep(0.25)
                        #pic360 = True
                        #move360 = True

                    if keycode & 0xFF == ord('q') :
                        pic360 = False
                        move360 = False 

            except Exception as e:
                print(e)
                break
                     

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        if args["save"]:
            out.release()
        myTelloDrone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

