#Import libraries 
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
# Parse command line to see if user need to save the drone videos
commandline_arguments = argparse.ArgumentParser().add_argument("-s", "--save",  action='store_true',help="save the video")
args = vars(commandline_arguments.parse_args())
# Call dnn in ResNet with pre-trained dataset and architecture
