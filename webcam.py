from flask import send_file, Flask, current_app, request, jsonify, send_from_directory
import io
import os
import sys
import json
import base64
import logging
import matplotlib
matplotlib.use('Agg')
import subprocess
import numpy as np
import glob
import requests
import urllib.request
from timeit import default_timer as timer
from multiprocessing import Pool
from keras.models import load_model
from werkzeug.datastructures import MultiDict
import tensorflow as tf
import cv2
from PIL import Image, ImageFont, ImageDraw
from waitress import serve

#yolo object detection
from yolo_webapp import YOLO, detect_video
#moments in time action classifier
from test_TRN import classify_actions, Runner

#load up models

trnModel = Runner()
yoloModel = YOLO(tf)

#streamVideo = 'processed_videos/video_stream.ogv'

#if os.path.isfile(streamVideo):
#    os.remove(streamVideo)

#cap = cv2.VideoCapture('http://192.168.1.132:8080/video')
#if arg present, read from there, else read from computer's webcam
videoInput = 0
if len(sys.argv) == 2:
    videoInput = sys.argv[1]

cap = cv2.VideoCapture(videoInput)

def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

make_1080p()
change_res(1920, 1080)

curr_frame = 0
frames = []
predictions = []

#width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')

#out = cv2.VideoWriter('out_vid.mp4', fourcc, 30.0, (int(width),int(height)))

cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
while(cv2.getWindowProperty('result', 0) >= 0):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    image = Image.fromarray(frame)
    frames.append(image)
    image, out_classes = yoloModel.detect_image(image)
    result = np.asarray(image)
    curr_frame += 1

    if len(frames) == 48:
        predictions.clear()
        probs, preds = trnModel.test_video(frames, str(videoInput))
        for index in range(0, 5):
            predictions.append(str(preds[index]) + ' ' + str("%.2f" % probs[index]))
        frames = frames[16:48]

    for i, prediction in enumerate(predictions):
        cv2.putText(result, text=prediction, org=(3, 30 + i * 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=3)
    
    cv2.imshow("result", result)

    #out.write(result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



