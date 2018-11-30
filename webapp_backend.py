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
from waitress import serve

#import yolo and moments
#change this path to wherever you have this
#sys.path.append('../SenDesignModels/keras-yolo3-master')
#sys.path.append('../SenDesignModels/TRN-pytorch')

#yolo object detection
from yolo_webapp import YOLO, detect_video
#moments in time action classifier
from test_TRN import classify_actions, Runner

app = Flask(__name__)

#load up models

trnModel = Runner()
yoloModel = YOLO(tf)

@app.route('/', methods=['POST', 'GET'])
def pred():
    global yoloModel
    global trnModel

    data = ""
    key = ""
    found = False

    try:
        video = request.get_json()['video']
        key = request.get_json()['key']
    except Exception:
        return jsonify(status_code='400', msg='Bad Request'), 400

    print('MODEL REQUEST:')
    print('Video: ' + video)
    print('Video Key: ' + key)

    #this comes from local server using GAE  
    input_video = video
    #for split video, we will need to add on the rest outselves for amount of vids we generate
    video_with_actions = 'processed_videos/' + key + '_actions.mp4'
    output_video = 'processed_videos/' + key + '_output.mp4'
    
    elapsedTime = timer()
    yoloModel.reload()
    yoloModelReload = str(timer() - elapsedTime)

    elapsedTime = timer()
    classify_actions(trnModel, input_video, video_with_actions, 8)
    trnModelProcess = str(timer() - elapsedTime)

    elapsedTime = timer()
    size, fps, framecount = detect_video(yoloModel, video_with_actions, 1, output_video)
    yoloModelProcess = str(timer() - elapsedTime)

    print('METRICS:')
    print('Video Size: ' + str(size))
    print('Video FPS: ' + str(fps))
    print('Video Duration: ' + str(framecount/fps))
    print('Took ' + yoloModelReload + ' seconds to reload YOLO')
    print('Took ' + yoloModelProcess + ' seconds to process video with YOLO')
    print('Took ' + trnModelProcess + ' seconds to process video with TRN')

    print('Forming JSON')
    
    predictions = {}
    #predictions['predCount'] = len(mitResults)
    
    #for count in range(0, len(mitResults)):
    #    probs, preds = mitResults[count]
    #    predictions['pred' + str(count)] = preds
    #    predictions['prob' + str(count)] = probs

    predictions['output_video'] = output_video

    preds = predictions.copy()

    current_app.logger.info('Predictions: %s', preds)

    return jsonify(predictions=predictions)

@app.route('/processed_videos/<path:path>')
def getVideo(path):
    print('VIDEO REQUEST:')
    print('Path: ' + path)
    return send_from_directory('processed_videos', path)


#use this for demo/deployment
serve(app, host='0.0.0.0', port=5000)

#use this instead of this ^ if debugging/testing
#if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000, threaded=True)
    #app.run(host='127.0.0.1', port=5000, debug=True)
