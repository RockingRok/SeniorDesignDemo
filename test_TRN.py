import numpy as np
import sys,os
import time
import os
import re
import cv2
import argparse
import functools
import subprocess
import numpy as np
from PIL import Image
import moviepy.editor as mpy

import torch.nn.parallel
import torch.optim
from models import TSN
from transforms import *
import datasets_video
from torch.nn import functional as F
import shutil

class Runner(object):
	def __init__(self):
		categories_file ='pretrain/reduced_categories.txt'
		self.categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
		self.num_class = len(self.categories)
		#self.arch = 'InceptionV3'
		self.arch = 'BNInception'
		# Load model.
		self.net = TSN(self.num_class, 8, 'RGB', base_model=self.arch, consensus_type='TRNmultiscale', img_feature_dim=256, print_spec=False)

		#weights = 'pretrain/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth.tar'
		weights = 'pretrain/seniordesign.pth.tar'
		checkpoint = torch.load(weights,map_location='cpu')
		#print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
                # print list(checkpoint['state_dict'].items())

		base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
		self.net.load_state_dict(base_dict)
		#self.net.eval() #.cuda().eval()
		self.net.cuda().eval()
		# Initialize frame transforms.

		# self.transform = torchvision.transforms.Compose([
		#     GroupOverSample(self.net.input_size, self.net.scale_size),
		#     Stack(roll=(self.arch in ['BNInception', 'InceptionV3'])),
		#     ToTorchFormatTensor(div=(self.arch not in ['BNInception', 'InceptionV3'])),
		#     GroupNormalize(self.net.input_mean, self.net.input_std),
		# ])

		self.transform = torchvision.transforms.Compose([
		    GroupScale(self.net.scale_size),
                    GroupCenterCrop(self.net.input_size),
		    Stack(roll=(self.arch in ['BNInception', 'InceptionV3'])),
		    ToTorchFormatTensor(div=(self.arch not in ['BNInception', 'InceptionV3'])),
		    GroupNormalize(self.net.input_mean, self.net.input_std),
		])





	def test_video(self,frames,videoname):
		data = self.transform(frames)
		input_var = torch.autograd.Variable(data.view(-1, 3, data.size(1), data.size(2)),
                                    volatile=True).unsqueeze(0).cuda()
		logits = self.net(input_var)
		h_x = torch.mean(F.softmax(logits, 1), dim=0).data
		probs, idx = h_x.sort(0, True)
		preds = {}
		actualProbs = {}

		# Output the prediction.
		# video_name = args.frame_folder if args.frame_folder is not None else args.video_file
		print('RESULT ON ' + videoname)
		for i in range(0, 5):
                        preds[i] = self.categories[idx[i]]
                        print('{:.3f} -> {}'.format(probs[i], self.categories[idx[i]]))
                        #print(probs[i].data.tolist())
                        #actualProbs[i] = probs[i] #with cuda
                        actualProbs[i] = probs[i].data.tolist() #without cuda
                        
		return actualProbs, preds

def my_extract_frames(video_file, num_frames=8):
    try:
        os.makedirs(os.path.join(os.getcwd(), 'frames'))
    except OSError:
        print("oh no it could not create frames/ folder, it may already exist\n")
        pass

    output = subprocess.Popen(['ffmpeg', '-i', video_file],
                              stderr=subprocess.PIPE).communicate()
    # Search and parse 'Duration: 00:05:24.13,' from ffmpeg stderr.
    re_duration = re.compile('Duration: (.*?)\.')
    duration = re_duration.search(str(output[1])).groups()[0]

    seconds = functools.reduce(lambda x, y: x * 60 + y,
                               map(int, duration.split(':')))
    rate = num_frames / float(seconds)

    output = subprocess.Popen(['ffmpeg', '-i', video_file,
                               '-vf', 'fps={}'.format(rate),
                               '-vframes', str(num_frames),
                               '-loglevel', 'panic',
                               'frames/%d.jpg']).communicate()
    frame_paths = sorted([os.path.join('frames', frame)
                          for frame in os.listdir('frames')])

    frames = my_load_frames(frame_paths, num_frames)
    #subprocess.call(['rmdir', '/s', './frames'], shell=True)
    shutil.rmtree('./frames')
    return frames


def my_load_frames(frame_paths, num_frames=8):
    frames = [Image.open(frame).convert('RGB') for frame in frame_paths]
    # print len(frames)
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise ValueError('Video must have at least {} frames'.format(num_frames))


def my_render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames

#will extract frames, classify actions on those frames and return a video
#with classifications overlayed
def classify_actions(model, input_video, output_video, frame_step):
        #extract frames from video
        vid = cv2.VideoCapture(input_video)
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_framecount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        classified_frame_count = int(video_framecount/2)
        
        frames = my_extract_frames(input_video, classified_frame_count)

        #next, run frame subsets through classifier
        rendered_frames_list = []
        finished = False
        for index in range(0, classified_frame_count, frame_step):
                end_index = 0
                if(classified_frame_count - index >= frame_step):
                        end_index = index + frame_step
                else:
                        #rendered_frames_list.append(frames[index:classified_frame_count])
                        break
                print(str(index) + ' ' + str(frame_step) + ' ' + str(end_index) + ' ' + str(classified_frame_count))
                frame_subset = frames[index:end_index]
                probs, preds = model.test_video(frame_subset, input_video)
                rendered_frames = my_render_frames(frame_subset, preds[0])
                rendered_frames_list.append(rendered_frames)

        final_frames = [item for sublist in rendered_frames_list for item in sublist]
        clip = mpy.ImageSequenceClip(final_frames, int(video_fps/2))
        clip.write_videofile(output_video)
