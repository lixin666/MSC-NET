# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import sys
sys.path.insert(0,'../python/')
import caffe
import cv2 as cv
import numpy as np
import os

from xml.dom import minidom
from random import shuffle
from threading import Thread
import Image
import matplotlib.pylab as plt
params = dict(batch_size=1, im_shape=[473,473])
def readFiles(filename):
    files=open(filename)
    filelist=[]
    for line in files:
        split_str=line.split(' ')
        filelist.append(split_str)
    return filelist
    
def load_next_batch(root,filelist,index):
    img_batch=Image.open(root+filelist[index][0])
    img_batch=np.array(img_batch,dtype=np.float32)
    img_batch=cv.resize(img_batch,(params['im_shape'][0], params['im_shape'][1]))
    label_batch=Image.open(root+filelist[index][1].strip())
    label_batch=np.array(label_batch,dtype=np.float32)
    label_batch=cv.resize(label_batch,(params['im_shape'][0], params['im_shape'][1]),interpolation=cv.INTER_NEAREST)
    label_batch=label_batch/255
    return img_batch, label_batch
    
class SaliencyDataLayer(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
#        params = eval(self.param_str)

        # store input as class variables
        self.batch_size = params['batch_size']
        self.root='dataset/'
        self.source='train.txt'
        self.filelist=readFiles(self.root+self.source)
        self.iter=0

        # Create a batch loader to load the images.
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        # Note the 20 channels (because PASCAL has 20 classes.)
        top[1].reshape(self.batch_size, 1 ,params['im_shape'][0], params['im_shape'][1])
        

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            if self.iter>len(self.filelist)-1:
            	self.iter=0
            im, label = load_next_batch(self.root,self.filelist,self.iter)
            im=im[:,:,::-1]
            im-=np.array([104.00698793, 116.66876762, 122.67891434])
            im=im.transpose(2,0,1)
            label=np.array([label])
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label
            self.iter+=1

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass
