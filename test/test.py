import io
import os
import sys, getopt
sys.path.insert(0,'../python/')
import numpy as np
import cv2 as cv
from scipy.io import savemat
import caffe
import matplotlib.pylab as plt
from preprocess_image import *
import datetime
gpu_ID=0
caffe_model='models/MSCNet.caffemodel'
deploy_path='models/res_saliency_rec_deploy.prototxt'
caffe.set_mode_gpu()
caffe.set_device(gpu_ID)
net = caffe.Net(deploy_path,caffe_model,caffe.TEST)

dataname='images'
root_dir=dataname+'/'
filelist=os.listdir(root_dir)
begin = datetime.datetime.now()
for fn in filelist:
  imname=fn
  I=cv.imread(root_dir+imname)
  I=cv.cvtColor(I, cv.COLOR_BGR2RGB)
  I=np.array(I.copy(),dtype = np.float32)
  or_sz=I.shape
  caffe_im = preprocess_image(I, 473)

  net.blobs['data'].reshape(1, *caffe_im.shape)
  net.blobs['data'].data[...] = caffe_im 
  scores=net.forward()
  seg_score=scores['side55_score']
  sal=np.squeeze(seg_score)
  sal=cv.resize(sal,(or_sz[1],or_sz[0]))
  cv.imwrite('results/'+imname[:-4]+'.png',255*sal)
  print '-----------------------------'
  print imname+' is done!'
end = datetime.datetime.now()
print (end-begin)/len(filelist)
