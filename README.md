## Multi-Scale Cascade Network for Salient Object Detection

by Xin Li, Fan Yang, Hong Cheng, Junyu Chen, Yuxiao Guo, Leiting Chen

### Introduction

This repository is for '[Multi-Scale Cascade Network for Salient Object Detection](http://delivery.acm.org/10.1145/3130000/3123290/p439-li.pdf?ip=121.49.77.113&id=3123290&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E21AB2B2297141EDA%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=840107087&CFTOKEN=16900689&__acm__=1513169790_7a38b564badf04f0c17e85a3ad61c9f4#URLTOKEN#)'.
### Installation

For installation, please follow the instructions of [Caffe](https://github.com/BVLC/caffe) and [hszhao](https://github.com/hszhao/PSPNet).

The code has been tested successfully on Ubuntu 14.04 with CUDA 8.0.

### Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/lixin666/MSC-NET.git
   ```

2. Build Caffe and pycaffe:

   ```shell
   cd $ROOT
   cp Makefile.config.example Makefile.config
   vim Makefile.config
   make -j8 && make pycaffe
   ```
ps: You should uncomment 'WITH_PYTHON_LAYER := 1' in Makefile.config before compiling.


3. Test:

   - Test code is in folder 'test'.
   - Download trained models and put them in folder 'test/models':
     - MSCNet.caffemodel: [BaiduYun](https://pan.baidu.com/s/1eSfaDto) or [GoogleDrive](https://drive.google.com/open?id=1wb71oU49G3gyor7vF1qDgPq0ePCFYHKG)
   - Put the test images in folder 'images', and run
   
   ```shell
   python test.py
   ```
   -After that, the results will be genrated in folder 'results'.
### Citation
If MSCNet is useful for your research, please consider citing:

    @inproceedings{li2017mscnet,
      author = {Xin Li and Fan Yang and Hong Cheng 
                and Junyu Chen and Leiting Chen},
      title = {Multi-Scale Cascade Network for Salient Object Detection},
      booktitle = {ACM MM},
      year = {2017}
    }

### Question
Please contact 'fanyang_uestc@hotmail.com'
