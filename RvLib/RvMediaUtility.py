# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

/* -----------------------------------------------------------------------------
  Copyright: (C) Daniel Lu, RasVector Technology.

  Email : dan59314@gmail.com
  Web :     http://www.rasvector.url.tw/
  YouTube : http://www.youtube.com/dan59314/playlist
  GitHub : https://github.com/dan59314
  LinkedIn : https://www.linkedin.com/in/daniel-lu-238910a4/

  This software may be freely copied, modified, and redistributed
  provided that this copyright notice is preserved on all copies.
  The intellectual property rights of the algorithms used reside
  with the Daniel Lu, RasVector Technology.

  You may not distribute this software, in whole or in part, as
  part of any commercial product without the express consent of
  the author.

  There is no warranty or other guarantee of fitness of this
  software for any purpose. It is provided solely "as is".

  ---------------------------------------------------------------------------------
  版權宣告  (C) Daniel Lu, RasVector Technology.

  Email : dan59314@gmail.com
  Web :     http://www.rasvector.url.tw/
  YouTube : http://www.youtube.com/dan59314/playlist

  使用或修改軟體，請註明引用出處資訊如上。未經過作者明示同意，禁止使用在商業用途。
*/


"""
#%%%
# Standard library----------------------------------------------
import sys
sys.path.append('../data')
sys.path.append('../RvLib')
import os
import time
from datetime import datetime

import cv2
from PIL import Image, ImageDraw, ImageFont

# Third-party libraries------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmn

import struct

# prvaite libraries---------------------------------------------
"""
import mnist_loader
import RvNeuralNetworks as rn
from RvNeuralNetworks import *
import RvAskInput as ri
import RvMiscFunctions as rf
import RvNeuNetworkMethods as nm
from RvNeuNetworkMethods import EnumDropOutMethod as drpOut
"""




#%%%  Function Section
def CvBGR_To_RGB(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def RGB_To_CvBGR(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)


#def saveAsPNG(_2dArray, filename):
#    if any([len(row) != len(_2dArray[0]) for row in _2dArray]): 
#        raise ValueError, "_2dArray should have elements of equal size"
#
#                                #First row becomes top row of image.
#    flat = []; map(flat.extend, reversed(_2dArray))
#                                 #Big-endian, unsigned 32-byte integer.
#    buf = b''.join([struct.pack('>I', ((0xffFFff & i32)<<8)|(i32>>24) )
#                    for i32 in flat])   #Rotate from ARGB to RGBA.
#
#    data = write_png(buf, len(_2dArray[0]), len(_2dArray))
#    f = open(filename, 'wb')
#    f.write(data)
#    f.close()
    
    

def Blend_TwoImages(image1,image2, ratio=0.5):
    # Load up the first and second demo images
#    image1 = Image.open("demo3_1.jpg")
#    image2 = Image.open("demo3_2.jpg")
    if (None==image1) or (None==image2): return
    
    # Create a new image which is the half-way blend of image1 and image2
    # The "0.5" parameter denotes the half-way point of the blend function.
    images1And2 = Image.blend(image1, image2, ratio)
    
    # Save the resulting blend as a file
#    images1And2.save("demo3_3.jpg")
    return images1And2


def ImageFilesToAvi(path, aviFn, durationSec=0.5, ratio=0.5):
    # Load up the first and second demo images, assumed is that image1 and image2 both share the same height and width
#    image1 = Image.open("demo3_1.jpg")
#    image2 = Image.open("demo3_2.jpg")
    
    path = os.path.abspath(path)
    fns = []
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".png"):
            fn = os.path.join(path, file)
#            print(fn)
            fns.append(fn)
            
    if len(fns)<=1: return False
    
    sId = -1
    for fn1 in fns:
        sId += 1
        image1 = Image.open(fn1)
        if (None!=image1): break
    if (sId<0):return False
    
    # Grab the stats from image1 to use for the resultant video
    height, width, layers =  np.array(image1).shape
    
    """">>> os.path.dirname(os.path.abspath(existGDBPath))
    'T:\Data\DBDesign'
    """
    aviDir =  os.path.dirname(os.path.abspath(aviFn)) 
    if not os.path.isdir(aviDir): os.mkdir(aviDir)
    aviFn = os.path.splitext(aviFn)[0]+".avi"
    
    framePerSec = 10
    frameBetweenTwoImages = int(framePerSec*durationSec)
    
    # Create the OpenCV VideoWriter
    video = cv2.VideoWriter(
        aviFn, # Filename
        1, # Negative 1 denotes manual codec selection. You can make this automatic by defining the "fourcc codec" with "cv2.VideoWriter_fourcc"
        framePerSec, # 10 frames per second is chosen as a demo, 30FPS and 60FPS is more typical for a YouTube video
        (width,height) # The width and height come from the stats of image1
        )
    
    # 下一張和前一張合併過度----------------------------------------
    for fn in (fns[sId+1:]):
        image2 = Image.open(fn)
        if (None!=image2): 
            for i in range(0,frameBetweenTwoImages):
                images1And2 = Image.blend(image1, image2, i/frameBetweenTwoImages)
                # Conversion from PIL to OpenCV from: http://blog.extramaster.net/2015/07/python-converting-from-pil-to-opencv-2.html
                video.write(cv2.cvtColor(np.array(images1And2), cv2.COLOR_RGB2BGR))
            image1 = image2    
      
    print("Video saved as \"{}\"".format(aviFn))
    # Release the video for it to be committed to a file
    
    video.release()
    
    return True
    

#%%% Test Section






#%%% Main Section





