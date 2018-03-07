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


# Third-party libraries------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmn


# prvaite libraries---------------------------------------------
import mnist_loader
import RvNeuralNetworks as rn
from RvNeuralNetworks import *
import RvAskInput as ri
import RvMiscFunctions as rf
import RvNeuNetworkMethods as nm
import RvFileIO as rfi
import RvMediaUtility as ru
import RvFileIO as rfi

from RvNeuNetworkMethods import EnumDropOutMethod as drpOut

#%%
AddNoise = False

#%%%  Function Section








#%%% Test Section






#%%% Main Section


#Load MNIST ****************************************************************

#Use mnist.pkl.gz(50000 data） Accuracy 0.96 
#Use mnist_expanded.pkl.gz(250000 data） Accuracy 0.97 
fn = "..\\data\\mnist.pkl.gz"  #".datamnist_expanded.pkl.gz"
lstTrain, lstV, lstT =  mnist_loader.load_data_wrapper(fn)
lstTrain = list(lstTrain)
lstV = list(lstV)
lstT = list(lstT)

# Prediction ------------------------------------------------
fn1= ".RvNeuralEnDeCoder_Decoder.decoder"


fns, shortFns =  rfi.Get_FilesInFolder(".\\NetData\\", [".decoder"])
aId = 0 #ri.Ask_SelectItem("Select Decoder file", shortFns, 0)    
fn1= fns[aId]

fns, shortFns =  rfi.Get_FilesInFolder(".\\NetData\\", [".encoder"])
aId = 0 #ri.Ask_SelectItem("Select Encoder file", shortFns, 0)    
fn2= fns[aId]

#addNoise = ri.Ask_YesNo("Add noise?", "n")
noiseStrength = ri.Ask_Input_Float("Input Noise Strength.", 0.0)

randomState = np.random.RandomState(int(time.time()))

if (os.path.isfile(fn1)):         
#    inputX = randomState.randn(784,1)
#    net = rn.RvNeuralNetwork.Create_Network(fn1)    
#    label, result = net.Predict_Digit([inputX,1], False)   
    
    
    decoder = rn.RvNeuralEnDeCoder.Create_Network(fn1)
    encoder = rn.RvNeuralEnDeCoder.Create_Network(fn2)
    
    imgPath = "{}\\{}".format(decoder.VideoImagePath, "EncoderDecoder")   
    if not os.path.isdir(imgPath): os.mkdir(imgPath)
    rfi.Delete_Files(imgPath, [".jpg",".png"])
        
    sampleNum=50 # 不含繪圖，辨識 10000張，費時 1.3 秒，平均每張 0.00013秒
    if (None!=decoder) and (None!=encoder):            
      
        rf.Test_Encoder_Decoder(encoder, decoder, lstT, 5, imgPath, noiseStrength)

        aviFn = "{}{}".format(imgPath, "EncoderDecoder.avi")
        
        if ru.ImageFilesToAvi(imgPath, aviFn ):
            os.system(r'start ' + aviFn)





