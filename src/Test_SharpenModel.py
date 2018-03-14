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

def PreProcessBlur(inputX):
    pxls = len(inputX)
    pxlW = int(np.sqrt(pxls))
    pxls = pxlW*pxlW
    nInput = np.array(inputX).reshape(pxlW,pxlW)
    for iy in range(1,pxlW-1):
        for ix in range(1, pxlW-1):
            tVal = 0.0
            for ay in range(iy-1,iy+2):
                for ax in range(ix-1, ix+2):
                    tVal+=nInput[ay][ax]
            avg = tVal/9
            nInput[iy][ix] = avg
    return nInput.reshape(pxls,1)



#%%% Test Section






#%%% Main Section


#Load MNIST ****************************************************************

#Use mnist.pkl.gz(50000 data） Accuracy 0.96 
#Use mnist_expanded.pkl.gz(250000 data） Accuracy 0.97 
#fn = "..\\data\\mnist.pkl.gz"  #".datamnist_expanded.pkl.gz"
#lstTrain, lstV, lstT =  mnist_loader.load_data_wrapper(fn)
#lstTrain = list(lstTrain)
#lstV = list(lstV)
#lstT = list(lstT)

# Load digit Images  lstTrain[0].shape = (pxls, label) = (784, 1)
imgPxls = 28*28
digitImages = rn.RvNeuralEnDeCoder.Load_DigitImages( ".\\Images\\Blur\\", imgPxls)

lstTest = []
id=0
for img in digitImages:
    # [模糊, label]
    lstTest.append(tuple([ img, rn.RvBaseNeuralNetwork.CreateLabelsY(10,id) ]) ) 
    id+=1
    
    
# Prediction ------------------------------------------------
fn1= "RvNeuralEnDeCoder_Sharpness_DontDelete.endecoder"

if not rfi.FileExists(fn1):
    fns, shortFns =  rfi.Get_FilesInFolder(".\\NetData\\", [".endecoder"])
    aId = min(1, len(fns)-1) #0 #ri.Ask_SelectItem("Select Decoder file", shortFns, 0)    
    fn1= fns[aId]


#addNoise = ri.Ask_YesNo("Add noise?", "n")
noiseStrength = 0.5 #ri.Ask_Input_Float("Input Noise Strength.", 0.0)

randomState = np.random.RandomState(int(time.time()))

if (os.path.isfile(fn1)):            
    
    sharpenModel = rn.RvNeuralEnDeCoder(fn1)
    
    imgPath = "{}\\{}\\".format( sharpenModel.VideoImagePath, "SharpenModel")   
    rfi.ForceDir(imgPath) #if not os.path.isdir(imgPath): os.mkdir(imgPath)
    rfi.Delete_Files(imgPath, [".jpg",".png"])
        
    sampleNum=20
    durationSec = min(1.0, 10/sampleNum)
    
    if (None!=sharpenModel):        
      
        rf.Test_EnDecoder(sharpenModel, lstTest, sampleNum, imgPath, noiseStrength)

        aviFn = "{}{}".format(imgPath, "SharpenModel.avi")
        
        if ru.ImageFilesToAvi(imgPath, aviFn, durationSec ):
            rfi.OpenFile(aviFn)





