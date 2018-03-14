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
import PlotFunctions as pltFn

from RvNeuNetworkMethods import EnumDropOutMethod as drpOut

#%%
AddNoise = False

#%%%  Function Section








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

# Prediction ------------------------------------------------

fns, shortFns =  rfi.Get_FilesInFolder(".\\NetData\\", [".decoder"])
aId = 0 #ri.Ask_SelectItem("Select Decoder file", shortFns, 0)    
fn1= fns[aId]

fns, shortFns =  rfi.Get_FilesInFolder(".\\NetData\\", [".encoder"])
aId = 0 #ri.Ask_SelectItem("Select Encoder file", shortFns, 0)    
fn2= fns[aId]

#addNoise = ri.Ask_YesNo("Add noise?", "n")
noiseStrength = 0.8 #ri.Ask_Input_Float("Input Noise Strength.", 0.0)

randomState = np.random.RandomState(int(time.time()))

if (os.path.isfile(fn1)):            
    
    decoder = rn.RvNeuralEnDeCoder(fn1)
    #encoder = rn.RvNeuralEnDeCoder(fn2)
    
    inputNum = decoder.Get_InputNum()
    outputNum = decoder.Get_OutputNum()
    
    imgPath = "{}\\{}\\".format(decoder.VideoImagePath, "EncoderDecoder")   
    rfi.ForceDir(imgPath) #if not os.path.isdir(imgPath): os.mkdir(imgPath)
    rfi.Delete_Files(imgPath, [".jpg",".png"])
        
    sampleNum=20
    durationSec = min(0.2, 10/sampleNum)
    
    if (None!=decoder): #and (None!=encoder):   
               
        pxlW = int(np.sqrt(outputNum))         
      
        #rf.Test_Encoder_Decoder(encoder, decoder, lstT, sampleNum, imgPath, noiseStrength)
          
        
        for j in range(100):  # 多做幾次，效果不會比較好
            encode = randomState.randn(inputNum,1)   
            encode = np.maximum(0.2, np.minimum(0.8, encode))
                          
            #output = decoder.Plot_Output(encode) #(784x1)    
            output = decoder.Get_OutputValues(encode)    
            
            print("code -> Output")              
            if rfi.PathExists(imgPath):
                imgFn = "{}vdoImg_{}.png".format(imgPath, j)
            else:
                imgFn = ""               
            
            pltFn.Plot_Images(np.array([output.transpose().reshape(pxlW,pxlW)*255]),
                1,1, "Test EnDeCoder", imgFn, 6)
            oneInput = output
    
        aviFn = "{}{}".format(imgPath, "EncoderDecoder.avi")
        
        if ru.ImageFilesToAvi(imgPath, aviFn, durationSec ):
            rfi.OpenFile(aviFn)





