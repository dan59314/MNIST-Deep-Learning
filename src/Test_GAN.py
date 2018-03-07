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

from RvNeuNetworkMethods import EnumDropOutMethod as drpOut

#%%

AppendInputX = False

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

randomState = np.random.RandomState(int(time.time()))

if (os.path.isfile(fn1)):         
#    inputX = randomState.randn(784,1)
#    net = rn.RvNeuralNetwork.Create_Network(fn1)    
#    label, result = net.Predict_Digit([inputX,1], False)   
    
    
    decoder = rn.RvNeuralEnDeCoder.Create_Network(fn1)
    encoder = rn.RvNeuralEnDeCoder.Create_Network(fn2)
    
    imgPath = decoder.VideoImagePath
    
    bottleneckNeuNum = decoder.Get_InputNum()
    inputX = randomState.randn(bottleneckNeuNum,1)     
    
    learnRate = 0.05
    # Load digit Images  lstTrain[0].shape = (pxls, label) = (784, 1)
    outputNeus = decoder.Get_OutputNum()
    decoderLabels = rn.RvNeuralEnDeCoder.Load_DigitImages( ".\\Images\\Digits\\", outputNeus)
    toDigit = randomState.randint(0,10)
    digitChange = [toDigit]
    labelY = decoder.CreateLabelsY(10, toDigit)
    
    inputDatas = [[inputX,labelY]]
    
    if (None!=decoder):        
        for i in range(20):            
            print("Plot_Output({}): ".format(i))                  
            
            #更新 decoder 權重 -------------------------------------------------
            decoder.Update_LayersNeurons_Weights_Biases_OneInput( \
               inputDatas, learnRate, trainOnlyDigit=-1) # 
            
            # 計算 outupt 並劃出 -----------------------
            fn = "{}{}_{}.png".format(imgPath,"genImg", i)
            output = decoder.Plot_Output(inputX, fn)  
            
            # 尋找最小 Error 的 digit，當作生成目標-----------------------
#            nDigit, _ = decoder.Get_MinMaxErrorDigit(output, decoderLabels)
#            if (nDigit!=toDigit): digitChange.append(nDigit)   
#            toDigit = nDigit
#            labelY = decoder.CreateLabelsY(10, toDigit)
            
            #更新 encoder 權重 -------------------------------------------------
            
            # 根據新的 output，輸入到 encoder，取得新的 inputX
            inputX = encoder.Get_OutputValues(output)  
            
            # 增加亂數 ---------------------------------------
#            inputX += randomState.randn(bottleneckNeuNum,1)/10
#            avgX = sum( np.linalg.norm(x) for x in inputDatas[0])/len(inputDatas)
#            inputX += (avgX*0.5) 
            
            # 更新 InputX -------------------------------------
            if AppendInputX:
              inputDatas.append([inputX,labelY])
            else:
              inputDatas = [[inputX,labelY]]  
              
            

    print("Digit Change : {}".format(digitChange) )

    aviFn = "{}{}".format(imgPath, "GenerateDigit_{}.avi".format(toDigit))
#    print("AviFn = {}".format(aviFn))
    if ru.ImageFilesToAvi(imgPath, aviFn ):
        os.system(r'start ' + aviFn)





