# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:47:01 2018

@author: dan59314
"""


"""
/* -----------------------------------------------------------------------------
  Copyright: (C) Daniel Lu, RasVector Technology.

  Email : dan59314@gmail.com
  Web :     http://www.rasvector.url.tw/
  YouTube : http://www.youtube.com/dan59314/playlist

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


#%%
# Standard library----------------------------------------------
import sys
sys.path.append('./RvLib')
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
import RvActivationCost as ac
from RvActivationCost import EnumCnvFilterMethod as fm
from RvActivationCost import EnumDropOutMethod as drpOut


#%% 測試
#oneInputPiece=np.asarray([1,2,3,4,5])
#oneFilter = np.asarray([1,2,3,4,5])
#print(ac.ClassCnvFilter.Get_CnvFilterValues( \
#        oneInputPiece, oneFilter, fm.fmAverageSum) )

channel = 1

inputW = 28
inputH = 28
inputX = np.ones((inputW,inputH))
inputFlatten = inputX.reshape((inputW*inputH,1))
inputShape =[len(inputX[0]), len(inputX),1]

print("\ninputData.shape={}".format(inputX.shape))
print("inputShape={}".format(inputShape))


filterW       = 5
filterH       = 5
filterNum     = 25
filterShape = np.asarray([filterW,filterH, channel, filterNum])

filterStride = 1
print("filterShape={}".format(filterShape))

stepX,restX = rn.RvConvolutionLayer.Get_StepsNum_Convolution(inputShape[0],filterShape[0], filterStride)
stepY,restY = rn.RvConvolutionLayer.Get_StepsNum_Convolution(inputShape[1],filterShape[1], filterStride)
print("inputStepsXY=({},{}), restXY=({},{})".format(stepX,stepY,restX,restY))

#inputX=np.insert(inputX, 0, 3, axis=0)
#print("Input.shape={}".format(inputX.shape))
        
inputX=rn.RvConvolutionLayer.Padding_Input(inputX, [filterW,filterH], filterStride)

cl = rn.RvConvolutionLayer(inputShape,filterShape,filterStride)
cl.Caculate_Neurons_Z_Activations(inputFlatten)
cl.Plot_NeuronsWeights([10, 20])

lyrData = cl.Get_LayerData()
print("LyrData:{}".format(lyrData))

    
    
    
    
    