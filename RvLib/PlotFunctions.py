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
import RvNeuNetworkMethods as nm
from RvNeuNetworkMethods import EnumDropOutMethod as drpOut



#%%%  Function Section

sResult = ["錯誤", "正確"]
    
    
    
#%%
def Predict_Digits(net, lstT, plotDigit=True):
    # 隨機測試某筆數字 ----------------------------------------------    
    start = time.time() 
    
    sampleNum=10000 # 不含繪圖，辨識 10000張，費時 1.3 秒，平均每張 0.00013秒
    plotNum = 1
    plotMod = int(sampleNum/plotNum) + 1
    correctNum=0    
    failNum = 0
    for i in range(0, sampleNum):
        doPlot = (i%plotMod == 0)
        aId = np.random.randint(0,len(lstT))
        label, result = net.Predict_Digit(lstT[aId], False)    
        if label==result: correctNum+=1
        else: 
            doPlot = (failNum<plotNum) 
            failNum+=1
        if doPlot and plotDigit:
            rf.Plot_Digit(lstT[aId], reslt, label)
            print("({}): Label={}, Predict:{} -> {} ".
              format(i, label,result, sResult[int(label==result)]))   
    
    dt = time.time()-start
    
    accurRatio = correctNum/sampleNum
    print("\nResult: Accuracy({:.3f}),  {}/{}(Correct/Total)".
          format(accurRatio, correctNum, sampleNum))    
    print("Elapsed(seconds)) : {:.3f} sec.\n".format(dt))    
    return accurRatio,dt

    
def Predict_Digits_FromNetworkFile(fn, lstT, plotDigit=True):
    if (os.path.isfile(fn)):
        net = rn.RvNeuralNetwork.Create_Network(fn)
        if (None==net): return 0.0, 0.0    
        return Predict_Digits(net, lstT, plotDigit)    







#%%% Test Section






#%%% Main Section





