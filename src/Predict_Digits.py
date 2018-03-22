

# -*- coding: utf-8 -*-


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


Created on Wed Feb  7 22:11:59 2018

@author: dan59314
"""

#%%
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
import matplotlib.cm as cm


# prvaite libraries---------------------------------------------
import mnist_loader
import RvNeuralNetworks as rn
from RvNeuralNetworks import *
import RvAskInput as ri
import RvMiscFunctions as rf
import RvNeuNetworkMethods as nm
import PlotFunctions as pltFn    
import RvFileIO as rfi    
        
    
#%%
    
    
#Load MNIST ****************************************************************

#mnist.pkl.gz(50000） Accuracy 0.96 
#mnist_expanded.pkl.gz(250000） Accuracy 0.97 
fn = "..\\data\\mnist.pkl.gz"  #".datamnist_expanded.pkl.gz"
lstTrain, lstV, lstT =  mnist_loader.load_data_wrapper(fn)
lstTrain = list(lstTrain)
lstV = list(lstV)
lstT = list(lstT)


fns, fn0s =  rfi.Get_FilesInFolder(".\\NetData\\", [".cnn", ".dnn"])

if len(fns)>0:
    aId = ri.Ask_SelectItem("Select Network file", fn0s, 0)    
    fn1= fns[aId]
        
    accur1,t1 = rf.Predict_Digits_FromNetworkFile(fn1, lstT, True)
    print("File : \"{}\"\n  nmcu:{}, Time:{:.3} sec\n".format(fn1,accur1,t1))


    
    
    
    