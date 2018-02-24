

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
import matplotlib.cm as cmn


# prvaite libraries---------------------------------------------
import mnist_loader
import RvNeuralNetworks as rn
from RvNeuralNetworks import *
import RvAskInput as ri
import RvMiscFunctions as rf
import RvNeuNetworkMethods as nm
from RvNeuNetworkMethods import EnumDropOutMethod as drpOut
import PlotFunctions as pltFn




#%%      
        

def Main():
    #Load MNIST ****************************************************************
    
    #mnist.pkl.gz(50000） Accuracy 0.96 
    #mnist_expanded.pkl.gz(250000） Accuracy 0.97 
    fn = "..\\data\\mnist_expanded.pkl.gz"  #".datamnist_expanded.pkl.gz"
    lstTrain, lstV, lstT =  mnist_loader.load_data_wrapper(fn)
    lstTrain = list(lstTrain)
    lstV = list(lstV)
    lstT = list(lstT)
        
    
    #Hyper pameters -------------------------------------------    
    loop = 10  # loop effect，10, 30 all above 0.95
    stepNum = 10  # stepNum effect,　10->0.9,  100->0.5
    learnRate = 0.1  # learnRate and lmbda will affect each other
    lmbda = 5.0     #add lmbda(Regularization) to solve overfitting 
        

    enableDropOut = ri.Ask_YesNo("Excute DropOut?", "n")
    if enableDropOut:
        enumDropOut = ri.Ask_Enum("Select DropOut Method.", 
        nm.EnumDropOutMethod,  drpOut.eoSmallActivation )
        rn.gDropOutRatio = ri.Ask_Input_Float("Input DropOut ratio.", rn.gDropOutRatio)
    
    monitoring = ri.Ask_YesNo("Watch the training process?", "y")
    
         
    
    #Auto caculate proper Hyper pameters  ---
    loop,stepNum,learnRate,lmbda = rf.Ask_Input_SGD(loop,stepNum,learnRate,lmbda)
    
    print( "Hyper Pameters: Loop({}), stepNum({}), learnRatio({}), lmbda({})\n".format(loop,stepNum,learnRate,lmbda)  )
    
    
    # Load net file and continune training--
    fns = []
    for file in os.listdir(".\\"):
        if file.endswith(".txt"):
           fns.append(file)
    aId = ri.Ask_SelectItem("Select network file", fns, 0)    
    fn1= fns[aId]
    while (True):
      DoKeepTraining = ri.Ask_YesNo("Select Network file {} and continue training?".format(fn1), "y")
      if DoKeepTraining:
        if (os.path.isfile(fn1)):
            net = rn.RvNeuralNetwork.Create_Network(fn1)        
            if (None!=net):
                net.Motoring_TrainningProcess = monitoring
                if enableDropOut:        
                    net.Set_DropOutMethod(enumDropOut, rn.gDropOutRatio)
                start = time.time()   
                # Start Training
                net.Train(lstTrain, loop, stepNum, learnRate, lstV, lmbda, False )                
                dT = time.time()-start         
                
                if net.BestAccuracyRatio>net.Get_NetworkFileData(fn1):    
                  rf.Save_NetworkDataFile(net, fn1, loop,stepNum,learnRate,lmbda, dT)
                  print("Save Network file \"{}\"".format(fn1))
      else:
        break
    
    


#%% Main Section ***************************************************************    
Main()
    
    
    