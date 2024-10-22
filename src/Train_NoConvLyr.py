

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
#from RvNeuralNetworks import *
import RvAskInput as ri
import RvMiscFunctions as rf
import RvNeuNetworkMethods as nm
from RvNeuNetworkMethods import EnumDropOutMethod as drpOut
import PlotFunctions as pltFn
import RvFileIO as rfi




#%%  

        
        

def Main():
    #Load MNIST ****************************************************************
    
    #Use mnist.pkl.gz(50000 data） Accuracy 0.96 
    #Use mnist_expanded.pkl.gz(250000 data） Accuracy 0.97 
    fn = "..\\data\\mnist.pkl.gz"  #".datamnist_expanded.pkl.gz"
    lstTrain, lstV, lstT =  mnist_loader.load_data_wrapper(fn)
    lstTrain = list(lstTrain)
    lstV = list(lstV)
    lstT = list(lstT)
    
    
    path = "..\\TmpLogs\\"
    if not os.path.isdir(path):
        os.mkdir(path)           
    
    fnNetworkData = "{}{}_DNN".format(path,rn.RvNeuralNetwork.__name__)   
    fnNetworkData1 = ""
    
    
    #Hyper pameters -------------------------------------------    
    loop = 10  # loop effect，10, 30 all above 0.95
    stepNum = 10  # stepNum effect,　10->0.9,  100->0.5
    learnRate = 0.1  # learnRate and lmbda will affect each other
    lmbda = 5.0     #add lmbda(Regularization) to solve overfitting 
    
    
    sTrain = "y"
    
    # Training ***********************************************************
    # Ask DoTraining-
    LoadAndTrain = ri.Ask_YesNo("Load exist model and continue training?", "n")    
    
    if LoadAndTrain:    
        fns, shortFns =  rfi.Get_FilesInFolder(".\\NetData\\", [".dnn"])
        aId = ri.Ask_SelectItem("Select DNN Network file", shortFns, 0)    
        fn1= fns[aId]
        fnNetworkData1 = fn1
        net = rn.RvNeuralNetwork(fn1) 
        initialWeiBias = False
        sTrain = "n"
        
    else:            
        """
        [784,50,10], loop=100, 0.9725
        """
         # Create RvNeuralNetWork----------------------------------------------
        inputNeusNum = len(lstTrain[0][0])
        lyr1NeuNum = 50
        lyr2NeuNum = len(lstTrain[0][1])
        
        lyrsNeus = [inputNeusNum, lyr1NeuNum]
        lyrsNeus = ri.Ask_Add_Array_Int("Input new layer Neurons num.", lyrsNeus, lyr1NeuNum)
        lyrsNeus.append(lyr2NeuNum)
        
        #net = rn.RvNeuralNetwork( \
        #   rn.RvNeuralNetwork.LayersNeurons_To_RvNeuralLayers(lyrsNeus))
        #net = rn.RvNeuralNetwork.Class_Create_LayersNeurons(lyrsNeus)
        net = rn.RvNeuralNetwork(lyrsNeus)  # ([784,50,10])
        initialWeiBias = True


        
    
    # Training ***********************************************************
    DoTrain = ri.Ask_YesNo("Do Training?", sTrain)    
    if DoTrain:        
        fnNetworkData = "{}_{}Lyr".format(fnNetworkData, len(net.NeuralLayers))
               
        net.DoPloatWeights = ri.Ask_YesNo("Plot Neurons Weights?", 'n')
            
        
        # Ask nmtivation  ------------------_----------
        enumActivation = ri.Ask_Enum("Select Activation method.", 
             nm.EnumActivation,  nm.EnumActivation.afReLU )
        for lyr in net.NeuralLayers:
            lyr.ClassActivation, lyr.ClassCost = \
            nm.Get_ClassActivation(enumActivation)
        
    
        net.NetEnableDropOut = ri.Ask_YesNo("Execute DropOut?", "n")
        if net.NetEnableDropOut:
            enumDropOut = ri.Ask_Enum("Select DropOut Method.", 
            nm.EnumDropOutMethod,  drpOut.eoSmallActivation )
            rn.gDropOutRatio = ri.Ask_Input_Float("Input DropOut ratio.", rn.gDropOutRatio)
            net.Set_DropOutMethod(enumDropOut, rn.gDropOutRatio)
        
        
        # Auto-Caculate proper hyper pameters ---
        DoEvaluate_ProperParams = ri.Ask_YesNo("Auto-Caculating proper hyper pameters?", "n")
        if DoEvaluate_ProperParams:
            loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_lmbda(
                    net, net.Train, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
            loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_learnRate(
                    net, net.Train, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
        else:      
            loop,stepNum,learnRate,lmbda = rf.Ask_Input_SGD(loop,stepNum,learnRate,lmbda)
        
        print( "Hyper pameters: Loop({}), stepNum({}), learnRatio({}), lmbda({})\n".format(loop,stepNum,learnRate,lmbda)  )
    
    
        
        # Start Training-
        keepTraining = True
        while (keepTraining):
            start = time.time()   
            net.Train(lstTrain, loop, stepNum, learnRate, lstV, lmbda, initialWeiBias )
            initialWeiBias = False
        
            dT = time.time()-start
            
            fnNetworkData1 = rf.Save_NetworkDataFile(net, fnNetworkData, loop,stepNum,
                learnRate,lmbda, dT, ".dnn")
            
            keepTraining = ri.Ask_YesNo("Keep Training?", "y")

    # Prediction ------------------------------------------------
    if (not os.path.isfile(fnNetworkData1)): 
        fnNetworkData1= ".\\NetData\\{}_DNN.dnn". \
            format(rn.RvNeuralNetwork.__name__)
    
    
    # Ask DoPredict----------------------------------------------------
    DoPredict=True
#    if DoTraining: DoPredict = ri.Ask_YesNo("Predict digits?", "n")   
#    else: DoPredict = True

    if DoPredict:          
        rn.Debug_Plot = True #ri.Ask_YesNo("Plot Digits?", "n") 
    #    Predict_Digits(net, lstT)
        rf.Predict_Digits_FromNetworkFile(fnNetworkData1, lstT, rn.Debug_Plot)    
    
    
    
    
#%% Test Section *********************************************************************
    
#Test RvNeuralLayer --------------------------------------
#rvLyr = RvNeuralLayer([30,10])
#rvLyr1 = RvNeuralLayer(10, 20)

#測試 RvNeuralNetwork() overload Constructor ---------------    
#lyrsNeus = [784,50,10]
#rvNeuLyrs = RvNeuralNetwork.LayersNeurons_To_RvNeuralLayers(lyrsNeus)
#net = RvNeuralNetwork(rvNeuLyrs)    
#net = RvNeuralNetwork.Class_Create_LayersNeurons(lyrsNeus)    
#net = RvNeuralNetwork(lyrsNeus)    
    
    
    
#print(ac.EnumActivation.afReLU)  #EnumActivation.afReLU
#print(ac.EnumActivation.afReLU.name)   #afReLU
#print(ac.EnumActivation.afReLU.value)  #2
#print(type(ac.EnumActivation.afReLU))  #<enum 'EnumActivation'>
#print(type(ac.EnumActivation.afReLU.name)) #<class 'str'>
#print(type(ac.EnumActivation.afReLU.value))    #<class 'int'>




#%% Main Section ***************************************************************    
Main()
    
    
    