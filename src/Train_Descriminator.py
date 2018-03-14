

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




#%%  Function Section



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
    
    fnNetworkData = "{}{}_NetData".format(path,rn.RvNeuralDescriminator.__name__)   
    fnNetworkData1 = ""
    
    
    sampleNum = 1000
    #Hyper pameters -------------------------------------------    
    loop = 10  # loop effect，10, 30 all above 0.95
    stepNum = sampleNum  # stepNum effect,　10->0.9,  100->0.5
    learnRate = 0.1  # learnRate and lmbda will affect each other
    lmbda = 5.0     #add lmbda(Regularization) to solve overfitting 
    
    
    
    # Training ***********************************************************
    # Ask DoTraining-
    DoTraining = ri.Ask_YesNo("Do Training?", "y")
    if DoTraining:             
        """
        [784,50,10], loop=100, 0.9725
        """
         # Create RvNeuralDescriminator----------------------------------------------
        inputNeusNum = len(lstTrain[0][0])
        lyr1NeuNum = 50
        outputNeuNum = 1 #len(lstTrain[0][1])
        
        lyrsNeus = [inputNeusNum, lyr1NeuNum]
        lyrsNeus = ri.Ask_Add_Array_Int("Input new layer Neurons num.", lyrsNeus, lyr1NeuNum)
        lyrsNeus.append(outputNeuNum)
        
      
        descriminator = rn.RvNeuralDescriminator(lyrsNeus)  # ([784,50,10])
                
        
        # Ask nmtivation  ------------------_----------
        enumActivation = ri.Ask_Enum("Select Activation method.", 
             nm.EnumActivation,  nm.EnumActivation.afReLU )
        for lyr in descriminator.NeuralLayers:
            lyr.ClassActivation, lyr.ClassCost = \
            nm.Get_ClassActivation(enumActivation)
        
        descriminator.Motoring_TrainningProcess = rn.Debug
    
        descriminator.NetEnableDropOut = ri.Ask_YesNo("Execute DropOut?", "n")
        if descriminator.NetEnableDropOut:
            enumDropOut = ri.Ask_Enum("Select DropOut Method.", 
            nm.EnumDropOutMethod,  drpOut.eoSmallActivation )
            rn.gDropOutRatio = ri.Ask_Input_Float("Input DropOut ratio.", rn.gDropOutRatio)
            descriminator.Set_DropOutMethod(enumDropOut, rn.gDropOutRatio)
        
        monitoring = ri.Ask_YesNo("Watch training process?", "y")
        descriminator.Motoring_TrainningProcess = monitoring
             
        
        # Auto-Caculate proper hyper pameters ---
        DoEvaluate_ProperParams = ri.Ask_YesNo("Auto-Caculating proper hyper pameters?", "n")
        if DoEvaluate_ProperParams:
            loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_lmbda(
                    descriminator, descriminator.Train, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
            loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_learnRate(
                    descriminator, descriminator.Train, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
        else:      
            loop,stepNum,learnRate,lmbda = rf.Ask_Input_SGD(loop,stepNum,learnRate,lmbda)
        
        print( "Hyper pameters: Loop({}), stepNum({}), learnRatio({}), lmbda({})\n".format(loop,stepNum,learnRate,lmbda)  )
    
    
    
        randomState = np.random.RandomState(int(time.time()))
    
        # 利用取樣集合 mini_trainingData[]，逐一以每個小測試集，　更新 weights 和 biases 
        TrueDigit_sets = [ lstTrain[k:k+stepNum] 
            for k in range(0, len(lstTrain), stepNum)]
        selId = randomState.randint(len(TrueDigit_sets))
        
        """image[:,:] = [[ min(pixel + dVal, 255) 
            for pixel in row] for row in image[:,:]]"""
        FakeDigit_set = [ 
            tuple([ randomState.randn(len(aTuple[0]),1) ,  
                    rn.RvBaseNeuralNetwork.CreateLabelsY(10,randomState.randint(10)) ]) 
            for aTuple in TrueDigit_sets[0] ]
                
        start = time.time()   
        # Start Training-
        descriminator.Train_Descriminator(TrueDigit_sets[selId], loop, stepNum, 
            learnRate, lmbda,  blInitialWeiBias=True, labelY=1 )
        
        descriminator.Train_Descriminator(FakeDigit_set, loop, stepNum, 
            learnRate, lmbda,  blInitialWeiBias=False, labelY=0 )
        
        dT = time.time()-start
        
        fnNetworkData1 = rf.Save_NetworkDataFile(descriminator, fnNetworkData, loop,stepNum,
            learnRate,lmbda, dT, ".descriminator")
    
    # Prediction ------------------------------------------------
    if (not os.path.isfile(fnNetworkData1)): 
        fnNetworkData1= ".\\NetData\\{}_NetData.descriminator". \
            format(rn.RvNeuralDescriminator.__name__)
    
    
#    # Ask DoPredict----------------------------------------------------
#    DoPredict=True
##    if DoTraining: DoPredict = ri.Ask_YesNo("Predict digits?", "n")   
##    else: DoPredict = True
#
#    if DoPredict:          
#        rn.Debug_Plot = True #ri.Ask_YesNo("Plot Digits?", "n") 
#    #    Predict_Digits(net, lstT)
#        rf.Predict_Digits_FromNetworkFile(fnNetworkData1, lstT, rn.Debug_Plot)    
#    
    
    
    
#%% Test Section *********************************************************************
    
#Test RvNeuralLayer --------------------------------------
#rvLyr = RvNeuralLayer([30,10])
#rvLyr1 = RvNeuralLayer(10, 20)

#測試 RvNeuralDescriminator() overload Constructor ---------------    
#lyrsNeus = [784,50,10]
#rvNeuLyrs = RvNeuralDescriminator.LayersNeurons_To_RvNeuralLayers(lyrsNeus)
#net = RvNeuralDescriminator(rvNeuLyrs)    
#net = RvNeuralDescriminator.Class_Create_LayersNeurons(lyrsNeus)    
#net = RvNeuralDescriminator(lyrsNeus)    
    
    
    
#print(ac.EnumActivation.afReLU)  #EnumActivation.afReLU
#print(ac.EnumActivation.afReLU.name)   #afReLU
#print(ac.EnumActivation.afReLU.value)  #2
#print(type(ac.EnumActivation.afReLU))  #<enum 'EnumActivation'>
#print(type(ac.EnumActivation.afReLU.name)) #<class 'str'>
#print(type(ac.EnumActivation.afReLU.value))    #<class 'int'>




#%% Main Section ***************************************************************    
Main()
    
    
    