

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
sys.path.append('..//data')
sys.path.append('..//RvLib')
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
import RvAskInput as ri
import RvMiscFunctions as rf
import RvNeuNetworkMethods as nm
from RvNeuNetworkMethods import EnumDropOutMethod as drpOut
import PlotFunctions as pltFn
import RvMediaUtility as ru


#%%
 
       


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
    
    # Load digit Images  lstTrain[0].shape = (pxls, label) = (784, 1)
#    imgPxls = lstTrain[0][0].shape[0]
#    digitImages = rn.RvNeuralEnDeCoder.Load_DigitImages( ".\\Images\\Digits\\", imgPxls)
    
    path = "..\\TmpLogs\\"
    if not os.path.isdir(path):
        os.mkdir(path)           
    
    fnNetworkData = "{}{}_NetData".format(path,rn.RvNeuralEnDeCoder.__name__)   
    fnNetworkData1 = ""
    
    
    #Hyper pameters -------------------------------------------    
    loop = 10  # loop effect，10, 30 all above 0.95
    stepNum = 10  # stepNum effect,　10->0.9,  100->0.5
    learnRate = 0.1  # learnRate and lmbda will affect each other
    lmbda = 5.0     #add lmbda(Regularization) to solve overfitting 
    
    
    
    # Training ***********************************************************
    # Ask DoTraining-
    DoTraining = ri.Ask_YesNo("Do Training?", "y")
    if DoTraining:             
        """
        [784,50,10], loop=100, 0.9725
        """
        
        #buildDeNoiseModel = ri.Ask_YesNo("Build DeNoise Model?", "n")
        
         # Create RvNeuralEnDeCoder----------------------------------------------
        inputNeusNum = len(lstTrain[0][0])
        #lyr2NeuNum = len(lstTrain[0][1])
        
        # [784,256,128,10] is suggested ----------------------------
        lyrsNeus = [inputNeusNum, 256,128]
        lyrsNeus = ri.Ask_Add_Array_Int("Input new layer Neurons num.", lyrsNeus, 50)
        
        bottleneckNeuNum = ri.Ask_Input_Integer("Input BottleNeck Neurons num.", 10)
        lyrsNeus.append(bottleneckNeuNum)
        for nNeu in reversed(lyrsNeus[1:-1]):
            lyrsNeus.append(nNeu)
        lyrsNeus.append(inputNeusNum)
        
        digitIdOnly = -1
        digitIdOnly = ri.Ask_Input_Integer("Build only Digit (0~9, -1 = All Digits): ", digitIdOnly)
        
        
        #net = RvNeuralEnDeCoder( 
        #   RvNeuralEnDeCoder.LayersNeurons_To_RvNeuralLayers(lyrsNeus))
        #net = RvNeuralEnDeCoder.Class_Create_LayersNeurons(lyrsNeus)
        net = rn.RvNeuralEnDeCoder(lyrsNeus)  # ([784,50,10])
                
        
        # Ask nmtivation  ------------------_----------
        enumActivation = ri.Ask_Enum("Select Activation method.", 
             nm.EnumActivation,  nm.EnumActivation.afSigmoid)
        for lyr in net.NeuralLayers:
            lyr.Set_EnumActivation(enumActivation)
        
        net.Motoring_TrainningProcess = rn.Debug
    
        net.NetEnableDropOut = ri.Ask_YesNo("Execute DropOut?", "n")
        if net.NetEnableDropOut:
            enumDropOut = ri.Ask_Enum("Select DropOut Method.", 
            nm.EnumDropOutMethod,  drpOut.eoSmallActivation )
            rn.gDropOutRatio = ri.Ask_Input_Float("Input DropOut ratio.", rn.gDropOutRatio)
            net.Set_DropOutMethod(enumDropOut, rn.gDropOutRatio)
        
        monitoring = ri.Ask_YesNo("Watch training process?", "y")
        net.Motoring_TrainningProcess = monitoring
             
        
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
    
    
        start = time.time()   
        # Start Training-
        encoder, decoder = net.Build_Encoder_Decoder( \
            lstTrain, loop, stepNum, learnRate, lmbda, True, digitIdOnly)
        
        dT = time.time()-start
        
        
        rf.Save_NetworkDataFile(net, fnNetworkData, 
                loop,stepNum,learnRate,lmbda, dT, ".endecoder")
    
        fn1 = rf.Save_NetworkDataFile(encoder, 
                "{}_Encoder".format(fnNetworkData), 
                loop,stepNum,learnRate,lmbda, dT, ".encoder")
        fn2 = rf.Save_NetworkDataFile(decoder, 
                "{}_Decoder".format(fnNetworkData), 
                loop,stepNum,learnRate,lmbda, dT, ".decoder")
    
        
        decoder = rn.RvNeuralEnDeCoder.Create_Network(fn2)
        encoder = rn.RvNeuralEnDeCoder.Create_Network(fn1)
        
        
        noiseStrength = ri.Ask_Input_Float("Input Noise Strength.", 0.0)
        rf.Test_Encoder_Decoder(encoder, decoder, lstT,10, 
            "", noiseStrength)
        
    
#%% Test Section *********************************************************************
    

#net = rn.RvNeuralEnDeCoder([784,50,10,50,784])
#ru.ImageFilesToAvi(net.VideoImagePath, 
#                   "{}{}".format(net.VideoImagePath, "EnDeCoder.avi") )
    
    


#%% Main Section ***************************************************************    
Main()

    