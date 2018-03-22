

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
import RvFileIO as rfi


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
    rfi.ForceDir(path)          
    
    fnNetworkData = "{}{}_EDC".format(path,rn.RvNeuralEnDeCoder.__name__)   
    
    
    #Hyper pameters -------------------------------------------    
    loop = 10  # loop effect，10, 30 all above 0.95
    stepNum = 10  # stepNum effect,　10->0.9,  100->0.5
    learnRate = 0.1  # learnRate and lmbda will affect each other
    lmbda = 5.0     #add lmbda(Regularization) to solve overfitting 
    
    
    endecoder = None
    
    sTrain = "y"
        
    # Training ***********************************************************
    # Ask DoTraining-
    LoadAndTrain = ri.Ask_YesNo("Load exist model and continue training?", "n")    
    
    if LoadAndTrain:       
        digitIdOnly = -1 
        fns, shortFns =  rfi.Get_FilesInFolder(".\\NetData\\", [".endecoder"])
        aId = ri.Ask_SelectItem("Select EnDecoder file", shortFns, 0)    
        fn1= fns[aId]
        endecoder = rn.RvNeuralEnDeCoder(fn1)   
        initialWeights = False
        sTrain = "n"
        encoder, decoder = endecoder.Get_Encoder_Decoder()
        
    else:             
        """
        [784,50,10], loop=100, 0.9725
        """
        #buildDeNoiseModel = ri.Ask_YesNo("Build DeNoise Model?", "n")
        
         # Create RvNeuralEnDeCoder----------------------------------------------
        inputNeusNum = len(lstTrain[0][0])
        #lyr2NeuNum = len(lstTrain[0][1])
        
        # [784,256,128,10] is suggested ----------------------------
        # [784, 256, 128, 10, 128, 256, 784 ] -> 0.9395 ... tested 10 epochs
        # [784, 400, 20, 400, 784] -> 0.9526 ... tested 5 epochs
        lyrsNeus = [inputNeusNum, 50] # 512, 256,128]
        lyrsNeus = ri.Ask_Add_Array_Int(\
            "Input new layer Neurons num.", lyrsNeus, 50)
        
        bottleneckNeuNum = ri.Ask_Input_Integer(\
            "Input BottleNeck(Code) Layer Neurons num.", 10)
        lyrsNeus.append(bottleneckNeuNum)
        for nNeu in reversed(lyrsNeus[1:-1]):
            lyrsNeus.append(nNeu)
        lyrsNeus.append(inputNeusNum)
        
        digitIdOnly = -1
        digitIdOnly = ri.Ask_Input_Integer("Build only Digit (0~9, -1 = All Digits): ", digitIdOnly)
        
        #net = RvNeuralEnDeCoder( 
        #   RvNeuralEnDeCoder.LayersNeurons_To_RvNeuralLayers(lyrsNeus))
        #net = RvNeuralEnDeCoder.Class_Create_LayersNeurons(lyrsNeus)
        endecoder = rn.RvNeuralEnDeCoder(lyrsNeus)  # ([784,50,10])        
        initialWeights = True
    
    
    
    
    # Training ***********************************************************
    DoTrain = ri.Ask_YesNo("Do Training?", sTrain)    
    if DoTrain:        
        fnNetworkData = "{}_{}Lyr".format(fnNetworkData, len(endecoder.NeuralLayers))
               
        endecoder.DoPloatWeights = ri.Ask_YesNo("Plot Neurons Weights?", 'n')
            
        # Ask nmtivation  ------------------_----------
        enumActivation = ri.Ask_Enum("Select Activation method.", 
             nm.EnumActivation,  nm.EnumActivation.afSigmoid)
        for lyr in endecoder.NeuralLayers:
            lyr.Set_EnumActivation(enumActivation)
        
        endecoder.NetEnableDropOut = ri.Ask_YesNo("Execute DropOut?", "n")
        if endecoder.NetEnableDropOut:
            enumDropOut = ri.Ask_Enum("Select DropOut Method.", 
            nm.EnumDropOutMethod,  drpOut.eoSmallActivation )
            rn.gDropOutRatio = ri.Ask_Input_Float("Input DropOut ratio.", rn.gDropOutRatio)
            endecoder.Set_DropOutMethod(enumDropOut, rn.gDropOutRatio)
        
        
        # Auto-Caculate proper hyper pameters ---
        DoEvaluate_ProperParams = ri.Ask_YesNo("Auto-Caculating proper hyper pameters?", "n")
        if DoEvaluate_ProperParams:
            loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_lmbda(
                    endecoder, endecoder.Train, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
            loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_learnRate(
                    endecoder, endecoder.Train, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
        else:      
            loop,stepNum,learnRate,lmbda = rf.Ask_Input_SGD(loop,stepNum,learnRate,lmbda)
        
        print( "Hyper pameters: Loop({}), stepNum({}), learnRatio({}), lmbda({})\n".format(loop,stepNum,learnRate,lmbda)  )
    
    
        # Start Training-         
        keepTraining = True
        while (keepTraining):
            start = time.time()  
            
            encoder, decoder = endecoder.Build_Encoder_Decoder( \
              lstTrain, loop, stepNum, learnRate, lmbda, initialWeights, digitIdOnly)
            initialWeights = False
              
            dT = time.time()-start            
            
            rf.Save_NetworkDataFile(endecoder, fnNetworkData, 
                    loop,stepNum,learnRate,lmbda, dT, ".endecoder")        
            fn1 = rf.Save_NetworkDataFile(encoder, 
                    "{}_Encoder".format(fnNetworkData), 
                    loop,stepNum,learnRate,lmbda, dT, ".encoder")
            fn2 = rf.Save_NetworkDataFile(decoder, 
                    "{}_Decoder".format(fnNetworkData), 
                    loop,stepNum,learnRate,lmbda, dT, ".decoder")
            
            keepTraining = ri.Ask_YesNo("Keep Training?", "y")
    
        decoder = rn.RvNeuralEnDeCoder(fn2)
        encoder = rn.RvNeuralEnDeCoder(fn1)
    
    
    
    
    noiseStrength = ri.Ask_Input_Float("Input Noise Strength.", 0.7)
    rf.Test_Encoder_Decoder(encoder, decoder, lstT,10, 
        "", noiseStrength)
    
    
#%% Test Section *********************************************************************
    

#endecoder = rn.RvNeuralEnDeCoder([784,50,10,50,784])
#ru.ImageFilesToAvi(endecoder.VideoImagePath, 
#                   "{}{}".format(endecoder.VideoImagePath, "EnDeCoder.avi") )
    
    


#%% Main Section ***************************************************************    
Main()

    