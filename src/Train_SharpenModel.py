

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


#%% Function Session:

def PreProcessBlur(inputX):
    pxls = len(inputX)
    pxlW = int(np.sqrt(pxls))
    pxls = pxlW*pxlW
    nInput = np.array(inputX).reshape(pxlW,pxlW)
    for iy in range(1,pxlW-1):
        for ix in range(1, pxlW-1):
            tVal = 0.0
            for ay in range(iy-1,iy+2):
                for ax in range(ix-1, ix+2):
                    tVal+=nInput[ay][ax]
            avg = tVal/9
            nInput[iy][ix] = avg
    return nInput.reshape(pxls,1)
 
       


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
    
    fnNetworkData = "{}{}_NetData".format(path,rn.RvNeuralEnDeCoder.__name__)   
    
    
    #Hyper pameters -------------------------------------------    
    loop = 10  # loop effect，10, 30 all above 0.95
    stepNum = 10  # stepNum effect,　10->0.9,  100->0.5
    learnRate = 0.1  # learnRate and lmbda will affect each other
    lmbda = 5.0     #add lmbda(Regularization) to solve overfitting 
    
    
    endecoder = None
    
    # Training ***********************************************************
    # Ask DoTraining-
    LoadAndTrain = ri.Ask_YesNo("Load exist model and continue training?", "n")    
    
    if LoadAndTrain:       
        digitIdOnly = -1 
        fns, shortFns =  rfi.Get_FilesInFolder(".\\NetData\\", [".endecoder"])
        aId = ri.Ask_SelectItem("Select Decoder file", shortFns, 0)    
        fn1= fns[aId]
        endecoder = rn.RvNeuralEnDeCoder(fn1)       
        
        initialWeights = False
        
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
        lyrsNeus = [inputNeusNum, 256, 100]
        lyrsNeus = ri.Ask_Add_Array_Int(\
            "Input new layer Neurons num.", lyrsNeus, 50)
        
#        bottleneckNeuNum = ri.Ask_Input_Integer(\
#            "Input BottleNeck(Code) Layer Neurons num.", 10)
#        lyrsNeus.append(bottleneckNeuNum)
#        for nNeu in reversed(lyrsNeus[1:-1]):
#            lyrsNeus.append(nNeu)
        lyrsNeus.append(inputNeusNum)
        
        digitIdOnly = -1
        digitIdOnly = ri.Ask_Input_Integer("Build only Digit (0~9, -1 = All Digits): ", digitIdOnly)
        
        
        #net = RvNeuralEnDeCoder( 
        #   RvNeuralEnDeCoder.LayersNeurons_To_RvNeuralLayers(lyrsNeus))
        #net = RvNeuralEnDeCoder.Class_Create_LayersNeurons(lyrsNeus)
        endecoder = rn.RvNeuralEnDeCoder(lyrsNeus)  # ([784,50,10])
        
        initialWeights = True
        
    
    
    randomState = np.random.RandomState(int(time.time()))
    
    
    
    # Ask nmtivation  ------------------_----------
    enumActivation = ri.Ask_Enum("Select Activation method.", 
         nm.EnumActivation,  nm.EnumActivation.afSigmoid)
    for lyr in endecoder.NeuralLayers:
        lyr.Set_EnumActivation(enumActivation)
    
    endecoder.Motoring_TrainningProcess = rn.Debug

    endecoder.NetEnableDropOut = ri.Ask_YesNo("Execute DropOut?", "n")
    if endecoder.NetEnableDropOut:
        enumDropOut = ri.Ask_Enum("Select DropOut Method.", 
        nm.EnumDropOutMethod,  drpOut.eoSmallActivation )
        rn.gDropOutRatio = ri.Ask_Input_Float("Input DropOut ratio.", rn.gDropOutRatio)
        endecoder.Set_DropOutMethod(enumDropOut, rn.gDropOutRatio)
    
    monitoring = ri.Ask_YesNo("Watch training process?", "y")
    endecoder.Motoring_TrainningProcess = monitoring
         
    
    # Auto-Caculate proper hyper pameters ---
    DoEvaluate_ProperParams = ri.Ask_YesNo("Auto-Caculating proper hyper pameters?", "n")
    if DoEvaluate_ProperParams:
        loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_lmbda(
                net, endecoder.Train, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
        loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_learnRate(
                net, endecoder.Train, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
    else:      
        loop,stepNum,learnRate,lmbda = rf.Ask_Input_SGD(loop,stepNum,learnRate,lmbda)
    
    print( "Hyper pameters: Loop({}), stepNum({}), learnRatio({}), lmbda({})\n".format(loop,stepNum,learnRate,lmbda)  )


    start = time.time()   
    # Start Training-
    if (None!=endecoder):
        sampleNum = 5000
        print( "\nPrepare ({}) label Images....\n".format(sampleNum)  )
        sampleEndId = len(lstTrain)-sampleNum
        sId = randomState.randint(sampleEndId)
        lstNew = []
        for aTuple in lstTrain[sId:sId+sampleNum]:
            # [模糊, 銳利]
            lstNew.append(tuple([ PreProcessBlur(aTuple[0]), aTuple[0] ]) )
            
        keepTraining = True
        while (keepTraining):
          encoder, decoder = endecoder.Build_Encoder_Decoder_AssignOutputY( \
            lstNew, loop, stepNum, learnRate, lmbda, initialWeights, digitIdOnly)
          keepTraining = ri.Ask_YesNo("Keep Training?", "y")
    
    dT = time.time()-start
    
    
    rf.Save_NetworkDataFile(endecoder, fnNetworkData, 
            loop,stepNum,learnRate,lmbda, dT, ".endecoder")

    fn1 = rf.Save_NetworkDataFile(encoder, 
            "{}_Encoder".format(fnNetworkData), 
            loop,stepNum,learnRate,lmbda, dT, ".encoder")
    fn2 = rf.Save_NetworkDataFile(decoder, 
            "{}_Decoder".format(fnNetworkData), 
            loop,stepNum,learnRate,lmbda, dT, ".decoder")

    
    decoder = rn.RvNeuralEnDeCoder(fn2)
    encoder = rn.RvNeuralEnDeCoder(fn1)
    
    
    noiseStrength = ri.Ask_Input_Float("Input Noise Strength.", 0.7)
    rf.Test_Encoder_Decoder(encoder, decoder, lstT,10, 
        "", noiseStrength)
    
    
#%% Test Section *********************************************************************
#fn = "..\\data\\mnist.pkl.gz"  #".datamnist_expanded.pkl.gz"
#lstTrain, lstV, lstT =  mnist_loader.load_data_wrapper(fn)
#lstTrain = list(lstTrain)
#lstV = list(lstV)
#lstT = list(lstT)    
#     
#pxls = len(lstTrain[0][0])
#pxlW = int(np.sqrt(pxls))
#pxls = pxlW*pxlW
#pltFn.Plot_Images(np.array([lstTrain[0][0].transpose().reshape(pxlW,pxlW)*255,
#        PreProcessInput(lstTrain[0][0]).transpose().reshape(pxlW,pxlW)*255]),1,2, "Test Blur")
    
#endecoder = rn.RvNeuralEnDeCoder([784,50,10,50,784])
#ru.ImageFilesToAvi(endecoder.VideoImagePath, 
#                   "{}{}".format(endecoder.VideoImagePath, "EnDeCoder.avi") )
    
    


#%% Main Section ***************************************************************    
Main()

    