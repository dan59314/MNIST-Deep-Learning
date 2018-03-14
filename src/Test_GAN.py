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
import PlotFunctions as pltFn

from RvNeuNetworkMethods import EnumDropOutMethod as drpOut

#%%

AppendInputX = False

#%%%  Function Section

def Get_MiniBatches(lstTrain, numPerMiniBatch=500):
    randomStt = np.random.RandomState(1234) #int(time.time())) 
    randomStt.shuffle(lstTrain) #隨機打亂測試樣本順序
    
    # 在 500000 筆訓練資料內，從0開始，累加10取樣，
    # 共 500000/10 = 1000筆mini_trainingData[]
    # [0~10], [10~20], [20~30]......[49990~49999]
    samplingStep = numPerMiniBatch
    n_train = len(lstTrain)
    sampleNum = n_train
    mini_trainingData = [
        lstTrain[k:k+samplingStep] 
        for k in range(0, sampleNum, samplingStep)]
    return mini_trainingData

def Get_OneMiniBatch_Random(miniBatches):
    i = np.random.randint(len(miniBatches))
    #j = np.random.randint(len(miniBatches[0]))
    return miniBatches[i]

def DoContrast(oneInput):
    avg = np.linalg.norm(oneInput, ord=1)/len(oneInput) # abs(x1)+abs(x2)+.....
    #avg *= 0.7
    shape = oneInput.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            oneInput[i][j] = oneInput[i][j]>avg
    return oneInput
    

def Get_BestAccuracy(encoder, decoder, oneMiniBatch, refOutput, searchNum=0,
        doConstract=True):
    
    digitId=0    
    digit=[] 
    
    nOutput = refOutput.copy()
    if (doConstract): nOutput = DoContrast(nOutput) #要 constract, 變形比較大
    
    nMiniBatch = len(oneMiniBatch)
    if (searchNum<=0): searchNum = nMiniBatch
    checkNum = min(searchNum, nMiniBatch)-1
    bestAccur = 0.0
    stId = np.random.randint(0, nMiniBatch-checkNum)
    for i in range(stId, stId+checkNum):  
        
        inputX = oneMiniBatch[i][0].copy()
        #if doConstract: inputX = DoContrast(inputX) 不要 constract, 變形比較大
        
        accur = decoder.Get_Accuracy_EnDeCoder(inputX, nOutput)
        
        if accur>bestAccur:
            bestAccur = accur
            digitId = np.argmax(oneMiniBatch[i][1])
            digit = oneMiniBatch[i][0]
    
    return bestAccur, digitId, digit        
        



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

sampleNum = 500 # sampleNum 越小，變形能力越大
miniBatches = Get_MiniBatches(lstTrain, sampleNum)


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
    decoder = rn.RvNeuralEnDeCoder(fn1)
    encoder = rn.RvNeuralEnDeCoder(fn2)
        
    saveImgPath = "{}\\{}\\".format(decoder.VideoImagePath, "EncoderDecoder")  
    rfi.ForceDir(saveImgPath) #if not os.path.isdir(imgPath): os.mkdir(imgPath)
    rfi.Delete_Files(saveImgPath, [".jpg",".png"])

    saveImgPath1 = "{}\\{}\\".format(decoder.VideoImagePath, "EncoderDecoder1")  
    rfi.ForceDir(saveImgPath1) #if not os.path.isdir(imgPath): os.mkdir(imgPath)
    rfi.Delete_Files(saveImgPath1, [".jpg",".png"])
    
    bottleneckNeuNum = decoder.Get_InputNum()
    inputX = randomState.randn(encoder.Get_InputNum(),1)  
    outupt = inputX.copy()
    nOutput = inputX.copy()
    
    learnRate = 0.05
    # Load digit Images  lstTrain[0].shape = (pxls, label) = (784, 1)
    outputNeus = decoder.Get_OutputNum()
#    decoderLabels = rn.RvNeuralEnDeCoder.Load_DigitImages( \
#        ".\\Images\\Digits\\", outputNeus)
    toDigit = randomState.randint(0,10)
    digitChange = [toDigit]
    labelY = decoder.CreateLabelsY(10, toDigit)          
    pxlW = int(np.sqrt(len(inputX)))
    
    bestAccur = 0.0
    bestDigit = np.full((decoder.Get_OutputNum(),1), 0)
    bestRefDigit = bestDigit.copy()
    bestDigitId = -1
    bestDigitUpdated = False
    
    if (None!=decoder):   
        for iGen in range(1):
            miniBatches = Get_MiniBatches(lstTrain, sampleNum)
#            inputX = randomState.randn(encoder.Get_InputNum(),1)  
           
            encode = randomState.randn(decoder.Get_InputNum(),1)   
            encode = np.maximum(0.2, np.minimum(0.8, encode))
            inputX = decoder.Get_OutputValues(encode)  
            
            Epochs = 20
            weiDescriminator = 0.1
            stepWei = 0.0 # 0.3/Epochs
            for i in range(Epochs):            
                
                encode = encoder.Get_OutputValues(inputX) #(784x1)          
                              
                #output = decoder.Plot_Output(encode) #(784x1)    
                output = decoder.Get_OutputValues(encode)   
                
                
                # 找出最相近的 train data -------------------------------
                oneMiniBatch = Get_OneMiniBatch_Random(miniBatches)    
                accur, nDigit, digit = Get_BestAccuracy(\
                    encoder, decoder, oneMiniBatch, output, 50, i%5==0 )    
                
                bestDigitUpdated = False
                sBest = ""
                if (accur>bestAccur):
                    sBest = "<--Best"
                    bestDigitUpdated = True
                    bestAccur = accur
                    bestRefDigit = digit
                    bestDigit = output
                    bestDigitId = nDigit
                
                # 用新的 labelY 更新 decoder 權重 -------------------------------------------------
                encode1 = encoder.Get_OutputValues(digit)       
                weiDescriminator += stepWei   
                #weiDescriminator = randomState.randint(10,700)/1000
                nEncode = (encode*(1-weiDescriminator) + encode1*weiDescriminator)
                decoder.Update_LayersNeurons_Weights_Biases_OneInput( \
                   [ [nEncode, digit] ], 
                   learnRate, trainOnlyDigit=-1) #   
                
                # 更新 encoder 權重 -------------------------------------------------
#                    encoder.Update_LayersNeurons_Weights_Biases_OneInput( \
#                       [ [output, labelY] ], learnRate, trainOnlyDigit=-1) # 
#                
                    
                if i%1==0:    
                
                    print("Input->Output->Digit({}) : Match({:.3f}), Accuracy={:.3f}{}".
                          format(nDigit, decoder.Get_Accuracy_EnDeCoder(inputX, output),
                              decoder.Get_Accuracy_EnDeCoder(output, digit),sBest))              
                    if rfi.PathExists(saveImgPath):
                        imgFn = "{}vdoImg_{}_{}.png".format(saveImgPath, iGen, i)
                    else:
                        imgFn = ""               
                    
                    pltFn.Plot_Images(np.array([
                        inputX.transpose().reshape(pxlW,pxlW)*255,
                        output.transpose().reshape(pxlW,pxlW)*255,
                        digit.transpose().reshape(pxlW,pxlW)*255 ]),1,3, 
                        "Test EnDeCoder", imgFn)
            
                inputX = (inputX*0.6 + output*0.4)
                               
                
                if (nDigit!=toDigit): digitChange.append(nDigit)   
                toDigit = nDigit            
                labelY = decoder.CreateLabelsY(10, toDigit)
                                      
    
            print("Digit Change : {}".format(digitChange) )
            
            
            
            oneMiniBatch = Get_OneMiniBatch_Random(miniBatches)    
            _, bestDigitId, bestRefDigit = Get_BestAccuracy(encoder, decoder, 
                    lstTrain, bestDigit, doConstract=False)    
            if rfi.PathExists(saveImgPath):
                imgFn = "{}vdoImg_{}.png".format(saveImgPath1, iGen)
            else:
                imgFn = ""               
            pltFn.Plot_Images(np.array([bestRefDigit.transpose().reshape(pxlW,pxlW)*255,
               bestDigit.transpose().reshape(pxlW,pxlW)*255]),1,2, 
                "Best Digit = {}".format(bestDigitId), imgFn)

    
        aviFn = "{}{}".format(saveImgPath, "GAN.avi")
    #    print("AviFn = {}".format(aviFn))
        if ru.ImageFilesToAvi(saveImgPath, aviFn ):
            os.system(r'start ' + aviFn)

        aviFn = "{}{}".format(saveImgPath1, "GAN1.avi")
    #    print("AviFn = {}".format(aviFn))
        if ru.ImageFilesToAvi(saveImgPath1, aviFn ):
            os.system(r'start ' + aviFn)


