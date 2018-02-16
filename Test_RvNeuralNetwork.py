

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
from RvActivationCost import EnumDropOutMethod as drpOut
import PlotFunctions as pltFn




#%%  


def Add_ConvLayer(lyrObjs, lyrsNeus,  inputNeusNum, cnvLyrSId):
    nChannel=1
    inputW = np.sqrt(inputNeusNum)
    inputH = inputW
    inputShape =[inputW,inputH,nChannel]     
    
    # filterW,H 越小，產生的神經元越多，精度越高 ------------------
    filterW       = 5
    filterH       = 5
    filterNum     = 10
    filterShape = np.asarray([filterW,filterH, nChannel, filterNum])     
    # filterStride 越小，重疊區越大，產生的神經元越多，精度越高 -----------
    filterStride = 1
    cnvLyr = rn.RvConvolutionLayer(
        inputShape, # eg. [pxlW, pxlH, Channel]
        filterShape, # eg. [pxlW, pxlH, Channel, FilterNum], 
        filterStride)        
    lyrObjs.append(cnvLyr)
    cnvLyrSId += 1
    lyrsNeus.insert(cnvLyrSId, cnvLyr.Get_NeuronsNum())
    return lyrsNeus, cnvLyrSId, cnvLyr


def Main():
    #Load MNIST ****************************************************************
    
    #使用 mnist.pkl.gz(50000筆） 準確率 0.96 
    #使用 mnist_expanded.pkl.gz(250000筆） 準確率提高到 0.97 
    fn = ".datamnist.pkl.gz"  #".datamnist_expanded.pkl.gz"
    lstTrain, lstV, lstT =  mnist_loader.load_data_wrapper(fn)
    lstTrain = list(lstTrain)
    lstV = list(lstV)
    lstT = list(lstT)
    
    
    # 主程式區 *********************************************************************
    path = ".\\TmpLogs\\"
    if not os.path.isdir(path):
        os.mkdir(path)           
    
    fnNetworkData = "{}{}_NetData".format(path,rn.RvNeuralNetwork.__name__)   
    fnNetworkData1 = ""
    
    # Training ***********************************************************
    # Ask DoTraining-
    DoTraining = ri.Ask_YesNo("要執行SGD()訓練嗎?", "y")
    if DoTraining:             
        """
        [784,50,10], loop=100, 0.9725
        """
         # 建立 RvNeuralNetWork----------------------------------------------
        inputNeusNum = len(lstTrain[0][0])
        lyr1NeuNum = 50
        lyr2NeuNum = len(lstTrain[0][1])
        
        lyrsNeus = [inputNeusNum, lyr1NeuNum]
        lyrsNeus = ri.Ask_Add_Array_Int("輸入增加新層神經元數", lyrsNeus, lyr1NeuNum)
        lyrsNeus.append(lyr2NeuNum)
        
        #net = rn.RvNeuralNetwork( \
        #   rn.RvNeuralNetwork.LayersNeurons_To_RvNeuralLayers(lyrsNeus))
        #net = rn.RvNeuralNetwork.Class_Create_LayersNeurons(lyrsNeus)
        #net = rn.RvNeuralNetwork(lyrsNeus)  # ([784,50,10])
        
        cnvLyrSId = 0
        lyrObjs=[]
        
        
        # 先加入RvConvolutionLayer--------------------------------   
        EnableCovolutionLayer = ri.Ask_YesNo("要加上 ConvolutionLayer 嗎?", "y")
        if EnableCovolutionLayer:             
          lyrsNeus, cnvLyrSId, cnvLyr = Add_ConvLayer(lyrObjs, lyrsNeus, inputNeusNum, cnvLyrSId)
          
          
        # 建立 Layer Object array -------------------------------
        for iLyr in range(cnvLyrSId, len(lyrsNeus)-1):
            lyrObjs.append( rn.RvNeuralLayer([lyrsNeus[iLyr], lyrsNeus[iLyr+1]]) )
        net = rn.RvNeuralNetwork(lyrObjs)
        
        
        # Ask Activation  ------------------_----------
        enumActivation = ri.Ask_Enum("選取 Activation 類別.", 
             ac.EnumActivation,  ac.EnumActivation.afReLU )
        for lyr in net.NeuralLayers:
            lyr.ClassActivation, lyr.ClassCost = \
            ac.Get_ClassActivation(enumActivation)
        
        net.Motoring_TrainningProcess = rn.Debug
    
        net.NetEnableDropOut = ri.Ask_YesNo("要執行DropOut嗎?", "n")
        if net.NetEnableDropOut:
            enumDropOut = ri.Ask_Enum("選取 DropOut Method.", 
            ac.EnumDropOutMethod,  drpOut.eoSmallActivation )
            rn.gDropOutRatio = ri.Ask_Input_Float("輸入DropOut ratio.", rn.gDropOutRatio)
            net.Set_DropOutMethod(enumDropOut, rn.gDropOutRatio)
        
        monitoring = ri.Ask_YesNo("是否監看訓練過程?", "y")
        net.Motoring_TrainningProcess = monitoring
        
        #輸入網路參數 -------------------------------------------    
        loop = 1  # loop影響正確率不大，10和 30都在 9成以上
        stepNum = 10 # stepNum越大，正確率越低　10->0.9,  100->0.5
        learnRate = 0.1  # 調整 learnRate 和 lmbda 的參數，會互相影響結果
        lmbda = 5.0     # 加上 lmbda(Regularization) 可以解決 overfitting 問題        
            
        
        #是否要計算最適合的值 ---
        DoEvaluate_ProperParams = ri.Ask_YesNo("要自動預估適合的網路參數嗎?", "n")
        if DoEvaluate_ProperParams:
            loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_lmbda(
                    net, net.Train, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
            loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_learnRate(
                    net, net.Train, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
        else:      
            loop,stepNum,learnRate,lmbda = rf.Ask_Input_SGD(loop,stepNum,learnRate,lmbda)
        
        print( "網路訓練參數  : Loop({}), stepNum({}), learnRatio({}), lmbda({})\n".format(loop,stepNum,learnRate,lmbda)  )
    
    
        start = time.time()   
        # 開始網路訓練-
        net.Train(lstTrain, loop, stepNum, learnRate, lstV, lmbda )
        dT = time.time()-start
        
        # 劃出 Neurons Weights
        iLyr=0
        print("Layer({}) : ".format(iLyr))
        aId = np.random.randint(0, len(net.NeuralLayers[iLyr].NeuronsWeights) )
        net.NeuralLayers[iLyr].Plot_NeuronsWeights([aId,aId+1])
        
        # 存出網路參數檔案
        sConvLyr, sDropOut = "", ""
        sConvLyr = "_CnvLyr" if ([]!=net.Get_ConvolutionLayerID()) else ""        
        sDropOut = "_DropOut_{}".format(net.NetEnumDropOut.name) \
            if net.NetEnableDropOut else ""        
        fnNetworkData1 = "{}{}{}_{:.2f}.txt".format(fnNetworkData, 
            sConvLyr, sDropOut, net.BestAccuracyRatio) 
        net.Save_NetworkData(fnNetworkData1)    
        

        lyrsWeis = net.Get_LayersNeuronsWeightsNum()
        weiMultiple = 0
        for wei in lyrsWeis:
            weiMultiple += wei
            
        s1 = "\n日期 : {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + \
             "{}, {}\n".format(sConvLyr, sDropOut) + \
             "LayersName = {}\nLayersNeurons = {}\n".format( 
                net.Get_LayersName(), net.Get_LayersNeurons() ) +\
             "LayersWeightsNum = {}\n".format(lyrsWeis) + \
             "WeightsMultiples = {}\n".format(weiMultiple ) + \
             "網路訓練參數 : Loop({}), stepNum({}), learnRatio({:.4f}), + \
               lmbda({:.4f})\n".format(loop,stepNum,learnRate,lmbda)  + \
             "準確度 : 最差({}), 最好({})\n".format(net.WorstAccuracyRatio, net.BestAccuracyRatio)  + \
             "耗費時間(秒) : {:.3f} sec.\n".format( dT ) 
        print(s1)
        
        if dT>600: # 超過十分鐘，把結果記下來
            fn = ".\Log_RvNeuralNetwork.txt"
            f = open(fn,'a') 
            f.write(s1)
            f.close() 
            print("log added to... \"{}\"".format(fn))
    
    # 預測 ------------------------------------------------
    if (not os.path.isfile(fnNetworkData1)): 
        fnNetworkData1= ".\\{}_NetData_DontDelete.txt". \
            format(rn.RvNeuralNetwork.__name__)
    
    
    # Ask DoPredict----------------------------------------------------
    DoPredict=True
#    if DoTraining: DoPredict = ri.Ask_YesNo("要做數字辨識嗎?", "n")   
#    else: DoPredict = True

    if DoPredict:          
        rn.Debug_Plot = True #ri.Ask_YesNo("要繪出數字嗎?", "n") 
    #    Predict_Digits(net, lstT)
        pltFn.Predict_Digits_FromNetworkFile(fnNetworkData1, lstT, rn.Debug_Plot)    
    
    
    
#%% 測試 *********************************************************************
    
#測試 RvNeuralLayer --------------------------------------
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




#%% 主程式 ***************************************************************    
Main()

    
    
    
    