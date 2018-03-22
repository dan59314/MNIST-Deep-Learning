

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




#%% Global Variable

randomState = np.random.RandomState(int(time.time()))
    





#%%  Function Section

def Generate_FakeData(fakeDataNum):
    fn1 = ".\\NetData\\RvNeuralEnDeCoder_DigitGenerator.decoder"
    if not rfi.FileExists(fn1):
        fns, shortFns =  rfi.Get_FilesInFolder(".\\NetData\\", [".decoder"])
        if len(fns)>0:
            aId = min(1, len(fns)-1) #0 #ri.Ask_SelectItem("Select Decoder file", shortFns, 0)    
            fn1= fns[aId]               
    if (os.path.isfile(fn1)): 
        generator = rn.RvNeuralNetwork(fn1)    
        if None!=generator:
            return rn.RvNeuralDiscriminator.Create_FakeData_Generator(
                generator, fakeDataNum)        
    return []
#            [ 
#            tuple([ randomState.randn(len(aTuple[0]),1) , [fakeId] ]) 
#            for aTuple in TrueDigit_sets[0] ]
        
        

def Initial_PlotParams(pxls,nRow,nCol):    
    # 設定繪圖參數 ----------------------------------
#    nCol = 5 #int(np.sqrt(len(digitFigs)))
#    nRow = 2 #int(len(digitFigs)/nCol)+1
#    pxls = outputNum
    pxlW = int(np.sqrt(pxls))
    #pxls = pxlW*pxlW
    dpi = 72
    zoom = 6   # 28 pxl * zoom
    pltInchW =  pxlW/dpi*nCol * zoom
    digitFigs = [ np.full((pxlW,pxlW),0) for i in range(nRow*nCol)]
    
    return digitFigs, pxlW, pltInchW




sResult = ["X", "O"]

def Discriminate_Digits(net, lstT, plotDigit=True):
    
    threshold_True = 0.7
    
    sFakeReal = ['Fake', 'Real']
    
    nRow, nCol = 2, 5
    digitNum = nRow*nCol
    digitFigs, pxlW, pltInchW = Initial_PlotParams(len(lstT[0][0]),nRow,nCol)
    digitTitles = ["" for a in digitFigs]
    digitId=0
    
    # 隨機測試某筆數字 ----------------------------------------------    
    start = time.time() 
    
    sampleNum= min(len(lstT), 1000) # 不含繪圖，辨識 10000張，費時 1.3 秒，平均每張 0.00013秒
    plotNum = 30
    plotMod = int(sampleNum/plotNum) + 1
    correctNum=0    
    failNum = 0
    
    for i in range(0, sampleNum):
        
        doPlot = (i%plotMod == 0)
        aId = np.random.randint(0,len(lstT))
        outputY = np.max(net.Get_OutputValues(lstT[aId][0]))
        label = int(np.max(lstT[aId][1]))
        result = (outputY>threshold_True)*1
        if label==result: correctNum+=1
        else: 
            doPlot = (failNum<plotNum) 
            failNum+=1
            
        if doPlot and plotDigit:
#            s1 ="label={}, result={}  ( {:.3f} ) -> {} ".\
#                 format(sFakeReal[label], sFakeReal[result], 
#                        outputY, sResult[int(label==result)] )
#            rf.Plot_Digit(lstT[aId], result, label)
#            print(s1)
#            # 畫線
#            x1, y1 = [-1, 12], [1, 4]
#            x2, y2 = [1, 10], [3, 2]
#            plt.plot(x1, y1, x2, y2, marker = 'o')
#            plt.show()
            
            s1 ="{}({:.3f}) -> {}".\
                 format(sFakeReal[result], 
                        outputY,sResult[int(label==result)] )
            digitId = digitId % digitNum
            digitFigs[ digitId ] = np.array(lstT[aId][0]).transpose().reshape(pxlW,pxlW)*255
            digitTitles[ digitId ] = s1            
            digitId+=1
            
            if (digitId==digitNum):
              pltFn.Plot_Images(np.array(digitFigs),
                  nRow,nCol, digitTitles, "", pltInchW)  
              digitId = 0
    
    
    dt = time.time()-start
    
    accurRatio = correctNum/sampleNum
    print("\nAccuracy({:.3f}),  {}/{}(Correct/Total)".
          format(accurRatio, correctNum, sampleNum))    
    print("Elapsed(seconds)) : {:.3f} sec.\n".format(dt))    
    return accurRatio,dt



def Get_TrainData(lstTrain, sampleNum):
    
    print("\nPreparing ({}) Real/Fake Images....\n".format(sampleNum*2)  )
    print("")
    
    # 準備 測試資料 -------------------------------------------    
    MixDigit_set = rn.RvNeuralDiscriminator.Create_RealData(lstTrain, sampleNum)
    
    fake_set = rn.RvNeuralDiscriminator.Create_FakeData_RandomNoise(lstTrain, sampleNum)
    MixDigit_set += fake_set
    
    # Generate Fake Data
    genDigit_set = Generate_FakeData(sampleNum)
    MixDigit_set += genDigit_set
    randomState.shuffle(MixDigit_set)
    
    return MixDigit_set


#%% Main Section


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
    
    fnNetworkData = "{}{}_DSCMNT".format(path,rn.RvNeuralDiscriminator.__name__)   
    
    
    #Hyper pameters -------------------------------------------    
    loop = 10  # loop effect，10, 30 all above 0.95
    stepNum = 10  # stepNum effect,　10->0.9,  100->0.5
    learnRate = 0.1  # learnRate and lmbda will affect each other
    lmbda = 5.0     #add lmbda(Regularization) to solve overfitting 
    
    MixDigit_set = Get_TrainData(lstTrain, 1000)
    
    sTrain = "y"
    
    # Training ***********************************************************
    LoadAndTrain = ri.Ask_YesNo("Load exist model?", "n")        
    if LoadAndTrain:    
        fns, shortFns =  rfi.Get_FilesInFolder(".\\NetData\\", [".discriminator"])
        aId = ri.Ask_SelectItem("Select Discriminator file", shortFns, 0)    
        fn1= fns[aId]
        discriminator = rn.RvNeuralDiscriminator(fn1) 
        initialWeiBias = False
        sTrain = "n"
        
    else:                 
         # Create RvNeuralDiscriminator----------------------------------------------
        inputNeusNum = len(lstTrain[0][0])        
        lyrsNeus = [inputNeusNum, 400, 50] # [784,50,1]
        lyrsNeus = ri.Ask_Add_Array_Int(\
            "Input new layer Neurons num.", lyrsNeus, 50)        
#        bottleneckNeuNum = 784 #ri.Ask_Input_Integer("Input BottleNeck(Code) Layer Neurons num.", 10)
#        lyrsNeus.append(bottleneckNeuNum)
#        for nNeu in reversed(lyrsNeus[1:-1]):
#            lyrsNeus.append(nNeu)
        lyrsNeus.append(1)        
        discriminator = rn.RvNeuralDiscriminator(lyrsNeus)  # ([784,50,10])                
        initialWeiBias = True
        
    
        
           
    # Training ***********************************************************
    DoTrain = ri.Ask_YesNo("Do Training?", sTrain)    
    if DoTrain:    
        fnNetworkData = "{}_{}Lyr".format(fnNetworkData, len(discriminator.NeuralLayers))
               
        discriminator.DoPloatWeights = ri.Ask_YesNo("Plot Neurons Weights?", 'n')
            
        
        # Ask nmtivation  ------------------_----------
        enumActivation = ri.Ask_Enum("Select Activation method.", 
             nm.EnumActivation,  nm.EnumActivation.afReLU )
        discriminator.Set_EnumActivation(enumActivation)
        
        discriminator.NetEnableDropOut = ri.Ask_YesNo("Execute DropOut?", "n")
        if discriminator.NetEnableDropOut:
            enumDropOut = ri.Ask_Enum("Select DropOut Method.", 
            nm.EnumDropOutMethod,  drpOut.eoSmallActivation )
            rn.gDropOutRatio = ri.Ask_Input_Float("Input DropOut ratio.", rn.gDropOutRatio)
            discriminator.Set_DropOutMethod(enumDropOut, rn.gDropOutRatio)
        
        
        # Auto-Caculate proper hyper pameters ---
        DoEvaluate_ProperParams = ri.Ask_YesNo("Auto-Caculating proper hyper pameters?", "n")
        if DoEvaluate_ProperParams:
            loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_lmbda(
                    discriminator, discriminator.Train, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
            loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_learnRate(
                    discriminator, discriminator.Train, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
        else:      
            loop,stepNum,learnRate,lmbda = rf.Ask_Input_SGD(loop,stepNum,learnRate,lmbda)
        
        print( "Hyper pameters: Loop({}), stepNum({}), learnRatio({}), lmbda({})\n".format(loop,stepNum,learnRate,lmbda)  )
        
        sampleNum = min(20000, len(lstTrain))
        MixDigit_set = Get_TrainData(lstTrain, sampleNum)
        
        keepTraining = True
        while (keepTraining):    
            start = time.time()          
            # Start Training-
            discriminator.Train_Discriminator(\
                MixDigit_set, loop, stepNum, learnRate, lmbda,  initialWeiBias )
            initialWeiBias=False
            
            dT = time.time()-start           
            
            rf.Save_NetworkDataFile(discriminator, fnNetworkData, 
                    loop,stepNum,learnRate,lmbda, dT, ".discriminator")
        
            keepTraining = ri.Ask_YesNo("Keep Training?", "y")      
            if keepTraining:
               loop = ri.Ask_Input_Integer("loop: ", loop)                                        
      
    
    
    
    #檢驗 True Fake Digit -------------------------------------
    Discriminate_Digits(discriminator, MixDigit_set)

        
    
#%% Test Section *********************************************************************
    
#Test RvNeuralLayer --------------------------------------
#rvLyr = RvNeuralLayer([30,10])
#rvLyr1 = RvNeuralLayer(10, 20)

#測試 RvNeuralDiscriminator() overload Constructor ---------------    
#lyrsNeus = [784,50,10]
#rvNeuLyrs = RvNeuralDiscriminator.LayersNeurons_To_RvNeuralLayers(lyrsNeus)
#net = RvNeuralDiscriminator(rvNeuLyrs)    
#net = RvNeuralDiscriminator.Class_Create_LayersNeurons(lyrsNeus)    
#net = RvNeuralDiscriminator(lyrsNeus)    
    
    
    
#print(ac.EnumActivation.afReLU)  #EnumActivation.afReLU
#print(ac.EnumActivation.afReLU.name)   #afReLU
#print(ac.EnumActivation.afReLU.value)  #2
#print(type(ac.EnumActivation.afReLU))  #<enum 'EnumActivation'>
#print(type(ac.EnumActivation.afReLU.name)) #<class 'str'>
#print(type(ac.EnumActivation.afReLU.value))    #<class 'int'>




#%% Main Section ***************************************************************    
Main()
    
    
    