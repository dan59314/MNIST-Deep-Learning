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


from RvNeuNetworkMethods import EnumDropOutMethod as drpOut

#%%
AddNoise = False


randomState = np.random.RandomState(int(time.time()))


#%%%  Function Section

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



def Get_Models_FromFile(intialDiscriminator=False):
     
    # Prediction ------------------------------------------------
    fns, shortFns =  rfi.Get_FilesInFolder(".\\NetData\\", [".decoder"])
    aId = ri.Ask_SelectItem("Select Generator file", shortFns, 1)    
    fn1= fns[aId]
    
    fns, shortFns =  rfi.Get_FilesInFolder(".\\NetData\\", [".encoder"])
    aId = ri.Ask_SelectItem("Select Encoder file", shortFns, aId)    
    fn2= fns[aId]
    
    fns, shortFns =  rfi.Get_FilesInFolder(".\\NetData\\", [".discriminator"])
    aId = ri.Ask_SelectItem("Select Discriminator file", shortFns, aId)    
    fn3= fns[aId]
    
    
    generator, discriminator, encoder = None, None, None
    
    if (os.path.isfile(fn1) and os.path.isfile(fn2) and os.path.isfile(fn3)):            
        
        generator = rn.RvNeuralEnDeCoder(fn1)
        encoder = rn.RvNeuralEnDeCoder(fn2)
        discriminator = rn.RvNeuralDiscriminator(fn3)
        
    if intialDiscriminator:
        for lyr in discriminator.NeuralLayers:
            lyr.Initial_Neurons_Weights_Biases()

    
    return generator, discriminator, encoder





def Get_Models_New(lstTrain, intialDiscriminator=False):    

    path = "..\\TmpLogs\\"
    if not os.path.isdir(path): os.mkdir(path)           

    fnNetworkDataEDC = "{}{}_EDC".format(path,rn.RvNeuralEnDeCoder.__name__)  
    fnNetworkDataDscmnt = "{}{}_DSCMNT".format(path,rn.RvNeuralDiscriminator.__name__)   

    #Hyper pameters -------------------------------------------    
    loop = 10  # loop effect，10, 30 all above 0.95
    stepNum = 10  # stepNum effect,　10->0.9,  100->0.5
    learnRate = 0.1  # learnRate and lmbda will affect each other
    lmbda = 5.0     #add lmbda(Regularization) to solve overfitting 
    
    loop,stepNum,learnRate,lmbda = rf.Ask_Input_SGD(loop,stepNum,learnRate,lmbda)
        
    print("Training new generator, discriminator, encoder....")
    print("This will take a long while.........")
    print("")

    generator, discriminator, encoder = None, None, None
    
    
    print("Input EnDeCoder Layers Structure...........")
    
    # Create RvNeuralEnDeCoder----------------------------------------------
    inputNeusNum = len(lstTrain[0][0])
    lyrsNeus = [inputNeusNum, 200] # 512, 256,128]
    lyrsNeus = ri.Ask_Add_Array_Int("Input EnDeCoder new layer Neurons num.", lyrsNeus, 50)    
    bottleneckNeuNum = ri.Ask_Input_Integer("Input EnDeCoder BottleNeck(Code) Layer Neurons num.", 10)
    lyrsNeus.append(bottleneckNeuNum)
    for nNeu in reversed(lyrsNeus[1:-1]): lyrsNeus.append(nNeu)
    lyrsNeus.append(inputNeusNum)    
    endecoder = rn.RvNeuralEnDeCoder(lyrsNeus)
    endecoder.DoPloatWeights = False
    
    
    print("Input Discriminator Layers Structure...........")
    # Create RvNeuralDiscriminator----------------------------------------------
    inputNeusNum = len(lstTrain[0][0])        
    lyrsNeus = [inputNeusNum, 200] # [784,50,1]
    lyrsNeus = ri.Ask_Add_Array_Int("Input Discriminator new layer Neurons num.", lyrsNeus, 50)  
    lyrsNeus.append(1)        
    discriminator = rn.RvNeuralDiscriminator(lyrsNeus)  # ([784,50,10]) 
    discriminator.DoPloatWeights = False               
    
    # Prepare data for Discriminator -----------------------     
    MixDigit_set = Get_TrainData(lstTrain, 2000)   
    
    
    print("Training Generator, Encorder, Discriminator...........")    
    initialEnDeCoderWeights = True
    initialDiscriminatorWeights = True
    # Start Training-         
    keepTraining = True
    if (keepTraining):
    #while (keepTraining):
        start = time.time()  
        
        encoder, generator = endecoder.Build_Encoder_Decoder( \
          lstTrain, loop, stepNum, learnRate, lmbda, initialEnDeCoderWeights)
        initialEnDeCoderWeights = False
        
        if not intialDiscriminator: #如果是 intialDiscriminator,就不須 Train
            discriminator.Train_Discriminator(\
                    MixDigit_set, loop, stepNum, learnRate, lmbda,  initialDiscriminatorWeights )
            initialDiscriminatorWeights=False
          
        dT = time.time()-start            
        
        
        rf.Save_NetworkDataFile(endecoder, fnNetworkDataEDC, 
                loop,stepNum,learnRate,lmbda, dT, ".endecoder")        
        rf.Save_NetworkDataFile(encoder, 
                "{}_Encoder".format(fnNetworkDataEDC), 
                loop,stepNum,learnRate,lmbda, dT, ".encoder")
        rf.Save_NetworkDataFile(generator, 
                "{}_Decoder".format(fnNetworkDataEDC), 
                loop,stepNum,learnRate,lmbda, dT, ".decoder")        
        
        rf.Save_NetworkDataFile(discriminator, fnNetworkDataDscmnt, 
                loop,stepNum,learnRate,lmbda, dT, ".discriminator")
        
        #keepTraining = ri.Ask_YesNo("Keep Training?", "y")
    
    return generator, discriminator, encoder


#%%% Test Section






#%%% Main Section

def Main():

    path = "..\\TmpLogs\\"
    if not os.path.isdir(path): os.mkdir(path)           
        
    #Load MNIST ****************************************************************
        
    #Use mnist.pkl.gz(50000 data） Accuracy 0.96 
    #Use mnist_expanded.pkl.gz(250000 data） Accuracy 0.97 
    fn = "..\\data\\mnist.pkl.gz"  #".datamnist_expanded.pkl.gz"
    lstTrain, lstV, lstT =  mnist_loader.load_data_wrapper(fn)
    lstTrain = list(lstTrain)
    lstV = list(lstV)
    lstT = list(lstT)
    
    #addNoise = ri.Ask_YesNo("Add noise?", "n")
    #noiseStrength = 0.8 #ri.Ask_Input_Float("Input Noise Strength.", 0.0)    
    #randomState = np.random.RandomState(int(time.time()))
    
    fnNetworkData = "{}{}_GAN".format(path,rn.RvNeuralGAN.__name__)  
    
    #Hyper pameters -------------------------------------------    
    loop = 20  # loop effect，10, 30 all above 0.95
    stepNum = 10  # stepNum effect,　10->0.9,  100->0.5
    learnRate = 0.1  # learnRate and lmbda will affect each other
    lmbda = 5.0     #add lmbda(Regularization) to solve overfitting 
        
    
    loop = ri.Ask_Input_Integer("loop: ", loop) 
    reality = ri.Ask_Input_Float("Input Generate Reality Degree(0.0~1.0).", 0.95)      
    fakeDataNum = ri.Ask_Input_Integer("Input Generate data num.", 30)      
    fakeDataNoise = ri.Ask_YesNo("Fake Data of random Noise?", "n")
    trueDataFromMNIST = ri.Ask_YesNo("Real Data from MNIST?", "y")
    # 清除現有的 Discriminator weights, biases------------------------
    intialDiscriminator = ri.Ask_YesNo("Initial Discriminator Weights/Biases?", "n")
   
    
    # Training ***********************************************************
    # Ask DoTraining-
    LoadAndTrain = ri.Ask_YesNo("Load exist model and continue training?", "y")    
    
    if LoadAndTrain:    
       generator, discriminator, encoder = Get_Models_FromFile(intialDiscriminator)
       #initialWeiBias = False
    else:
       print( "Hyper pameters: Loop({}), stepNum({}), learnRatio({}), lmbda({})\n".format(loop,stepNum,learnRate,lmbda)  )
       generator, discriminator, encoder = Get_Models_New(lstTrain,intialDiscriminator)
       #initialWeiBias = True
       
       
            
    if (None==generator or None==discriminator or None==encoder):
        return
           
    
    gan = rn.RvNeuralGAN(generator, discriminator, encoder)
    
    
    # Training ***********************************************************    
    fnNetworkData = "{}_{}Lyr".format(fnNetworkData, len(gan.NeuralLayers))           
    gan.DoPloatWeights = False #ri.Ask_YesNo("Plot Neurons Weights?", 'n')
                   
    """       
    gan.NetEnableDropOut = ri.Ask_YesNo("Execute DropOut?", "y")
    if gan.NetEnableDropOut:
        enumDropOut = ri.Ask_Enum("Select DropOut Method.", 
        nm.EnumDropOutMethod,  drpOut.eoSmallActivation )
        rn.gDropOutRatio = ri.Ask_Input_Float("Input DropOut ratio.", rn.gDropOutRatio)
        gan.Set_DropOutMethod(enumDropOut, rn.gDropOutRatio)
    """
        
    
    
    print("Preparing data....\n")  
    if trueDataFromMNIST:
        trueSampleNum = 10000  # 數值越小，越能將形狀限制在一定形象上
        true_Data = rn.RvNeuralDiscriminator.Create_RealData(lstTrain, trueSampleNum)
        sRealData = "Real Data form MNIST ({}).".format(trueSampleNum)
    else:
        # 將形狀限制在 Digits 形象上
        imgPxls = lstTrain[0][0].shape[0]
        digitImages = rn.RvNeuralEnDeCoder.Load_DigitImages( ".\\Images\\Digits\\", imgPxls)
        true_Data = [ tuple([img, [[1]]]) for img in digitImages]
        n = int(fakeDataNum/len(true_Data)) # 讓 fakeData 和 trueData 數量相等
        for i in range(n-1): true_Data += [ tuple([img, [[1]]]) for img in digitImages]
        sRealData = "Real Data form Digit Files ({}).".format(len(digitImages))
        
        
    if fakeDataNoise:
        fake_Data = rn.RvNeuralDiscriminator.Create_FakeData_RandomNoise(lstTrain, fakeDataNum)
        sFakeData = "Fake Data form noise ({}).".format(len(fake_Data))
    else:
        fake_Data = rn.RvNeuralDiscriminator.Create_FakeData_Generator(generator, fakeDataNum, 0.0, 1.0)
        sFakeData = "Fake Data form Generator ({}).".format(len(fake_Data))
    
        
    
    # Start Training------------------------------------------------------------------     
    keepTraining = True
    while (keepTraining):   
        start = time.time()
        generator, discriminator, fake_Data, noMoreUpdate = \
          gan.Train_GAN(true_Data, fakeDataNum, loop, stepNum, learnRate, lmbda, fake_Data,
            atReality=reality)
    
        dT = time.time()-start
        
        rf.Save_NetworkDataFile(generator, 
                "{}_Encder".format(fnNetworkData), 
                loop,stepNum,learnRate,lmbda, dT, ".encoder")
        rf.Save_NetworkDataFile(generator, 
                "{}_Generator".format(fnNetworkData), 
                loop,stepNum,learnRate,lmbda, dT, ".decoder")
        rf.Save_NetworkDataFile(discriminator, 
                "{}_Discriminator".format(fnNetworkData), 
                loop,stepNum,learnRate,lmbda, dT, ".discriminator")
        
        if noMoreUpdate:
            print( "No more update. ")
            break
        else:
            keepTraining = ri.Ask_YesNo("Keep Training?", "y")
            if keepTraining:
               loop = ri.Ask_Input_Integer("loop: ", loop)     
        
    print("Reality : {}".format(reality))
    print(sRealData)
    print(sFakeData)



#%%
    
Main()

