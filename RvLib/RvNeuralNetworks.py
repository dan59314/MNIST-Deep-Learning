# -*- coding: utf-8 -*-
# coding: utf-8 　　　←表示使用 utf-8 編碼，加上這一行遇到中文註解才能編譯成功

#%%
"""
Created on Thu Feb  8 14:36:42 2018

Python is really a good tool to quickly code and run.
Start Learn Python : 2018/1/10 
Code This : 2018/2/8


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


RvNeuralNetworks.py

 1. Stochastic gradient descent updating Weights, Biases
 2. Cost Classes :
     Cost_Quadratic, Cost_CrossEntropy 
 3. RvNeuralLayer, RvCovolutionLayer, RvPoolingLayer
  
"""


#%% Libraries

# Standard library----------------------------------------------
#import sys
#sys.path.append('..//data')
#sys.path.append('..//RvLib')
import os, inspect
import json
from enum import Enum
import time
#from datetime import datetime
from datetime import datetime, timedelta
import random


# Third-party libraries------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
#matplotlib.use("Agg")
#import matplotlib.animation as animation

import cv2
from PIL import Image, ImageDraw, ImageFont


# prvaite libraries---------------------------------------------
import RvNeuNetworkMethods as nm
from RvNeuNetworkMethods import EnumDropOutMethod as drpOut
from RvNeuNetworkMethods import EnumCnvFilterMethod as fm
from RvNeuNetworkMethods import EnumActivation as af
from RvNeuNetworkMethods import EnumPoolingMethod as pm
import RvMiscFunctions as rf
import RvFileIO as rfi
import RvMediaUtility as ru
import RvGVariable as rg
import PlotFunctions as pltFn


#%%  Directive Definintion ----------------------------------------------------
Debug = True
SaveVideo = Debug

gDropOutRatio = 0.5
gDropOutPerLoop = 2


gFilterShareWeights = True # Share Weight 精度降低，最高 0.916


#%%
def Get_PlotWidthInch(imgPxlW, imgColNum, PlotImageWidth = 720):
    
    MinImageWidth = min(PlotImageWidth, 240)
    ScreenDPI = 60
    PixelEnlarge = 5
    return max(MinImageWidth, 
        min(PlotImageWidth, imgPxlW*imgColNum*PixelEnlarge))/ScreenDPI



#%%
class Enum_WeiBiasInit(Enum):
    iwbStdError = 1
    iwbLarge = 2
    iwbNormal = 3
    
#    def __int__(self): #輸出 1, 2, 3 而不是iwbSmall.....
#        return str(self.value)
#    
#    def __str__(self): #輸出iwbSmall.....
#        return str(self.name)

     
    
    
#%%  ***************************************************************************

class RvNeuralLayer(object):    
    """=============================================================
    Constructor:
    ============================================================="""
    def __init__(self,  *args):         
        self.__Initial_Members()
        nArgs = len(args)
        if nArgs>0:            
            if isinstance(args[0], list):  
                self.__Create_1Args(*args)
            elif isinstance(args[0], int):
                if nArgs in [2,3,4]: #(inputN, neurN, enumAct, enumCoste)
                    self.__Create_4Args(*args)
                else:
                    print("Need InputNum and NeuronNum")
            elif isinstance(args[0], str): # lyrfileName
                self.__Create_File(*args)              
            elif isinstance(args[0], object): # lyrObj
                self.__Create_RefLyrObj(*args)                
            else:
                print("Need InputNum and NeuronNum")        
        else:
            print("Need InputNum and NeuronNum")    
        
        
        
    def __Create_File(self, filename):
        if not rfi.FileExists(filename): return #os.path.isfile(filename): return  
        self.Load_Neurons_Parameters(filename)
        
    def __Create_RefLyrObj(self, refLyrObj):
        #if (refLyrObj.__class__.__name__ == RvNeuralLayer.__name__):
            inputNum, neuroNum = refLyrObj.Get_InputNum(), refLyrObj.Get_NeuronsNum()
            #self = RvNeuralLayer(inputNum, neuroNum )
            self.Update_LayerData(inputNum, neuroNum, 
                refLyrObj.NeuronsWeights,
                refLyrObj.NeuronsBiases)
            self.__Assign_Members(refLyrObj)
        
        
    def __Create_1Args(self, inOutNums,
                 enumActivation=af.afReLU):  
        if len(inOutNums)<2: 
            print("Need InputNum and NeuronNum") 
            while (len(inOutNums)<2): inOutNums.append(1)   
        self.__Initial_Neurons_Weights_Biases0(inOutNums[0], inOutNums[1])   
        self.Set_EnumActivation(enumActivation )
            
            
    def __Create_4Args(self, inputNeuronsNum, oneLyrNeuronsNum,
                 enumActivation=af.afReLU):   
        self.__Initial_Neurons_Weights_Biases0(inputNeuronsNum, oneLyrNeuronsNum)  
        self.Set_EnumActivation(enumActivation)  
#        if Debug:          
#            print("Biase = {}\n".format(self.NeuronsBias) )
#            print("Weight = {}\n".format(self.NeuronsWeights) )
#            print("Cost Class : \"{}\"".format(self.ClassCost.__name__)) 
#            print("Activation Class : \"{}\"".
#                format(self.ClassActivation.__name__))  
            
    
    """=============================================================
    Static:
    ============================================================="""
    
    
    """=============================================================
    Private :
    ============================================================="""  
    def __Assign_Members(self, frLyrObj):
        self.DoUpdateNeuronsWeightsBiases = frLyrObj.DoUpdateNeuronsWeightsBiases
        self.RandomState = np.random.RandomState(int(time.time())) 
        #self.NeuronsBiases = [0.0] 
        #self.NeuronsWeights = np.zeros((len(self.NeuronsBiases),1))
#        self.__DoDropOut =frLyrObj.
#        self.__DropOutRatio =frLyrObj.
#        self.__EnumDropOut = frLyrObj.
        self.Set_EnumActivation(frLyrObj.__EnumActivation)
            
            
    def __Initial_Members(self):     
        self.DoUpdateNeuronsWeightsBiases = True
        self.RandomState = np.random.RandomState(int(time.time())) 
        self.NeuronsBiases = [0.0] 
        self.NeuronsWeights = np.zeros((len(self.NeuronsBiases),1))
        self.__DoDropOut = False
        self.__DropOutRatio = gDropOutRatio
        self.__EnumDropOut = drpOut.eoNone
        self.ClassActivation,self.ClassCost = None, None
        self.Set_EnumActivation(af.afReLU)
        
    def __Initial_Neurons_Weights_Biases0(self,inputNeuronsNum, oneLyrNeuronsNum):  
        """
        eg. [L1, L2, L3] = [784, 50, 10] 
        Biases = [ [ # L2, 50 Neurons, 50 x 1 array-----------------------------
                     [b1], 
                     ....   # L2, 50x1 bias
                     [b50] ],   
    
                   [ # L3, 10 Neurons, 10 x 1 array ----------------------------
                     [b1],     
                     ...    # L3, 10x1 bias
                     [b10] ] ] 
        Weights = [ [ # L2, 50 Neurons, 50 x 784 array--------------------------
                     [w1 ... w784], 
                     ....   # L2, 50x784 weight
                     [w1 ... w784] ],   
    
                    [ # L3, 10 Neurons, 10 x 50 array --------------------------
                     [w1 ... w50],     
                     ...    # L3, 10x50 weight
                     [w1 ... w50] ] ] 
        """
        self.NeuronsBiases = self.RandomState.randn(oneLyrNeuronsNum, 1)  
        self.NeuronsWeights = \
          self.RandomState.randn(oneLyrNeuronsNum, inputNeuronsNum)/np.sqrt(inputNeuronsNum)
        
#        if Debug:
#          if self.Get_InputNum()>1:
#            print("\n{} :\n  Input({}), Neurons({})".
#              format(self.__class__.__name__, self.Get_InputNum(), self.Get_NeuronsNum() ))
        
            
    def __Read_Neurons_Parameters(self, pF):     
        data = json.load(pF)      
        self.Set_LayerData(data)
        
    def __Write_Neurons_Parameters(self, pF, lyrIndex=0):     
        data = self.Get_LayerData(lyrIndex)       
        json.dump(data, pF)
        
    """=============================================================
    Public :
    ============================================================="""
    # ----------------------------------------------------------
    # Get Functions 
    # ----------------------------------------------------------
    def Get_InputNum(self):
        return len(self.NeuronsWeights[0])
    
    def Get_NeuronsNum(self):        
        return len(self.NeuronsBiases)
    
    def Get_NeuronsValuesZ(self, prLyrInputX):
        # outputZ = Sum(wi*xi)+b
        return np.dot(self.NeuronsWeights, np.asarray(prLyrInputX)) + self.NeuronsBiases
    
    def Get_NeuronsActivations(self, lyrNeusZs):
        return self.ClassActivation.activation(lyrNeusZs)
    
    def Get_LayerData(self, lyrIndex=-1):
        return {                
                "ClassName": self.__class__.__name__,
                "LayerIndex": lyrIndex,
                "InputNum": self.Get_InputNum(),
                "NeuronsNum": self.Get_NeuronsNum(),
                "NeuronsWeights": [w.tolist() for w in self.NeuronsWeights],
                "NeuronsBiases": [b.tolist() for b in self.NeuronsBiases],
                "EnumActivationValue": self.__EnumActivation.value,
                #"EnumDropOutValue": self.__EnumDropOut.value, Layer不須儲存
#                "ClassCost": str(self.ClassCost.__name__),
                } 
    
    # ----------------------------------------------------------
    # Set Functions 
    # ---------------------------------------------------------- 
    def Set_LayerData(self, data):          
        self.__Initial_Members()            
#        self.__Initial_Neurons_Weights_Biases0(data["InputNum"],data["NeuronsNum"])
#        # 要將 NeuronsWeights, NeuronsBiases 轉為 array, 否則會在 np.dot() 時候，
#        # 因為 list 和 array dot() 造成運算速度緩慢
#        self.NeuronsWeights = np.asarray(data["NeuronsWeights"])
#        self.NeuronsBiases = np.asarray(data["NeuronsBiases"] )       
        
        self.Update_LayerData(data["InputNum"],data["NeuronsNum"], \
           data["NeuronsWeights"], data["NeuronsBiases"])
         
        enumValue = data["EnumActivationValue"]
        enumActivation = af( enumValue )
        self.Set_EnumActivation(enumActivation)       
            
#        enumValue = data["EnumDropOutValue"]
#        enumDropOut = nm.EnumDropOutMethod( enumValue )
#        self.__EnumDropOut = enumDropOut
            
#        cls = getattr(sys.modules[__name__], data["ClassCost"])  
#        self.ClassCost = cls             
                    

    def Set_DropOut(self, doDropOut, \
            enumDropOut=drpOut.eoSmallActivation, \
            ratioDropOut=0.5):   
        self.__EnumDropOut = enumDropOut 
        self.__DoDropOut = doDropOut
        if doDropOut:  #  dropOut both a, d(a)
            self.__DropOutRatio = ratioDropOut
        
    def Set_EnumActivation(self, enumActivation):
        self.__EnumActivation = enumActivation 
        self.ClassActivation,self.ClassCost = \
            nm.Get_ClassActivation(enumActivation)
    
    def Get_EnumActivation(self):
        return self.__EnumActivation
        
    def Create_LayerObj(self, refLyrObj):
         return RvNeuralLayer(refLyrObj)
         
         
    
    
    # ----------------------------------------------------------
    # Main Functions 
    # ----------------------------------------------------------      
    def Initial_Neurons_Weights_Biases(self):
        self.__Initial_Neurons_Weights_Biases0(self.Get_InputNum(), self.Get_NeuronsNum())
        
    # Initial a structure as like self.NeuronsBiases ---------------
    def Create_ArrayOf_NeuronsBiases(self, initialValue):
        return np.full(self.NeuronsBiases.shape, initialValue)
    
    # Initial a structure as like self.NeuronsBiases ---------------
    def Create_ArrayOf_NeuronsWeights(self, initialValue):
        return np.full(self.NeuronsWeights.shape, initialValue)
    
    def Caculate_dCost_OutputLayer(self, oneLyrNeusZs, oneLyrNeusActvs, \
            oneLyrNeusLabels,preLyrNeusActvs):
        # caculate last output layer derivation d(cost)/d(wei), d(cost)/d(bias)-----------
        # error = dC/dA               
        oneLyrNeusErrs = (self.ClassCost).errorValue(oneLyrNeusZs, \
            oneLyrNeusActvs, oneLyrNeusLabels)         
        # d(cost)/d(bias) = dC/dA * dA/dZ * dZ/dB = error * dA/dZ,  
        # dZ/dB=d(wx+b)/db=1
        # dC/dB = error * (dA/dZ, 留待稍後再乘)
        oneLyrNeus_dCost_dBiases = oneLyrNeusErrs       
        # d(Cost)/d(wei) = dC/dA * dA/dZ * dZ/dW = error*a * dA/dZ,  
        # dZ/dW=d(wx+b)/dw=x, x=前一層的a
        # dC/dW = error*a * (dA/dZ, 留待稍後乘)
        oneLyrNeus_dCost_dWeis = np.dot(np.asarray(oneLyrNeusErrs), \
            np.asarray(preLyrNeusActvs).transpose()) 
        return oneLyrNeusErrs, oneLyrNeus_dCost_dWeis, oneLyrNeus_dCost_dBiases    
    
    
    def Caculate_dCost_HiddenLayer(self, preLyrNeusActvs, curLyrNeusZs, 
                nxtLyrObj, nxtLyrNeusErrs):
        
#        if not self.DoUpdateNeuronsWeightsBiases:
#            #curLyrNeusErrs, curLyrNeus_dCost_dWeis, curLyrNeus_dCost_dBiases
#            return np.zeros(curLyrNeusZs.shape), \
#                np.zeros(curLyrNeusZs.shape), np.zeros(curLyrNeusZs.shape)
                
                
        # The hidden layers backPropagationagation -------------------------------------------
        curLyrNeus_dA_dZ = self.ClassActivation.derivation(curLyrNeusZs)         
        
        if self.__DoDropOut: # dropOut both a, d(a)
            # Caculate DropOutMask every time get a better result-----------------
            #curLyrNeus_dA_dZ = Get_NoneDropOutValues(curLyrNeus_dA_dZ, self.__DropOutRatio) 
            curLyrNeus_dA_dZ = nm.ClassDropOut.Get_NonDropOutValues( self.__EnumDropOut,
                curLyrNeus_dA_dZ, self.__DropOutRatio)
            
        # error = dC/dA
        # dC/dB = error * dA/dZ
        # dC/dW = error*a * dA/dZ
        # 當前層的誤差 = 下一層誤差反推回來的誤差權重 * 當前層的激活值偏微分
        """
        curLyrNeusErrs = nxtLyrNeusWeis.T 。 nxLyrNeusErrs
        [[err0]          [[w0..wn]         [[err0]
         ..n個Neus..      ..m個neus..        ..m個neus..
         [err_n] ]        [w0..wn] ].T      [err_m] ]   
        """
        nxtLyrNeusErrs = np.asarray(nxtLyrNeusErrs)
        curLyrNeus_dA_dZ = np.asarray( curLyrNeus_dA_dZ)
        preLyrNeusActvs = np.asarray(preLyrNeusActvs)
        
        curLyrNeusErrs = np.dot(nxtLyrObj.NeuronsWeights.transpose(),  
            nxtLyrNeusErrs) * curLyrNeus_dA_dZ
                   
        # 當前層的 cost(bias) = 當前層的 error
        curLyrNeus_dCost_dBiases = curLyrNeusErrs
        # 當前層的 cost(weight) = 前一層的 a*當前層error
        curLyrNeus_dCost_dWeis = np.dot(curLyrNeusErrs, 
            preLyrNeusActvs.transpose())
        return curLyrNeusErrs, curLyrNeus_dCost_dWeis, curLyrNeus_dCost_dBiases
        
          
          
    def Caculate_Neurons_Z_Activations(self, oneInput):        
#        if Debug: print("Input = {}\n".format(oneInput))  
        if len(oneInput) != len(self.NeuronsWeights[0]): return        
        oneLyrNeurosZs = self.Get_NeuronsValuesZ(oneInput)   
        oneInput = self.Get_NeuronsActivations(oneLyrNeurosZs)
        
        if self.__DoDropOut:  # a和 d(a) 都要 dropOut  
            # 每次重算 DropOutMask 效果比較好-----------------
            oneInput = nm.ClassDropOut.Get_NonDropOutValues( self.__EnumDropOut,
                oneInput, self.__DropOutRatio)
            #oneInput = Get_BigValues(oneInput, self.__DropOutRatio)
            #oneInput = Get_NoneDropOutValues(oneInput, self.__DropOutRatio) 
            
        return oneLyrNeurosZs, oneInput   
                    
    def Update_LayerData(self, inputNum, lyrNeursNum, lyrNeursWeights, lyrBiases):
        self.__Initial_Neurons_Weights_Biases0(inputNum, lyrNeursNum)        
        # 要將 NeuronsWeights, NeuronsBiases 轉為 array, 否則會在 np.dot() 時候，
        # 因為 list 和 array dot() 造成運算速度緩慢
        self.NeuronsWeights = np.asarray(lyrNeursWeights)
        self.NeuronsBiases = np.asarray(lyrBiases)
        
    def Plot_OneNeuronWeights(self, neuId, neuWeis, lyrId=-1, saveFn=""):
        if (lyrId<0): sLyr=""
        else: sLyr="Lyr({})".format(lyrId)
        weiNum = len(neuWeis)
        pxlW = int(np.sqrt(weiNum))
        pxls = pxlW*pxlW
        nRow, nCol = 1, 1
        pltInchW = Get_PlotWidthInch(pxlW, nCol)
        aMin = np.min(neuWeis)
        aMax = np.max(neuWeis)
        #aMul = (x-aMin)/(aMax-aMin)*255
        aMul = 255/(aMax-aMin)
        pltFn.Plot_Images([np.array(
            (neuWeis[:pxls].transpose()-aMin)*aMul).reshape(pxlW,pxlW)],
            nRow,nCol, ["{} NeuWei({})[{}]: {:.4f}~{:.4f}".format(
                sLyr, neuId,weiNum,aMin,aMax)],saveFn, pltInchW)
        print("")
        
        
    def Plot_NeuronsWeights(self, atNeuroID=[0,1]):
        if len(self.NeuronsWeights)<=0: return      
        sId = min( len(self.NeuronsWeights)-1,  max(atNeuroID[0], 0))
        eId = min(atNeuroID[1], len(self.NeuronsWeights))
        iNeuro = sId
        # Plot
        print("{}:".format(self.__class__.__name__))
        for aNeuro in self.NeuronsWeights[sId:eId]:
            self.Plot_OneNeuronWeights(iNeuro, aNeuro )
            iNeuro +=1 
      
    
    # ----------------------------------------------------------
    # File IO Functions ------
    # ----------------------------------------------------------
                
    def Load_Neurons_Parameters(self, filename):
        if not rfi.FileExists(filename): return #os.path.isfile(filename): return
        pf = open(filename, "r")
        self.__Read_Neurons_Parameters(pf)
        pf.close()
        
        
    def Save_Neurons_Parameters(self, filename=""):         
        if ""==filename: 
            filename="{}_{}.txt".format(self.__class__.__name__, 
                      self.ClassActivation.__name__)            
        pf = open(filename, "w")
        self.__Write_Neurons_Parameters(pf)
        pf.close()        
        
    



#%%  ***************************************************************************
        
class RvConvolutionLayer(RvNeuralLayer):
    
    """=============================================================
    Constructor:
    ============================================================="""
    def __init__(self,  *args):  
        super().__init__() #加上此，產生所有 parent 的 members 
        self.__Initial_Members()
        nArgs = len(args)
        if nArgs>0:            
            if isinstance(args[0], list):  # inputShape
                self.__Create_Normal(*args) 
            elif isinstance(args[0], str):  
                self.__Create_File(*args)  
            elif isinstance(args[0], object): # lyrObj
                self.__Create_RefLyrObj(*args)        
             
        
        
    def __Create_Normal(self, inputShape=[1,1,1], # eg. [pxlW, pxlH, Channel]
                 filterShape=[1,1,1,1], # eg. [pxlW, pxlH, Channel, FilterNum], 
                 filterStride=1, # eg 1, 每次移動 1 pxl
                 enumFilterMethod=fm.fmAverageSum,
                 enumActivation=af.afReLU):  
        self.__Initial_Members()
        assert (len(inputShape)>0) and (len(filterShape)>0)
        # Channel 必須相等
        assert inputShape[2] == filterShape[2] 
                
#        self.__Initial_Neurons_Weights_Biases0(inOutNums[0], inOutNums[1])   
        self.EnumFilterMethod = enumFilterMethod
        # 呼叫 private members --------------------------
        self.InputShape = inputShape
        self.FilterShape  = filterShape
        self.FilterStride = filterStride
        self.__Initial_Neurons_Weights_Biases0(inputShape, filterShape, filterStride)
        
        # access parent private memebers -> self._parentClassName__privateMembers
        self.Set_EnumActivation(enumActivation)  
            #### Load network file and create RvConvolutionLayer ----------------------------------------------
    
    
    def __Create_File(self, filename):
        self.Load_Neurons_Parameters(filename)
    
    
    def __Create_RefLyrObj(self, refLyrObj):
        #if (refLyrObj.__class__.__name__  == RvConvolutionLayer.__name__ ):
            self.Update_LayerData(
               refLyrObj.InputShape,
               refLyrObj.FilterShape,
               refLyrObj.FilterStride,
               refLyrObj.NeuronsWeights,
               refLyrObj.NeuronsBiases)
            self.__Assign_Members(refLyrObj)
            
            
            
    """=============================================================
    Static:
    ============================================================="""
    @staticmethod
    def Get_StepsNum_Convolution(inputNum, filterNum, filterStride):
        filterStride = min(filterStride, filterNum)
#        steps=0
#        sId=0
#        eId=filterNum
#        while (eId<=inputNum):
#            steps+=1
#            sId+=filterStride
#            eId=sId+filterNum
#        return steps
        ovlp=filterNum-filterStride
        n=(inputNum-ovlp)
        div=(filterNum-ovlp)
        steps=int(n/div)
        rest=int(n)%div
        return steps,rest 
    
    @staticmethod
    def Get_StepsNum_Pooling(inputNum, poolNum, poolStride=0):
        poolStride = min(poolStride, poolNum)
        ovlp=poolNum-poolStride
        return int((inputNum-ovlp)/(poolNum-ovlp))
    
    @staticmethod
    #### 將 input上下左右補齊，以符合 filter,stride 整數取樣----------------
    def Padding_Input(input_2D, filterShapeWH, filterStride, fillValue=0.0):
        assert( ( len(input_2D.shape)==2 ) and (len(filterShapeWH)==2))
#        assert( filterStride <= min(filterShapeWH[0], filterShapeWH[1]) )
        filterStride = min(filterStride, filterShapeWH[0],filterShapeWH[1])
        """
        input_2D=(28x28)      filterShapeWH=[5, 5]
          [ [0..PxlW],
            [0..pxlW]
            ..pxlY 個..
            [0..pxlW] ]
        """
        pxlsY = len(input_2D)        
        pxlsX = len(input_2D[0])
        stepX,restX = RvConvolutionLayer.Get_StepsNum_Convolution( \
            pxlsX,filterShapeWH[0],filterStride)
        stepY,restY = RvConvolutionLayer.Get_StepsNum_Convolution( \
            pxlsY,filterShapeWH[1],filterStride)
        
        padLeft=int(restX/2)   
        padRight=restX-padLeft   
        padTop=int(restY/2)    
        padBottom=restY-padTop  
        """ 
        array([[1, 1],
               [2, 2],
               [3, 3]])
        np.insert(a, 1, 5, axis=1)
        >>>array([[1, 5, 1],
                  [2, 5, 2],
                  [3, 5, 3]])
        """        
        if Debug: 
          if (filterShapeWH[0]>1):
            print("\nShape Before Padding = {}".format(input_2D.shape))
            print("input({}x{}), filter({}x{}), stride({})".
                  format(pxlsX,pxlsY,filterShapeWH[0],filterShapeWH[1],filterStride))
            print("inputStepsXY=({},{}), restXY=({},{})".format(stepX,stepY,restX,restY))
            
#      
        if (padLeft>0): # 每筆 input_2D[]左邊增加一筆
            # np.insert(array, 插入index, 插入值， 第幾維度)
            for ix in range(padLeft): 
                input_2D=np.insert(input_2D, 0, fillValue, axis=1)
        if (padRight>0):
            #  np.append(array, 插入值，第幾個維度)
            for ix in range(padRight): 
                input_2D=np.insert(input_2D, len(input_2D[0]), fillValue, axis=1)
        if (padTop>0):
            for iy in range(padTop): 
                input_2D=np.insert(input_2D, 0, fillValue, axis=0)
        if (padBottom>0):
            for iy in range(padBottom): 
                input_2D=np.insert(input_2D, len(input_2D), fillValue, axis=0)
        if Debug: print("Shape After Padding = {}".format(input_2D.shape))
        return input_2D
    
    
    
    """=============================================================
    Private :
    ============================================================="""    
    def __Initial_Members(self):  # override     
        # 呼叫 private members --------------------------
#        self._RvNeuralLayer__Initial_Members() 
#        self.DoUpdateNeuronsWeightsBiases = True
        """ 
        下面改在 parent.__Initial_Members() 內呼叫
        self.RandomState = np.random.RandomState(int(time.time()))
        self.NeuronsBiases = [] 
        self.NeuronsWeights = np.zeros((len(self.NeuronsBiases),0))   
        """
        self.FilterShareWeights = gFilterShareWeights
        # 用來儲存此層所有 Neurons 對應到前一層的那些 Neurons Index
        self.PreNeuronIDsOfWeights = np.zeros((len(self.NeuronsBiases),0))
        
#        if Debug: 
#            print("\nCall Parent Public Func : RvNeuralLayer.Get_InputNum() = {}".
#                   format(RvNeuralLayer.Get_InputNum(self)))
#            print("Class({}).__Initial_Members".format(self.__class__.__name__))
        
        
    def __Initial_Neurons_Weights_Biases0(self,
            inputShape, # eg. [pxlW, pxlH, Channel]
            filterShape, # eg. [pxlW, pxlH, Channel, FilterNum], 
            filterStride): # eg 1, 每次移動 1 pxl):  
        # 指定每個 Neurons 對應到前層的那些 NeuronsfilterStride = min(filterStride, filterShape[0],filterShape[1])
        # 計算需要多少個神經元, eg. image 4,4, filter 3x3, stride=1,
        filterStride = min(filterStride,filterShape[0], filterShape[1] )
        
        self.InputShape = np.asarray(inputShape)
        self.FilterShape = np.asarray(filterShape)
        self.FilterStride = int(filterStride)        
        
        filterPxls = self.FilterShape[0]*self.FilterShape[1] 
        stepX,restX = RvConvolutionLayer.Get_StepsNum_Convolution(
                self.InputShape[0], self.FilterShape[0], self.FilterStride)
        stepY,restY = RvConvolutionLayer.Get_StepsNum_Convolution(
                self.InputShape[1], self.FilterShape[1], self.FilterStride)
        nNeurons = stepX * stepY     
         
        if Debug: 
          if (inputShape[0]>1):
            print("\n{}:".format(self.__class__.__name__))
            print("  Input({}), Neurons({})".format(self.Get_AllInputNum(), nNeurons))
            print("  InputShape \t= {}\n  FilterShape \t= {}\n  FilterStride \t= {}".
                  format(inputShape,filterShape,filterStride))
            print("  inputStepsXY \t= ({},{}), restXY = ({},{})".format(stepX,stepY,restX,restY))
                    
            
        # 為每個神經元 Assign 對應的 Weights, Biases   
        if self.FilterShareWeights:
            # 相同 Filter 共用一組 Weights
            self.NeuronsBiases = self.RandomState.randn(nNeurons, 1)  
            self.NeuronsWeights = \
              self.RandomState.randn(1, filterPxls)/np.sqrt(filterPxls) 
        else:
            self._RvNeuralLayer__Initial_Neurons_Weights_Biases0(
                filterPxls, nNeurons)
        
        # 指定每個神經元的 Weights 對應到前一層的那些 Neurons    
        self.__Set_PreNeuronIDsOfWeights(inputShape, filterShape, filterStride)
       
    
    
    #逐一將每個 Weights 所對應的前一層 neuronID 指定好
    def __Set_PreNeuronIDsOfWeights(self, 
            inputShape, # eg. [pxlW, pxlH, Channel]
            filterShape, # eg. [pxlW, pxlH, Channel, FilterNum], 
            filterStride):
       
        filterStride = min(filterStride,filterShape[0], filterShape[1] )
        filterPxls = filterShape[0]*filterShape[1] 
        stepX,restX = RvConvolutionLayer.Get_StepsNum_Convolution(inputShape[0], filterShape[0], filterStride)
        stepY,restY = RvConvolutionLayer.Get_StepsNum_Convolution(inputShape[1], filterShape[1], filterStride)
        nNeurons =  stepX * stepY  
        
        self.PreNeuronIDsOfWeights = np.full((nNeurons, filterPxls), 0)            
            
        nNeurs = len(self.PreNeuronIDsOfWeights)
        nWeis = len(self.PreNeuronIDsOfWeights[0])        
#        if Debug: 
#            print("\nNeus_x_Weis({}x{})".format(nNeurs, nWeis))
        assert(nNeurs == nNeurons)
        assert(nWeis == filterPxls)
        
        neurId = 0
        for stpY in range(stepY):
          curPxlY = stpY*filterStride
          for stpX in range(stepX):
            curPxlX = stpX*filterStride
            stPxlId = curPxlY*inputShape[0]+curPxlX   
            
            # 開始取得此 Neurons.Weis的前一層neur id         
            weiId=0
            for iy in range(filterShape[1]):
              pxlId = stPxlId + iy*inputShape[0]
              for ix in range(filterShape[0]):
                self.PreNeuronIDsOfWeights[neurId][weiId]=pxlId
                pxlId+=1
                weiId+=1                  
            neurId+=1
        
#        if Debug: 
#            print("PreNeuronIDsOfWeights[:{}]=\n{}".
#                  format(filterShape[0], \
#                         self.PreNeuronIDsOfWeights[:filterShape[0]]))
    
    # 以 NeuronID 到前一層尋找對應的 Input
    def __Get_BelongedInputByNeuronIDs(self, prLyrInputX, iNeuro):
        nInput = [ prLyrInputX[preNeuId] 
            for preNeuId in self.PreNeuronIDsOfWeights[iNeuro]]            
        return np.asarray(nInput)
        
    # 找出下層中，連接到此層 aNeurId 的所有wei
    def __Get_BelongedWeightedErrorByNeuronsIDs(self, aNeurId, nxtLyrObj, nxtLyrNeusErrs):
       assert( None!=nxtLyrObj )
       # 下一層如果是 convolution Layer，則必須萃取新的 Errors
       if (nxtLyrObj.__class__.__name__ == RvConvolutionLayer.__name__):
         nNeusErrs = 0.0
         # 瀏覽所有的 nxtLyrObj.PreNeuronIDsOfWeights[][],
         # 如果發現有 和 aNeuId 一樣的，則將 Errs[iNeuro] 加到 nNeusErrs[]
         for iNeuro in range(len(nxtLyrObj.PreNeuronIDsOfWeights)):             
           blFind_aNeurId = False
           for iWei in range(nxtLyrObj.PreNeuronIDsOfWeights[iNeuro]):
             if (True==blFind_aNeurId):
               break
             if (aNeurId==nxtLyrObj.PreNeuronIDsOfWeights[iNeuro][iWei]):
               blFind_aNeurId = True
               err = nxtLyrNeusErrs[iNeuro]
               if self.FilterShareWeights:
                 wei = nxtLyrObj.NeuronsWeights[0][iWei]
               else:
                 wei = nxtLyrObj.NeuronsWeights[iNeuro][iWei]
               nNeusErrs += (wei*err)
               #跳出 for iWei 迴圈               
         return nNeusErrs
     
       else:
         return 0.0
       
     
    def __Caculate_dCost_HiddenLayer_ConvolutionLayer(self, preLyrNeusActvs, curLyrNeusZs, 
                nxtLyrObj, nxtLyrNeusErrs):
        
#        if not self.DoUpdateNeuronsWeightsBiases:
#            #curLyrNeusErrs, curLyrNeus_dCost_dWeis, curLyrNeus_dCost_dBiases
#            return np.zeros(curLyrNeusZs.shape), \
#                np.zeros(curLyrNeusZs.shape), np.zeros(curLyrNeusZs.shape)
        
        # 非輸出層的 backPropagationagation，從倒數第二層開始 -------------------------------------------
        curLyrNeus_dA_dZ = self.ClassActivation.derivation(curLyrNeusZs)         
        
        if self._RvNeuralLayer__DoDropOut: # a和 d(a) 都要 dropOut 
            # 每次重算 DropOutMask 效果比較好-----------------
            #curLyrNeus_dA_dZ = Get_NoneDropOutValues(curLyrNeus_dA_dZ, self.__DropOutRatio) 
            curLyrNeus_dA_dZ = nm.ClassDropOut.Get_NonDropOutValues( 
                self._RvNeuralLayer__EnumDropOut,
                curLyrNeus_dA_dZ, self._RvNeuralLayer__DropOutRatio)
            
        # error = dC/dA
        # dC/dB = error * dA/dZ
        # dC/dW = error*a * dA/dZ
        # 當前層的誤差 = 下一層誤差反推回來的誤差權重 * 當前層的激活值偏微分
        """
        curLyrNeusErrs = nxtLyrNeusWeis.T 。 nxLyrNeusErrs
        [[err0]          [[w0..wn]         [[err0]
         ..n個Neus..      ..m個neus..        ..m個neus..
         [err_n] ]        [w0..wn] ].T      [err_m] ]   
        """
        nxtLyrNeusErrs = np.asarray(nxtLyrNeusErrs)
        
        if (nxtLyrObj.__class__.__name__ == RvConvolutionLayer.__name__):
            curLyrNeusErrs = []
            # 逐一算出此層所有 Neurnos的 Error = nxtLyrWeisOf(iNero)。nxtLyrsErrsOf(iNero)
            for iNeuro in range(len(self.NeuronsBiases)):
                # 找出下層中所有 PreNeuronIDsOfWeightsWeights[]中，只要有 iNeuro的
                # 就計算加總 Error
                curNeuErr = self.__Get_BelongedWeightedErrorByNeuronsIDs( 
                  iNeuro, nxtLyrObj, nxtLyrNeusErrs)
                curLyrNeusErrs.append(curNeuErr)                  
            curLyrNeusErrs = np.asarray(curLyrNeusErrs )
        else: # 下一層是 RvNeuralLayer            
            curLyrNeusErrs = np.dot(nxtLyrObj.NeuronsWeights.transpose(),  
              nxtLyrNeusErrs) * curLyrNeus_dA_dZ
                   
        # 當前層的 cost(bias) = 當前層的 error    
        curLyrNeus_dCost_dBiases = curLyrNeusErrs
        # CnvLyr不加bias
        # curLyrNeus_dCost_dBiases = np.zeros(curLyrNeusErrs.shape)
        
        # 當前層的 cost(weight) = 前一層的 a*當前層error
        if self.FilterShareWeights:
            curLyrNeus_dCost_dWeis = np.full(self.PreNeuronIDsOfWeights[0].shape,0.0)
            for iNeuro in range(len(self.PreNeuronIDsOfWeights)):
                preAs = self.__Get_BelongedInputByNeuronIDs(preLyrNeusActvs, iNeuro)
                err_x_As = curLyrNeusErrs[iNeuro] * preAs
                curLyrNeus_dCost_dWeis =  np.array(curLyrNeus_dCost_dWeis)+(err_x_As.transpose()[0])
            curLyrNeus_dCost_dWeis /= len(self.PreNeuronIDsOfWeights) 
        else:            
            curLyrNeus_dCost_dWeis = []
            for iNeuro in range(len(self.PreNeuronIDsOfWeights)):
                preAs = self.__Get_BelongedInputByNeuronIDs(preLyrNeusActvs, iNeuro)
                err_x_As = curLyrNeusErrs[iNeuro] * preAs
                curLyrNeus_dCost_dWeis.append(err_x_As.transpose()[0])
            
        curLyrNeus_dCost_dWeis = np.asarray(curLyrNeus_dCost_dWeis)
        
        return curLyrNeusErrs, curLyrNeus_dCost_dWeis, curLyrNeus_dCost_dBiases
    
     
       
    """=============================================================
    Public :
    ============================================================="""
    # ----------------------------------------------------------
    # Get Functions 
    # ----------------------------------------------------------    
    def Get_ChannelNum(self):
        return self.InputShape[2]
        
    def Get_InputWidth(self):
        return self.InputShape[0]
        
    def Get_InputHeight(self):
        return self.InputShape[1]    
        
    def Get_AllInputNum(self):
        return int(self.InputShape[0]*self.InputShape[1])
    
    def Get_FilterWidth(self):
        return self.FilteShape[0]
        
    def Get_FilterHeight(self):
        return self.FilteShape[1]
    
    def Get_ConvolutionValue(self, inputX, filterWeis):
        # CnvLyr不加bias
        return np.dot(filterWeis, inputX) # + bias
        
        
    def Get_NeuronsValuesZ(self, prLyrInputX):
        # outputZ = Sum(wi*xi)+b
        Zs = []
        # 逐一計算 每列 Weight*input 相加
        for iNeuro in range(len(self.PreNeuronIDsOfWeights)):
            nInputX = self.__Get_BelongedInputByNeuronIDs(prLyrInputX, iNeuro)    
            if self.FilterShareWeights:
              z = np.dot(self.NeuronsWeights[0], nInputX) + self.NeuronsBiases[iNeuro]    
            else:
              z = np.dot(self.NeuronsWeights[iNeuro], nInputX) + self.NeuronsBiases[iNeuro]    
            # CnvLyr不加bias
            #z = self.Get_ConvolutionValue(nInputX, self.NeuronsWeights[iNeuro]) 
            Zs.append(z)
        Zs = np.asarray(Zs)
        # if Debug: print("\nZs[:5]={}, Shape={}".format(Zs[:5], Zs.shape))
        return Zs
    
    def Get_NeuronsActivations(self, lyrNeusZs):
        # 呼叫 parent 的 public functions
        return self.ClassActivation.activation(lyrNeusZs)    
    
    
    def Get_LayerData(self, lyrIndex=-1):
        # 呼叫 parent 的 public functions
        data = RvNeuralLayer.Get_LayerData(self, lyrIndex)
        # dictionary.update() 新增 dictionary 值
        data.update({
                'InputShape' : self.InputShape.tolist(), # array 無法 儲存，須轉成list
                'FilterShape' : self.FilterShape.tolist(),
                'FilterStride' : self.FilterStride,        
                'FilterShareWeights' : self.FilterShareWeights,
                })
        return dict(data)
    
    
    
    
    # ----------------------------------------------------------
    # Set Functions 
    # ---------------------------------------------------------- 
    def Set_LayerData(self, data):          
        self.__Initial_Members()       
        
        
        key1, key2, key3 = "InputShape", "FilterShape", "FilterStride"
        if (key1 in data) and (key2 in data) and (key3 in data):        
            self.Update_LayerData(
               np.asarray(data["InputShape"]),
               np.asarray(data["FilterShape"]), 
               data["FilterStride"], data["NeuronsWeights"], data["NeuronsBiases"])
         
        key = "FilterShareWeights"
        if (key in data): 
            self.FilterShareWeights = data["FilterShareWeights"]
            gFilterShareWeights = self.FilterShareWeights
        if (len(self.NeuronsWeights)!=len(self.NeuronsBiases)):
            self.FilterShareWeights = True
            gFilterShareWeights = self.FilterShareWeights
            
        
        enumValue = data["EnumActivationValue"]
        enumActivation = af( enumValue )
        self.Set_EnumActivation(enumActivation)     
    
    
    
    def Create_LayerObj(self, refLyrObj):
         return RvConvolutionLayer(refLyrObj)
    
    # ----------------------------------------------------------
    # Main Functions 
    # ----------------------------------------------------------    
    def Initial_Neurons_Weights_Biases(self):
        self.__Initial_Neurons_Weights_Biases0(
                self.InputShape, self.FilterShape, self.FilterStride)
       
    
    def Update_LayerData(self, inputShape, filterShape, filterStride, 
            lyrNeursWeights, lyrBiases):
        
        self.__Initial_Neurons_Weights_Biases0(
                inputShape, filterShape, filterStride)        
        # 要將 NeuronsWeights, NeuronsBiases 轉為 array, 否則會在 np.dot() 時候，
        # 因為 list 和 array dot() 造成運算速度緩慢
        self.NeuronsWeights = np.asarray(lyrNeursWeights)
        self.NeuronsBiases = np.asarray(lyrBiases)
        
      
    def Caculate_dCost_HiddenLayer(self, preLyrNeusActvs, curLyrNeusZs, 
                nxtLyrObj, nxtLyrNeusErrs):        
        # 如果下層是RvNeuralLayer，則以 RvNeuralLayer.Caculate_dCost_HiddenLayer()計算
        if (self.__class__.__name__ == RvNeuralLayer.__name__):
            return RvNeuralLayer.Caculate_dCost_HiddenLayer(
              self, preLyrNeusActvs, curLyrNeusZs, nxtLyrObj, nxtLyrNeusErrs)
        # 如果是 RvConvolution Layer, 則另外計算
        elif (self.__class__.__name__ == RvConvolutionLayer.__name__):     
            return self.__Caculate_dCost_HiddenLayer_ConvolutionLayer(
              preLyrNeusActvs, curLyrNeusZs, nxtLyrObj, nxtLyrNeusErrs)
        else:            
            return RvNeuralLayer.Caculate_dCost_HiddenLayer(
              self, preLyrNeusActvs, curLyrNeusZs, nxtLyrObj, nxtLyrNeusErrs)
          
            
            
    # oneInput 是前一層所有神經元的 input，在此必須依據 PreNeuronIDsOfWeights[]
    # 篩選出真正所屬的和對應的 Weights
    def Caculate_Neurons_Z_Activations(self, oneInput):     
#        if Debug:
#            print("\noneInputShape={}".format(oneInput.shape))
        #if len(self.NeuronsWeights[0])<0: return oneInput,oneInput        
        oneLyrNeurosZs = self.Get_NeuronsValuesZ(oneInput)   
        oneInput = self.Get_NeuronsActivations(oneLyrNeurosZs)
        
        if self._RvNeuralLayer__DoDropOut:  # a和 d(a) 都要 dropOut  
            # 每次重算 DropOutMask 效果比較好-----------------
            oneInput = nm.ClassDropOut.Get_NonDropOutValues( 
                self._RvNeuralLayer__EnumDropOut,
                oneInput, self._RvNeuralLayer__DropOutRatio)
             
#        if Debug:
#            print("NeurosZsShape={}".format(oneLyrNeurosZs.shape))
#            print("oneInputShape={}".format(oneInput.shape))
        return oneLyrNeurosZs, oneInput   
    
    # ----------------------------------------------------------
    # File IO Functions ------
    # ----------------------------------------------------------
    
            
    def __Read_Neurons_Parameters(self, pF):     
        data = json.load(pF)      
        self.Set_LayerData(data)
        
    def Load_Neurons_Parameters(self, filename):
        if not rfi.FileExists(filename): return #os.path.isfile(filename): return
        pf = open(filename, "r")
        self.__Read_Neurons_Parameters(pf)
        pf.close()
    
    





#%%  ***************************************************************************

class RvPollingLayer(RvConvolutionLayer):
 
    """=============================================================
    Constructor:
    ============================================================="""
    def __init__(self,  *args):  
        super().__init__() #加上此，產生所有 parent 的 members 
        self.__Initial_Members()
        nArgs = len(args)
        if nArgs>0:            
            if isinstance(args[0], list):  # inputShape
                self.__Create_Normal(*args) 
            elif isinstance(args[0], str):  
                self.__Create_File(*args)  
            elif isinstance(args[0], object): # lyrObj
                self.__Create_RefLyrObj(*args)        
    
    
        super().__init__(*args) #加上此，產生所有 parent 的 members 
        
        
    def __Create_Normal(self, inputShape=[1,1,1], # eg. [pxlW, pxlH, Channel]
                 poolSize=2,  
                 enumPoolingMethod=pm.pmMaxValue,
                 enumActivation=af.afReLU):  
        self.__Initial_Members()
        assert ( len(inputShape)>0)       
                
#        self.__Initial_Neurons_Weights_Biases0(inOutNums[0], inOutNums[1])   
        self.EnumPoolingMethod = enumPoolingMethod
        # 呼叫 private members --------------------------
        self.InputShape = inputShape
        self.__Initial_Neurons_Weights_Biases0(inputShape )
        
        # access parent private memebers -> self._parentClassName__privateMembers
        self.Set_EnumActivation(enumActivation)     

    def __Create_File(self, filename):
        if not rfi.FileExists(filename): return #os.path.isfile(filename): return# 看網路參數檔案存在否
        self._RvConvolutionLayer__Create_File(filename)
        
    def __Create_RefLyrObj(self, refLyrObj):
        self._RvConvolutionLayer__Create_RefLyrObj(refLyrObj)
 
    def __Initial_Members(self):  # override     
        # 呼叫 private members --------------------------
        self._RvConvolutionLayer__Initial_Members() 
        RvConvolutionLayer.DoUpdateNeuronsWeightsBiases = True
        # 用來儲存此層所有 Neurons 對應到前一層的那些 Neurons Index
        RvConvolutionLayer.PreNeuronIDsOfWeights = np.zeros((len(self.NeuronsBiases),0))
        
#        if Debug: 
#            print("\nCall Parent Public Func : RvNeuralLayer.Get_InputNum() = {}".
#                   format(RvNeuralLayer.Get_InputNum(self)))
#            print("Class({}).__Initial_Members".format(self.__class__.__name__))
        
        
    def __Initial_Neurons_Weights_Biases0(self,
            inputShape, # eg. [pxlW, pxlH, Channel]
            filterShape, # eg. [pxlW, pxlH, Channel, FilterNum], 
            filterStride): # eg 1, 每次移動 1 pxl):  
        # 指定每個 Neurons 對應到前層的那些 NeuronsfilterStride = min(filterStride, filterShape[0],filterShape[1])
        # 計算需要多少個神經元, eg. image 4,4, filter 3x3, stride=1,
       
        # 為每個神經元 Assign 對應的 Weights, Biases   
        #self._RvNeuralLayer__Initial_Neurons_Weights_Biases0(
#                filterPxls, nNeurons)
        
        # 指定每個神經元的 Weights 對應到前一層的那些 Neurons    
        #self.__Set_PreNeuronIDsOfWeights(inputShape, filterShape, filterStride)   
        
        return
        
             
    
    
    
    
    
#%% ***************************************************************************
    
class RvBaseNeuralNetwork(object):    
    """=============================================================
    Static:
    ============================================================="""     
    @staticmethod
    def Add_Noise(oneInput, strength=0.5):
        shape = oneInput.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if ( random.randint(0,10) == 0 ):
                  oneInput[i][j] = min(1.0, oneInput[i][j]+np.random.random())  
                #oneInput[i][j] = min(1.0, oneInput[i][j]+np.random.random()*strength)                
        return oneInput
                
            
    @staticmethod
    def LayersNeurons_To_RvNeuralLayers( layersNeuronsNum ):
        return [ RvNeuralLayer(inputNeusNum, lyrNeusNum) 
            for inputNeusNum, lyrNeusNum in \
                zip(layersNeuronsNum[:-1], layersNeuronsNum[1:]) ]
        
    @staticmethod    
    def CreateLabelsY(numY, labelId):
        e = np.zeros((numY, 1))
        e[labelId] = 1.0
        return e    
    
    """=============================================================
    Constructor :
    ============================================================="""        
    # 代替 overload constructor
    @classmethod
    def Class_Create_LayersNeurons(cls, layersNeuronsNum ):
        rvLyrs = RvNeuralNetwork.LayersNeurons_To_RvNeuralLayers(layersNeuronsNum)
        return cls(rvLyrs)    
    
    def __init__(self,  *args):
        self.__Initial_Members()
        nArgs = len(args)
        if nArgs>0:            
#            print("args[0] : {}".format(type(args[0])))
#            print("args[0][0] : {}".format(type(args[0][0])))
            if isinstance(args[0], list):  
                if isinstance(args[0][0], RvNeuralLayer):
                    self.__Create_LayerObjects(*args)
                elif isinstance(args[0][0], int): 
                    self.__Create_LayerNeurons(*args)
                elif isinstance(args[0][0], RvBaseNeuralNetwork): 
                    self.__Create_CombineNetworks(*args)
            elif isinstance(args[0], str): # lyrfileName
                self.__Create_File(*args)              
            elif isinstance(args[0], object): # netObj
                self.__Create_RefNetObj(*args)            
            else:
                print("Need InputNum and NeuronNum")        
        else:
            print("No Arguments")    
            
        if Debug and self.__class__.__name__=='RvBaseNeuralNetwork':  
            self.Show_LayersInfo()  
            
    
    
    """=============================================================
    Private :
    ============================================================="""  
    def __Create_LayerObjects(self, lstLayersObjs):
        self.NeuralLayers = lstLayersObjs
        # 最後一層強制為 sigmoid
        self.NeuralLayers[-1].Set_EnumActivation(nm.EnumActivation.afSigmoid)
        
    def __Create_LayerNeurons(self, lstLayersNeurons):
        self.__Create_LayerObjects( \
            RvNeuralNetwork.LayersNeurons_To_RvNeuralLayers(lstLayersNeurons)) 



    def __Create_RefNetObj(self, refNetObj):
        self.NeuralLayers = refNetObj.NeuralLayers
        self.__Assign_Members(refNetObj)
        
    def __Create_CombineNetworks(self, netObjs):
        if None==netObjs: return
        if not isinstance(netObjs, list): return
        self.NeuralLayers = []
        for net in netObjs:
          if isinstance(net, RvBaseNeuralNetwork): 
            self.NeuralLayers += net.NeuralLayers
            self.__Assign_Members(net)
        

    def __Create_File(self, filename):
        if not rfi.FileExists(filename): return #os.path.isfile(filename) return None # 看網路參數檔案存在否
        
        f = open(filename, "r")
        data = json.load(f)
        f.close()       
        
        if len(data)<=0: return None
            
        if Debug: 
            print("\nCreate Network From File : \"{}\"".format(filename))
            
        lyrObjs = []
        for lyr in data['NeuralLayers']:
            if lyr['ClassName']==RvConvolutionLayer.__name__:
                lyrObjs.append( RvConvolutionLayer([1,1,1],[1,1,1,1],1) )
            else:
                lyrObjs.append( RvNeuralLayer(1,1) )
                
        i=0
        for lyrObj in lyrObjs:
            lyrObj.Set_LayerData(data['NeuralLayers'][i])
            i+=1
            
        self.NeuralLayers = lyrObjs
            
        # 2018/02/12 新增
        key1 = "NetEnableDropOut"
        if (key1 in data): self.NetEnableDropOut = data[key1]
        key1, key2 = "EnumDropOutValue", "DropOutRatio"
        if (key1 in data) and (key2 in data):
            enumDropOut = nm.EnumDropOutMethod( data[key1] )            
            self.Set_DropOutMethod(enumDropOut, data[key2])       
        if ('BestAccuracyRatio' in data): self.BestAccuracyRatio = data['BestAccuracyRatio']
        if ('Train_LearnRate' in data): self.Train_LearnRate = data['Train_LearnRate']
        if ('Train_Lmbda' in data): self.Train_Lmbda = data['Train_Lmbda']
            
            
    
    def __Assign_Members(self, refNetObj):   
        #self.NeuralLayers = []
        self.RandomState = np.random.RandomState(int(time.time()))
        self.NetEnableDropOut = refNetObj.NetEnableDropOut
        self.NetEnumDropOut = refNetObj.NetEnumDropOut
        self.NetDropOutRatio = refNetObj.NetDropOutRatio
        self.WorstAccuracyRatio = refNetObj.WorstAccuracyRatio
        self.BestAccuracyRatio = refNetObj.BestAccuracyRatio
        self.AverageAccuracyRatio = refNetObj.AverageAccuracyRatio      
        self.AverageCost = refNetObj.AverageCost
        self.Train_Loop = refNetObj.Train_Loop
        self.Train_LearnRate = refNetObj.Train_LearnRate
        self.Train_Lmbda = refNetObj.Train_Lmbda
        self.Train_TimeSec = refNetObj.Train_TimeSec      
        # Method -------------------------------------------
        self.PreProcessInput = refNetObj.PreProcessInput
        self.PreProcessOutput = refNetObj.PreProcessOutput
          
        
    def __PreProcessInput(self, inputX):
       if self.PreProcessInput == None: return inputX
       else: return self.PreProcessInput(inputX)
        
        
    def __PreProcessOutput(self, outputY):
       if self.PreProcessOutput == None: return outputY
       else: return self.PreProcessOutput(outputY)
        
        
        
    def __Initial_Members(self):   
        self.NeuralLayers = []
        self.RandomState = np.random.RandomState(int(time.time()))
        self.NetEnableDropOut = False
        self.NetEnumDropOut = drpOut.eoSmallActivation
        self.NetDropOutRatio = gDropOutRatio
        self.WorstAccuracyRatio = 1.0
        self.BestAccuracyRatio = 0.0
        self.AverageAccuracyRatio = 0.0      
        self.AverageCost = 0.0
        self.Train_Loop = 0
        self.Train_LearnRate = 0.0
        self.Train_Lmbda = 0.0
        self.Train_TimeSec = 0.0
        
        # Function Initialization ------------------------------------
        self.Caculate_Sum_LayersCostDerivations = \
            self.__Caculate_Sum_LayersCostDerivations_Normal
        self.Total_Cost = self.__Total_Cost_Normal
        self.Accuracy = self.__Accuracy_Normal
        # Method -------------------------------------------
        self.PreProcessInput = None
        self.PreProcessOutput = None
        
        path = "..\\TmpLogs\\{}\\".format(self.__class__.__name__)
        rfi.ForceDir(path) #if not os.path.isdir(path): os.mkdir(path)
        self.LogPath = path        
        self.__FnNetworkData = "{}{}_NetData".format(path, RvNeuralNetwork.__name__)   
        
        self.SaveVideo = Debug
        self.__VideoOutputPath = "{}{}\\".format(self.LogPath,"VideoOutput")
        self.__VidoeImageFn = "vdoImg"
        self.Set_VideoOutputPath(self.__VideoOutputPath)
        self.CurLoop = 0
        
        # Debug------------------------------------------------
        self.DoPloatActivations = Debug
        self.DoPloatWeights = Debug
        self.DoPlotOutput = Debug
        self.DoPlotTraingTrend = Debug
        self.DebugLog = Debug
              
        
    
    def __Create_ArrayOf_LayersNeuronsBiases(self, initialValue):
        return np.asarray([ lyr.Create_ArrayOf_NeuronsBiases(initialValue) 
            #np.zeros(lyr.NeuronsBiases.shape) 
            for lyr in self.NeuralLayers])
    
    def __Create_ArrayOf_LayersNeuronsWeights(self, initialValue):
        return np.asarray([ lyr.Create_ArrayOf_NeuronsWeights(initialValue) 
            #np.zeros(lyr.NeuronsBiases.shape) 
            for lyr in self.NeuralLayers])
        
        
    """----------------------------------------------------------------------------
    # 對單筆資料 正向計算所有神經元的輸出數值 valueZ 和 activation
    """  
    def __LayersNeurons_FeedForward(self, oneInput):
        lyrsNeusZs = self.__Create_ArrayOf_LayersNeuronsBiases(0.0)
        lyrsNeusActvs = self.__Create_ArrayOf_LayersNeuronsBiases(0.0)
        lyrId = 0
        for lyr in self.NeuralLayers:
#            if Debug: 
#                print("L({}):\n".format(self.NeuralLayers.index(lyr))) 
            lyrsNeusZs[lyrId], lyrsNeusActvs[lyrId] = \
                lyr.Caculate_Neurons_Z_Activations( oneInput ) 
            oneInput = lyrsNeusActvs[lyrId]
            lyrId += 1
        return lyrsNeusZs, lyrsNeusActvs
    
    
    """----------------------------------------------------------------------------
    # 計算單筆資料 對 所有神經元 Cost 偏微分的加總
    """
    def  __LayersNeurons_BackPropagation(self, oneInputX, labelY, lyrsNeusZs, lyrsNeusActvs): 
        #初始化一份和　self.biase結構、維度(多層，每層數量不同)一樣的 0值。
        lyrsNeus_dCost_dBiases = self.__Create_ArrayOf_LayersNeuronsBiases(0.0)
        lyrsNeus_dCost_dWeis = self.__Create_ArrayOf_LayersNeuronsWeights(0.0)
        # 指定前一筆輸入值          
        if len(lyrsNeusActvs)>1: priLyrNeusActvns = lyrsNeusActvs[-2]
        else: priLyrNeusActvns = oneInputX         
         
        # Step2 : 先計算最後一層輸出層的 cost 對 weight, bias 的偏微分，然後往前一直更新  
        nxtLyrNeusErrs, lyrsNeus_dCost_dWeis[-1], lyrsNeus_dCost_dBiases[-1] = \
            self.NeuralLayers[-1].Caculate_dCost_OutputLayer( \
              lyrsNeusZs[-1], lyrsNeusActvs[-1], labelY, priLyrNeusActvns)            
            
        # Step3 : 非輸出層的 backPropagationagation，從倒數第二層開始 
        nLyr = len(self.NeuralLayers)        
        for lyrId in range(nLyr-2, -1, -1 ):  #range(sId,eId,step)#倒數第二層開始 
            # 每次求出的 nxtLyrNeusErrs 在丟到函式內，當作下一層的 Error
            if lyrId==0: # input layer
                preLyrNeusAs = oneInputX                
            else:
                preLyrNeusAs =lyrsNeusActvs[lyrId-1]            
                
            nxtLyrNeusErrs, lyrsNeus_dCost_dWeis[lyrId], lyrsNeus_dCost_dBiases[lyrId] = \
                self.NeuralLayers[lyrId].Caculate_dCost_HiddenLayer(
                    #前一層的 Actvns, 當前層的valueZs
                    preLyrNeusAs, lyrsNeusZs[lyrId], 
                    #下一層的 NeuronsWeights, 上一層的 NeuronsErrs
                    self.NeuralLayers[lyrId+1], nxtLyrNeusErrs)                
        return lyrsNeus_dCost_dWeis, lyrsNeus_dCost_dBiases
    
    
    
    """----------------------------------------------------------------------------
    # 計算所有資料 對 所有神經元 Cost 偏微分的加總
    """
    
    def __Add_LayersCostDerivations0(self, inputX, labelY, sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias):
         # 正向計算所有有層的所有神經元的 valueZ, activation       
        lyrsNeusZs, lyrsNeusActvs = self.__LayersNeurons_FeedForward(inputX) 
        
                    
  
        # 計算 cost 對單筆 dataTuples[]的　所有 biases 和 weights 的偏微分
        lyrsNeus_dCost_dWei, lyrsNeus_dCost_dBias = \
            self.__LayersNeurons_BackPropagation(inputX, labelY, lyrsNeusZs, lyrsNeusActvs) 

        # 逐一累加到所屬的 lyrsNeus_dCost_dBias[layer, neuron]　
        sum_lyrsNeus_dCost_dWei = [nw+dnw for nw, dnw in 
            zip(sum_lyrsNeus_dCost_dWei, lyrsNeus_dCost_dWei)]
        sum_lyrsNeus_dCost_dBias = [nb+dnb for nb, dnb in 
            zip(sum_lyrsNeus_dCost_dBias, lyrsNeus_dCost_dBias)]
        
        return sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias                      
            
        
        
    def __Caculate_Sum_LayersCostDerivations_Normal(self, dataTuples, 
                trainOnlyDigit=-1):
        
        # 為所有層的所有神經元配置空間，儲存 dCost 對所屬 weight 和 bias 的偏微分  
        sum_lyrsNeus_dCost_dBias = self.__Create_ArrayOf_LayersNeuronsBiases(0.0)
        sum_lyrsNeus_dCost_dWei = self.__Create_ArrayOf_LayersNeuronsWeights(0.0) 
        
        # 遍覽 dataTuples[] 內每筆資料，計算每筆input產生的偏微分，加總起來 ---------------            
        for eachX, eachY in dataTuples: #每筆dataTuples[] = [x[784], y[10]]
            digitId = np.argmax(eachY)          
            doTrain = (digitId == trainOnlyDigit) or (trainOnlyDigit<0)
            if doTrain:                           
                # 正向計算所有有層的所有神經元的 valueZ, activation, cost對 wei, bias 微分       
                sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias = \
                    self.__Add_LayersCostDerivations0(eachX, eachY,
                        sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias)                    
            
        return sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias
    
    
        
    
    
    def __Caculate_Sum_LayersCostDerivations_EnDeCoder(self, dataTuples, 
                trainOnlyDigit=-1):        
        # 為所有層的所有神經元配置空間，儲存 dCost 對所屬 weight 和 bias 的偏微分  
        sum_lyrsNeus_dCost_dBias = self.__Create_ArrayOf_LayersNeuronsBiases(0.0)
        sum_lyrsNeus_dCost_dWei = self.__Create_ArrayOf_LayersNeuronsWeights(0.0) 
        
        # 遍覽 dataTuples[] 內每筆資料，計算每筆input產生的偏微分，加總起來 ---------------
        for eachX, eachY in dataTuples: #每筆dataTuples[] = [x[784], y[10]]
            digitId = np.argmax(eachY)                
            doTrain = (digitId == trainOnlyDigit) or (trainOnlyDigit<0)
            if doTrain:  
                 # 正向計算所有有層的所有神經元的 valueZ, activation, cost對 wei, bias 微分
                if (None!=self.PreProcessInput) or (None!=self.PreProcessOutput):  
                  # 非常耗時，準備資料時預先作。self.PreProcessInput()，可將 input blur，當作 sharpness
                  sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias = \
                    self.__Add_LayersCostDerivations0(\
                        self.__PreProcessInput(eachX), 
                        self.__PreProcessOutput(eachX),  # 以自己的影像當作目標 label
                        sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias)
                else:
                  # 正向計算所有有層的所有神經元的 valueZ, activation, cost對 wei, bias 微分       
                  sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias = \
                    self.__Add_LayersCostDerivations0(eachX, eachX,  # 以自己的影像當作目標 label
                        sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias)
            
        return sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias
    
    
    
    def __Caculate_Sum_LayersCostDerivations_EnDeCoder_AssignOutputY(self, dataTuples, 
                trainOnlyDigit=-1):        
        # 為所有層的所有神經元配置空間，儲存 dCost 對所屬 weight 和 bias 的偏微分  
        sum_lyrsNeus_dCost_dBias = self.__Create_ArrayOf_LayersNeuronsBiases(0.0)
        sum_lyrsNeus_dCost_dWei = self.__Create_ArrayOf_LayersNeuronsWeights(0.0) 
        
        # 遍覽 dataTuples[] 內每筆資料，計算每筆input產生的偏微分，加總起來 ---------------
        for eachX, eachY in dataTuples: #每筆dataTuples[] = [x[784], y[10]]
#            digitId = np.argmax(eachY)                
#            doTrain = (digitId == trainOnlyDigit) or (trainOnlyDigit<0)
#            if doTrain:  
                 # 正向計算所有有層的所有神經元的 valueZ, activation, cost對 wei, bias 微分       
                 sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias = \
                    self.__Add_LayersCostDerivations0(eachX, eachY,  # 以自己的影像當作目標 label
                        sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias)
            
        return sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias
            
    
    
    def __Caculate_Sum_LayersCostDerivations_Discriminator(self, dataTuples, 
                onlyLabelY=-1):        
        # 為所有層的所有神經元配置空間，儲存 dCost 對所屬 weight 和 bias 的偏微分  
        sum_lyrsNeus_dCost_dBias = self.__Create_ArrayOf_LayersNeuronsBiases(0.0)
        sum_lyrsNeus_dCost_dWei = self.__Create_ArrayOf_LayersNeuronsWeights(0.0) 
        
        # 遍覽 dataTuples[] 內每筆資料，計算每筆input產生的偏微分，加總起來 ---------------
        for eachX,eachY in dataTuples: #每筆dataTuples[] = [x[784], y[10]]
            curLabel = np.max(eachY)          
            doTrain = (curLabel == onlyLabelY) or (onlyLabelY<0)
            if doTrain:  
                # 正向計算所有有層的所有神經元的 valueZ, activation, cost對 wei, bias 微分       
                sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias = \
                    self.__Add_LayersCostDerivations0(eachX, eachY,  
                        sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias)
            
        return sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias
    
    
    
    def __Caculate_Sum_LayersCostDerivations_GAN(self, dataTuples, 
                onlyLabelY=-1):               
        # GAN 的輸出是 Discriminator,因此使用相同函數----------------   
        return self.__Caculate_Sum_LayersCostDerivations_Discriminator(\
            dataTuples, onlyLabelY)
    
    def Plot_LayerNeuronsActivations(self, iLyr, lyrNeusActs, saveFn=""):
        neuNum = len(lyrNeusActs)
        pxlW = int(np.sqrt(neuNum))
        pxls = pxlW*pxlW
        nRow, nCol = 1, 1
        pltInchW = Get_PlotWidthInch(pxlW, nCol)
        aMin = np.min(lyrNeusActs)
        aMax = np.max(lyrNeusActs)
        #aMul = (x-aMin)/(aMax-aMin)*255
        aMul = 255/(aMax-aMin)
        pltFn.Plot_Images([np.array(
            (lyrNeusActs[:pxls].transpose()-aMin)*aMul).reshape(pxlW,pxlW)],
            nRow,nCol, ["Lyr({}) NeuActs[{}]: {:.4f}~{:.4f}".format(\
                iLyr,neuNum,aMin,aMax)],saveFn, pltInchW)
        print("")
        
        
    def Plot_LayersNeuronsActivations(self, lyrsNeusActs):
        for lyrId in range(len(lyrsNeusActs)):
            self.Plot_LayerNeuronsActivations(lyrId, lyrsNeusActs[lyrId])
            
        
    def Plot_LayerNeuronWeights(self, iLyr, iNeuron, saveFn=""):
        if (iLyr<0) or (iLyr>=len(self.NeuralLayers)): return        
        if (iNeuron<0) or (iNeuron>=len(self.NeuralLayers[iLyr].NeuronsWeights)): return      
        #print("{}:".format(self.__class__.__name__))
        self.NeuralLayers[iLyr].Plot_OneNeuronWeights(iNeuron, 
            self.NeuralLayers[iLyr].NeuronsWeights[iNeuron], iLyr, saveFn )
            
    
    def Plot_LayerNeuronsWeights(self, iLyr, saveFn=""):
        if (iLyr<0) or (iLyr>=len(self.NeuralLayers)): return   
        sFn = rfi.ExtractFileName(saveFn)
        sExt = rfi.ExtractFileExt(saveFn)   
        for iNeuron in range(len(self.NeuralLayers[iLyr].NeuronsWeights)):
            nFn = "{}_N{:04}Wei{}".format(sFn,iNeuron,sExt)  
            self.Plot_LayerNeuronWeights(iLyr, iNeuron, nFn)
        
    def Plot_LayersNeuronsWeights(self, saveFn=""):
        sFn = rfi.ExtractFileName(saveFn)
        sExt = rfi.ExtractFileExt(saveFn)
        for ilyr in range(len(self.NeuralLayers)):
            nFn = "{}_L{:02}{}".format(sFn,ilyr,sExt)
            self.Plot_LayerNeuronsWeights(ilyr, nFn)
        
    # 計算 輸出層的 Activations ------------------------------------------------------
    def __Caculate_OutputActivations_FeedForward(self, oneInput):  #前面加 "__xxxfuncname()" 表示 private
        for lyr in self.NeuralLayers:
            oneInput = lyr.Get_NeuronsActivations(lyr.Get_NeuronsValuesZ(oneInput) )
                        
        return oneInput
    
    
    def __Caculate_OutputActivations_FeedForward_ClassActivation(self, oneInput, clsActivation=None):  #前面加 "__xxxfuncname()" 表示 private
        if (None==clsActivation):
          for lyr in self.NeuralLayers:
            oneInput = lyr.Get_NeuronsActivations(lyr.Get_NeuronsValuesZ(oneInput) )
        else:
          for lyr in self.NeuralLayers:
            oneInput = clsActivation.activation(lyr.Get_NeuronsValuesZ(oneInput) )
          
        return oneInput

   
    # 預測值，傳入一筆測試資料 ------------------------------------------------------
    def __Get_CorrectNum(self, test_data):  #前面加 "__xxxfuncname()" 表示 private
        #測試資料 test_data[10000] = [ x[784], y[1] ], [ x[784], y[1] ],..... 10000筆   
        # 找到最大值所在的 index=數字結果 ---------------------
        test_results = \
            [(np.argmax(self.__Caculate_OutputActivations_FeedForward(x)), y)
             for (x, y) in test_data] # x,y 分別代表 test_data[0], test_data[1]
        return sum(int(x == y) for (x, y) in test_results)
        
    
    
    # 計算每個輸出的 cost ------------------------------------------------
    def __Total_Cost_Normal(self, dataTuples, lmbda, createLabelY=False, plotOutput=False):
        cost = 0.0
        #drawDigits = [False]*10 # -> [False, False, False... False]
        if plotOutput: 
            #digitFigs = {title : [] for title in range(10)}
            digitFigs = [ [] for title in range(10)]
            outputFigs = [ [] for title in range(10)] #np.copy(digitFigs)
            nCol = 5 #int(np.sqrt(len(digitFigs)))
            nRow = 2 #int(len(digitFigs)/nCol)+1
            pxls = len(dataTuples[0][0])
            pxlW = int(np.sqrt(pxls))
            pxls = pxlW*pxlW
            pltInchW = Get_PlotWidthInch(pxlW, nCol)
            Ypxls = self.NeuralLayers[-1].Get_NeuronsNum() #10 if createLabelY else len(dataTuples[0][1])
            YpxlW = int(np.sqrt(Ypxls)) #+ (1 * (Ypxls%YpxlW>0) )
            Ypxls = YpxlW*YpxlW
            YpltInchW =Get_PlotWidthInch(YpxlW, nCol)
            
        n_Data = len(dataTuples)
        for x, y in dataTuples:
            if createLabelY: y = RvBaseNeuralNetwork.CreateLabelsY(10, y)
            digitId =  np.argmax(y)
            finalLyrNeuronsActvns = self.__Caculate_OutputActivations_FeedForward(x)
            # 以最後一層的 classCost計算 -------------------
            cost += self.NeuralLayers[-1].ClassCost.costValue(finalLyrNeuronsActvns, y)/n_Data 
            
            if plotOutput: 
                if digitFigs[digitId]==[]:
#                    digitFigs[digitId]=
#                      np.array(x[:pxls].transpose()).reshape(pxlW,pxlW)*255
                    digitFigs[digitId]= \
                      np.array(x[:pxls].transpose()).reshape(pxlW,pxlW)*255      
                    outputFigs[digitId]= \
                      np.array(finalLyrNeuronsActvns[:Ypxls].transpose()).reshape(YpxlW,YpxlW)*255
                      
        if plotOutput: 
            if self.SaveVideo: 
                fn = "{}{}_{}.png".format(self.__VideoOutputPath, self.__VidoeImageFn, self.CurLoop)
            else:
                fn = ""
            inputShape = (len(dataTuples[0][0]), 1)
            for i in range(len(digitFigs)):
                if digitFigs[i]==[]:digitFigs[i]=np.full(inputShape,0) 
            for i in range(len(outputFigs)):
                if outputFigs[i]==[]:outputFigs[i]=np.full(inputShape,0) 
            #pltFn.Plot_Figures(digitFigs, nRow, nCol)    
            
            pltFn.Plot_Images(np.array(digitFigs),nRow,nCol, 
                ["Input[{}]".format(pxls)], fn, pltInchW)
            pltFn.Plot_Images(np.array(outputFigs),nRow,nCol, 
                ["Output[{}]".format(Ypxls)],"", YpltInchW)
            
        # Regularization = lmbda/2n * Sum(wei^2)
        cost += 0.5*(lmbda/n_Data)*sum(
            np.linalg.norm(lyr.NeuronsWeights)**2 for lyr in self.NeuralLayers)
        return cost    
    
    
    # 計算每個輸出的 cost ------------------------------------------------
    def __Total_Cost_EnDeCoder(self, dataTuples, lmbda, createLabelY=False, plotOutput=False):
        
        if plotOutput: 
            digitFigs = [ [] for title in range(10)]
            outputFigs = [ [] for title in range(10)] #np.copy(digitFigs)
            nCol = 5 #int(np.sqrt(len(digitFigs)))
            nRow = 2 #int(len(digitFigs)/nCol)+1
            pxls = len(dataTuples[0][0])
            pxlW = int(np.sqrt(pxls))
            pxls = pxlW*pxlW
            pltInchW = Get_PlotWidthInch(pxlW, nCol)
            
        cost = 0.0
        n_Data = len(dataTuples)        
        for x, y in dataTuples:                
            digitId =  np.argmax(y)
            finalLyrNeuronsActvns = self.__Caculate_OutputActivations_FeedForward(x)
            cost += self.NeuralLayers[-1].ClassCost.costValue(
                      finalLyrNeuronsActvns, x)/n_Data
                
            if plotOutput: 
                if digitFigs[digitId]==[]:
                    digitFigs[digitId]= \
                      np.array(x[:pxls].transpose()).reshape(pxlW,pxlW)*255      
                    outputFigs[digitId]= \
                      np.array(finalLyrNeuronsActvns[:pxls].transpose()).reshape(pxlW,pxlW)*255
                      
        if plotOutput:       
            # fn = "{}_{}.png".format(self.__VidoeImageFn, self.CurLoop)
            if self.SaveVideo: 
                fn = "{}{}_{}.png".format(self.__VideoOutputPath, self.__VidoeImageFn, self.CurLoop)
            else:
                fn = ""
                
            inputShape = (len(dataTuples[0][0]), 1)
            for i in range(len(digitFigs)):
                if digitFigs[i]==[]:digitFigs[i]=np.full(inputShape,0) 
            for i in range(len(outputFigs)):
                if outputFigs[i]==[]:outputFigs[i]=np.full(inputShape,0)       
            pltFn.Plot_Images(np.array(digitFigs),nRow,nCol, 
                ["Input[{}]".format(pxls)],"", pltInchW)
            pltFn.Plot_Images(np.array(outputFigs),nRow,nCol, 
                ["Output[{}]".format(pxls)], fn, pltInchW)
            
        # Regularization = lmbda/2n * Sum(wei^2)
        cost += 0.5*(lmbda/n_Data)*sum(
            np.linalg.norm(lyr.NeuronsWeights)**2 for lyr in self.NeuralLayers)
        return cost    
    
    
    # 計算每個輸出的 cost ------------------------------------------------
    def __Total_Cost_EnDeCoder_AssignOutputY(self, dataTuples, lmbda, createLabelY=False, plotOutput=False):
        
        if plotOutput: 
            nCol = 3 #int(np.sqrt(len(digitFigs)))
            nRow = 1 #int(len(digitFigs)/nCol)+1
            pxls = len(dataTuples[0][0])
            pxlW = int(np.sqrt(pxls))
            pxls = pxlW*pxlW
            pltInchW = Get_PlotWidthInch(pxlW, nCol)
            Ypxls = len(dataTuples[0][1])
            YpxlW = int(np.sqrt(Ypxls))
            Ypxls = YpxlW*YpxlW
            YpltInchW = Get_PlotWidthInch(YpxlW, nCol)
            
        cost = 0.0
        n_Data = len(dataTuples)        
        for x, y in dataTuples:                
            #digitId =  np.argmax(y)
            finalLyrNeuronsActvns = self.__Caculate_OutputActivations_FeedForward(x)
            cost += self.NeuralLayers[-1].ClassCost.costValue(
                      finalLyrNeuronsActvns, x)/n_Data
                
            if plotOutput: 
                #fn = "{}_{}.png".format(self.__VidoeImageFn, self.CurLoop)
                if self.SaveVideo: 
                    fn = "{}{}_{}.png".format(self.__VideoOutputPath, 
                      self.__VidoeImageFn, self.CurLoop)
                else:
                    fn = ""
                plotOutput = False   
                img1 = np.array(x[:pxls].transpose()).reshape(pxlW,pxlW)*255
                img2 = np.array(y[:Ypxls].transpose()).reshape(YpxlW,YpxlW)*255   
                img3 = np.array(finalLyrNeuronsActvns[:Ypxls].transpose()).reshape(YpxlW,YpxlW)*255
                pltFn.Plot_Images([img1,img2,img3], nRow,nCol,
                    ["Input", "LableY", "Output[{}]".format(pxls)], 
                    fn, pltInchW)            
            
        # Regularization = lmbda/2n * Sum(wei^2)
        cost += 0.5*(lmbda/n_Data)*sum(
            np.linalg.norm(lyr.NeuronsWeights)**2 for lyr in self.NeuralLayers)
        return cost    
    
    
    
    # 計算每個輸出的 cost ------------------------------------------------
    def __Total_Cost_Discriminator(self, dataTuples, lmbda, createLabelY=False, plotOutput=False):
        
        if plotOutput: 
            inputShape = (len(dataTuples[0][0]), 1)
            #digitFigs = [ np.full(inputShape,1) for title in range(10)]
            digitFigs = [ [] for title in range(10)]
            #outputFigs = [ [] for title in range(10)] #np.copy(digitFigs)
            nCol = 5 #int(np.sqrt(len(digitFigs)))
            nRow = 2 #int(len(digitFigs)/nCol)+1
            pxls = len(dataTuples[0][0])
            pxlW = int(np.sqrt(pxls))
            pxls = pxlW*pxlW
            pltInchW = Get_PlotWidthInch(pxlW, nCol)
            
        sTitles = []
            
        cost = 0.0;  digitId=0
        n_Data = len(dataTuples)        
        for x, y in dataTuples:       
            finalLyrNeuronsActvns = self.__Caculate_OutputActivations_FeedForward(x)
            cost += self.NeuralLayers[-1].ClassCost.costValue(
                      finalLyrNeuronsActvns, x)/n_Data
                
            if plotOutput: 
                if digitId<len(digitFigs) and digitFigs[digitId]==[]:
                    fn = "{}_{}.png".format(self.__VidoeImageFn, self.CurLoop)
                    digitFigs[digitId]= \
                      np.array(x[:pxls].transpose()).reshape(pxlW,pxlW)*255  
                    sTitles += ["Output ({:.3f})".format(\
                        np.max(finalLyrNeuronsActvns))]
                    digitId+=1
                    
        if plotOutput:       
            if self.SaveVideo: 
                fn = "{}{}_{}.png".format(self.__VideoOutputPath, self.__VidoeImageFn, self.CurLoop)
            else:
                fn = ""
            inputShape = (len(dataTuples[0][0]), 1)
            for i in range(len(digitFigs)):
                if digitFigs[i]==[]:digitFigs[i]=np.full(inputShape,0) 
            pltFn.Plot_Images(np.array(digitFigs),nRow,nCol, sTitles,fn, pltInchW)
            
        # Regularization = lmbda/2n * Sum(wei^2)
        cost += 0.5*(lmbda/n_Data)*sum(
            np.linalg.norm(lyr.NeuronsWeights)**2 for lyr in self.NeuralLayers)
        return cost    
    
    
    
    def __Total_Cost_GAN(self, dataTuples, lmbda, createLabelY=False, plotOutput=False):
        
        if plotOutput: 
            inputShape = (len(dataTuples[0][0]), 1)
            #digitFigs = [ np.full(inputShape,1) for title in range(10)]
            digitFigs = [ [] for title in range(10)]
            #outputFigs = [ [] for title in range(10)] #np.copy(digitFigs)
            nCol = 5 #int(np.sqrt(len(digitFigs)))
            nRow = 2 #int(len(digitFigs)/nCol)+1
            pxls = len(dataTuples[0][0])
            pxlW = int(np.sqrt(pxls))
            pxls = pxlW*pxlW
            pltInchW = Get_PlotWidthInch(pxlW, nCol)
            
        sTitles = []
            
        cost = 0.0;  digitId=0
        n_Data = len(dataTuples)        
        for x, y in dataTuples:       
            finalLyrNeuronsActvns = self.__Caculate_OutputActivations_FeedForward(x)
            cost += self.NeuralLayers[-1].ClassCost.costValue(
                      finalLyrNeuronsActvns, x)/n_Data
                
            if plotOutput: 
                if digitId<len(digitFigs) and digitFigs[digitId]==[]:
                    fn = "{}_{}.png".format(self.__VidoeImageFn, self.CurLoop)
                    digitFigs[digitId]= \
                      np.array(x[:pxls].transpose()).reshape(pxlW,pxlW)*255  
                    sTitles += ["( {:.3f} )".format(np.max(finalLyrNeuronsActvns))]
                    digitId+=1
                    
        if plotOutput:       
            if self.SaveVideo: 
                fn = "{}{}_{}.png".format(self.__VideoOutputPath, self.__VidoeImageFn, self.CurLoop)
            else:
                fn = ""
            inputShape = (len(dataTuples[0][0]), 1)
            for i in range(len(digitFigs)):
                if digitFigs[i]==[]:digitFigs[i]=np.full(inputShape,0) 
            pltFn.Plot_Images(np.array(digitFigs),nRow,nCol, sTitles,fn, pltInchW)
            
        # Regularization = lmbda/2n * Sum(wei^2)
        cost += 0.5*(lmbda/n_Data)*sum(
            np.linalg.norm(lyr.NeuronsWeights)**2 for lyr in self.NeuralLayers)
        return cost    
    
    
    
    # 計算 y=a 辨識正確的data 數量-----------------------------------------
    def __Accuracy_Normal(self, dataTuples, isLabelArray=False):
        if isLabelArray:
            # 將 計算結果 a 和 label 放在 array[a,y], np.argmax(y):找出 y[]中最大值所在的 idx
          results = \
             [(np.argmax(self.__Caculate_OutputActivations_FeedForward(x)), np.argmax(y))
              for (x, y) in dataTuples]
        else:
          results = \
             [(np.argmax(self.__Caculate_OutputActivations_FeedForward(x)), y)
              for (x, y) in dataTuples]
        #隨機畫出測試的數字 ------    
        #rf.Plot_Digit(dataTuples[self.RandomState.randint(0,len(dataTuples))])
        # 瀏覽所有結果，如果
        iCorrectNum = sum(int(x == y) for (x, y) in results)
        
        return iCorrectNum
    
   
    # 計算 y=a 辨識正確的data 數量-----------------------------------------
    def __Accuracy_EnDeCoder(self, dataTuples, isLabelArray=False):
        
        errSum = 0.0
        nDatas = len(dataTuples)
        outputNeus = len(dataTuples[0][0])
        iCorrectNum = 0
        for (x, y) in dataTuples:
            #digitId =  np.argmax(y)
            acts = self.__Caculate_OutputActivations_FeedForward(x)
#                acts = self.__Caculate_OutputActivations_FeedForward_ClassActivation(x,
#                    nm.Activation_Sigmoid)
            """
            np.linalg.norm(x, ord=None, axis=None, keepdims=False)
            默認	二范數：ℓ2 ->	sqrt(x1^2 + x2^2 + .....)
            ord=2	二范數：ℓ2 ->	同上
            ord=1	一范數：ℓ1 ->	|x1|+|x2|+…+|xn|
            ord=np.inf	無窮范數：ℓ∞ ->		max(|xi|)
            """
            #err = (np.linalg.norm(acts-decoderLabels[digitId])**2 )/outputNeus
            # np.linalg.norm([x1,x2...xn], ord=1) -> abs(x1) + abs(x2) +....
            err = (np.linalg.norm(acts-x, ord=1) ) /outputNeus
            # 加總所有 error
            errSum += err
            
        iCorrectNum = abs(nDatas*1.0 - errSum)
        
        return iCorrectNum 
    
    
    # 計算 y=a 辨識正確的data 數量-----------------------------------------
    def __Accuracy_EnDeCoder_AssignOutputY(self, dataTuples, isLabelArray=False):
        
        errSum = 0.0
        nDatas = len(dataTuples)
        outputNeus = len(dataTuples[0][0])
        iCorrectNum = 0
        for (x, y) in dataTuples:
            #digitId =  np.argmax(y)
            acts = self.__Caculate_OutputActivations_FeedForward(x)
#                acts = self.__Caculate_OutputActivations_FeedForward_ClassActivation(x,
#                    nm.Activation_Sigmoid)
            """
            np.linalg.norm(x, ord=None, axis=None, keepdims=False)
            默認	二范數：ℓ2 ->	sqrt(x1^2 + x2^2 + .....)
            ord=2	二范數：ℓ2 ->	同上
            ord=1	一范數：ℓ1 ->	|x1|+|x2|+…+|xn|
            ord=np.inf	無窮范數：ℓ∞ ->		max(|xi|)
            """
            #err = (np.linalg.norm(acts-decoderLabels[digitId])**2 )/outputNeus
            # np.linalg.norm([x1,x2...xn], ord=1) -> abs(x1) + abs(x2) +....
            err = (np.linalg.norm(acts-y, ord=1) ) /outputNeus
            # 加總所有 error
            errSum += err
            
        iCorrectNum = abs(nDatas*1.0 - errSum)
        
        return iCorrectNum 
    


    def __Accuracy_Discriminator(self, dataTuples, isLabelArray=False):
        errSum = 0.0
        nDatas = len(dataTuples)
        #outputNeus = len(dataTuples[0][0])
        iCorrectNum = 0
        for (x, y) in dataTuples:
            #digitId =  np.argmax(y)
            acts = self.__Caculate_OutputActivations_FeedForward(x)
#                acts = self.__Caculate_OutputActivations_FeedForward_ClassActivation(x,
#                    nm.Activation_Sigmoid)
            labelY = np.max(acts)
            err = abs(labelY-np.max(y))  #(np.linalg.norm(acts-x, ord=1) ) /outputNeus
            # 加總所有 error
            errSum += err
            
        iCorrectNum = abs(nDatas*1.0 - errSum)
        
        return iCorrectNum 
    
    
    def __Accuracy_GAN(self, dataTuples, isLabelArray=False):
        errSum = 0.0
        nDatas = len(dataTuples)
        #outputNeus = len(dataTuples[0][0])
        iCorrectNum = 0
        for (x, y) in dataTuples:
            #digitId =  np.argmax(y)
            acts = self.__Caculate_OutputActivations_FeedForward(x)
#                acts = self.__Caculate_OutputActivations_FeedForward_ClassActivation(x,
#                    nm.Activation_Sigmoid)
            labelY = np.max(acts)
            err = abs(labelY-np.max(y))  #(np.linalg.norm(acts-x, ord=1) ) /outputNeus
            # 加總所有 error
            errSum += err
            
        iCorrectNum = abs(nDatas*1.0 - errSum)
        
        return iCorrectNum 
    
    
    
    
    
           
    def __Write_NetworkData(self, pF):          
        data = {
            'ClassName': self.__class__.__name__,
            'LayersNeurons' : self.Get_LayersNeurons(),
            'NetEnableDropOut' : self.NetEnableDropOut,
            'EnumDropOutValue' : self.NetEnumDropOut.value,
            'DropOutRatio' : self.NetDropOutRatio,
            'BestAccuracyRatio' : self.BestAccuracyRatio,
            'Train_Loop': self.Train_Loop,
            'Train_LearnRate' : self.Train_LearnRate,
            'Train_Lmbda': self.Train_Lmbda,
            'Train_TimeSec': self.Train_TimeSec,
            'NeuralLayers': [ lyr.Get_LayerData(self.NeuralLayers.index(lyr)) 
                for lyr in self.NeuralLayers ],
            'Comapny': rg.cRasVectorCompany,
            'Web': rg.cRasVectorWeb,
            'Email': rg.cRasVectorEmail }
        json.dump(data, pF)
               
            
    
        
    """
    # 根據抽樣訓練集 mini_batch[]，利用梯度下降 來更新當前網路的所有 biases[], weights[]
    # 新的 nBiases = oBiases[] - learnRate * d(Cost)/d(bias)
    # 新的 nWeis = oWeis[] - learnRate * d(Cost)/d(wei)
    """
    def __Update_LayersNeurons_Weights_Biases(self, dataTuples, learnRate, trainOnlyDigit=-1): 
        n_Data = len(dataTuples) 
        # 計算所有 layers 的所有 Neurons 的 Cost偏微分, dCost/dWei, dCost/dBia 
        lyrsNeus_dCost_dWei_sum, lyrsNeus_dCost_dBias_sum = \
            self.Caculate_Sum_LayersCostDerivations(dataTuples, trainOnlyDigit)        
        
        # 計算新的 nWei = oWei - learnRate * nw,   nw = d(Cost)/d(wei) 
        # Adagrad learnRate = learnRate*nW / sqrt(sum(grad(i)^2))
        lrDivN = learnRate/n_Data
        for lyr, oneLyrNWs in zip(self.NeuralLayers, lyrsNeus_dCost_dWei_sum):
            if lyr.DoUpdateNeuronsWeightsBiases: # ConvolutionLayer,不更新FilterWeights
                lyr.NeuronsWeights = np.asarray(
                    #lyrWs - learnRate*(oneLyrNWs/n_Data)
                    lyr.NeuronsWeights - lrDivN*oneLyrNWs)  #Adagrad /np.linalg.norm(oneLyrNWs)) 
        # 計算新的 nbias = obias - learnRate * nb,   nb = d(Cost)/d(bias) 
        for lyr, oneLyrNBiases in zip(self.NeuralLayers, lyrsNeus_dCost_dBias_sum):
            if lyr.DoUpdateNeuronsWeightsBiases:
                lyr.NeuronsBiases = np.asarray( 
                    #lyrBiases - learnRate*(oneLyrNBiases/n_Data)
                    lyr.NeuronsBiases - lrDivN*oneLyrNBiases)
                
        self.LastLyrsNeus_dCost_dWei_sum, self.LastLyrsNeus_dCost_dBias_sum = \
            lyrsNeus_dCost_dWei_sum, lyrsNeus_dCost_dBias_sum
    

    def __Show_TrainingInfo(self, sFunc, training_data=None, trainOnlyDigit=-1):
        print("\n************************************************")
        print("{} with Stochastic Gradient Desent ".format(sFunc))
        if trainOnlyDigit in (0,10): print("Train Only Digit = {}".format(trainOnlyDigit))
        print("Best Accuracy = {}".format(self.BestAccuracyRatio))
        print("DropOut={}, DropRatio={}, DropOutMethod={}".
              format(self.NetEnableDropOut, self.NetDropOutRatio, self.NetEnumDropOut.name))
        print("Random DropOut doesn't help in MNIST training")
        print("************************************************")
        if (None!=training_data) and (len(training_data[0])>0):
            print("Input = {}".format(len(training_data[0][0])))
        print("LayersName = {}\nLayers Neurons = {}\nLayers Weights = {}".
              format(self.Get_LayersName(), self.Get_LayersNeurons(),
                     self.Get_LayersNeuronsWeightsNum()) )        
        for lyr in self.NeuralLayers:
            if (lyr.__class__.__name__ == RvConvolutionLayer.__name__):
              print("FilterShareWeights = {}".format(lyr.FilterShareWeights))
              break        
        if Debug:
            print("Layers UpdateWeiBias = {}\nLayers Actvation = {}\nLayers Cost = {}".
              format(
                [lyr.DoUpdateNeuronsWeightsBiases for lyr in self.NeuralLayers],
                [lyr.ClassActivation.__name__ for lyr in self.NeuralLayers],
                [lyr.ClassCost.__name__ for lyr in self.NeuralLayers]
                ))
            
            
    """=============================================================
    Public :
    ============================================================="""
        
    #---------------------------------------------------------------------
    # Get Functions
    #---------------------------------------------------------------------
    def Get_MinMaxErrorDigit(self, output, labelY):
        if (None==labelY): return 0,0
        outputNeus = len(labelY[0])        
        errs = [ (np.linalg.norm(output-digitValue, ord=1) ) / outputNeus
                for digitValue in labelY]        
        return np.argmin(errs), np.argmax(errs)
    
    def Get_Accuracy_EnDeCoder(self, inputX,outputY):
        """
        np.linalg.norm(x, ord=None, axis=None, keepdims=False)
        默認	二范數：ℓ2 ->	sqrt(x1^2 + x2^2 + .....)
        ord=2	二范數：ℓ2 ->	同上
        ord=1	一范數：ℓ1 ->	|x1|+|x2|+…+|xn|
        ord=np.inf	無窮范數：ℓ∞ ->		max(|xi|)
        """
        #err = (np.linalg.norm(acts-decoderLabels[digitId])**2 )/outputNeus
        # np.linalg.norm([x1,x2...xn], ord=1) -> abs(x1) + abs(x2) +....
        return 1.0 - (np.linalg.norm(outputY-inputX, ord=1) ) / len(inputX)
             
               
    def Get_InputNum(self):
        if len(self.NeuralLayers)<=0: return 0            
        return self.NeuralLayers[0].Get_InputNum()
    
    def Get_OutputNum(self):
        if len(self.NeuralLayers)<=0: return 0            
        return self.NeuralLayers[-1].Get_NeuronsNum()
    
    def Get_LayersNum(self):
        return len(self.NeuralLayers)
    
    def Get_LayersName(self):
        return [lyrObj.__class__.__name__ for lyrObj in self.NeuralLayers]

    def Get_LayersNeurons(self):
        if self.Get_LayersNum()<=0: return []
        lyrsNeus = [ lyr.Get_NeuronsNum() for lyr in self.NeuralLayers ]
        #lyrsNeus.insert(0, self.NeuralLayers[0].Get_InputNum())
        return lyrsNeus
    
    def Get_LayersNeuronsWeightsNum(self):        
        if self.Get_LayersNum()<=0: return []
        lyrsNeusWeis = [ len(lyr.NeuronsWeights)*len(lyr.NeuronsWeights[0])
            #lyr.Get_NeuronsNum()*lyr.Get_InputNum() 
            for lyr in self.NeuralLayers ]
        #lyrsNeusWeis.insert(0, 0)
        return lyrsNeusWeis

    def Get_ConvolutionLayerID(self):
        convLyrIds = []
        id=-1
        for lyr in self.NeuralLayers:
            id+=1
            if lyr.__class__.__name__ == RvConvolutionLayer.__name__:
                convLyrIds.append(id)
        return convLyrIds
    
    
    def Get_NetworkFileData(self, fnNetData):  
        accur = 0.0
        if not rfi.FileExists(fnNetData): #os.path.isfile(fnNetData):  # 看網路參數檔案存在否
            return accur
        
        f = open(fnNetData, "r")
        data = json.load(f)
        f.close()       
        
        if len(data)<=0: return accur
        """        
        'ClassName': self.__class__.__name__,
        'LayersNeurons' : self.Get_LayersNeurons(),
        'NetEnableDropOut' : self.NetEnableDropOut,
        'EnumDropOutValue' : self.NetEnumDropOut.value,
        'DropOutRatio' : self.NetDropOutRatio,
        'BestAccuracyRatio' : self.BestAccuracyRatio,
        'Train_Loop': self.Train_Loop,
        'Train_LearnRate' : self.Train_LearnRate,
        'Train_Lmbda': self.Train_Lmbda,
        'Train_TimeSec': self.Train_TimeSec,
        'NeuralLayers': [ lyr.Get_LayerData(self.NeuralLayers.index(lyr)) 
            for lyr in self.NeuralLayers ],
        'Comapny': cRasVectorCompany,
        'Web': cRasVectorWeb,
        'Email': cRasVectorEmail }
        """
        # key1, key2, key3 = "InputShape", "FilterShape", "FilterStride"
        # if (key1 in data) and (key2 in data) and (key3 in data):    
        key1 = 'BestAccuracyRatio'
        if (key1 in data):
            accur= data['BestAccuracyRatio']
            
        return accur
    
    def Get_VideoOutputPath(self):
        return self.__VideoOutputPath
        
        
        
    
    #---------------------------------------------------------------------
    # Set Functions
    #---------------------------------------------------------------------
    def Set_VideoOutputPath(self, vdoPath):
        if (""==vdoPath): return
        self.__VideoOutputPath =  vdoPath #"{}{}\\".format(vdoPath,"VideoImgs")
        rfi.ForceDir(self.__VideoOutputPath)
        
        
    def Set_DropOutMethod(self, enumDropOut, ratioDropOut=0.5):   
        self.NetDropOutRatio = ratioDropOut
        self.NetEnumDropOut = enumDropOut    
    
    def Set_EnumActivation(self, enumActivation=nm.EnumActivation.afSigmoid):
        for lyr in self.NeuralLayers:
            lyr.ClassActivation, lyr.ClassCost =  nm.Get_ClassActivation(enumActivation)
        

    #---------------------------------------------------------------------
    # Display Functions
    #---------------------------------------------------------------------
    def Show_LayersInfo(self):    
        if [] == self.NeuralLayers:return 
        
        print("\n{}".format(self.__class__.__name__))
        print("  BestAccuracy = {}".format(self.BestAccuracyRatio))
        print("  LayersName = {}\n  LayersNeurons = {}".
            format(self.Get_LayersName(), self.Get_LayersNeurons() ) )      
        print("  LayersWeightsNum = {}".
              format(self.Get_LayersNeuronsWeightsNum()) )              

        for lyr in self.NeuralLayers:
            print("  L({}) : Input({}), Neurons({}), ({}),({})".
                  format(self.NeuralLayers.index(lyr)+1, lyr.Get_InputNum(), \
                         lyr.Get_NeuronsNum(), lyr.ClassActivation.__name__, \
                         lyr.ClassCost.__name__)) 
    
    
    #---------------------------------------------------------------------
    # File IO Functions
    #---------------------------------------------------------------------           
    def Save_NetworkData(self, filename=""):         
        if ""==filename: 
            filename="{}_NetData.nnf".format(self.__class__.__name__)  
        
        path = os.path.dirname(os.path.abspath(filename)) 
        rfi.ForceDir(path) #if not os.path.isdir(path): os.mkdir(path)
        
        pf = open(filename, "w")
        self.__Write_NetworkData(pf)
        pf.close()        
        
        
    def Get_OutputValues(self, oneInput):
        return self.__Caculate_OutputActivations_FeedForward(oneInput)
    
    
    def Get_LayerNeuronsActivations(self, oneInput, atLyr=-1): # atLyr=-1, 計算到最後一層
        iLyr = 0
        for lyr in self.NeuralLayers:
            oneInput = lyr.Get_NeuronsActivations(lyr.Get_NeuronsValuesZ(oneInput) )
            if (atLyr<0) and (iLyr==atLyr): break
            iLyr+=1
            
        return oneInput
    
    def Get_LayersNeuronsActivations(self, oneInput):
        lyrNeusActs = []
        for lyr in self.NeuralLayers:
            oneInput = lyr.Get_NeuronsActivations(lyr.Get_NeuronsValuesZ(oneInput) )
            lyrNeusActs.append(oneInput)
        return lyrNeusActs

        
    def Update_LayersNeurons_Weights_Biases_OneInput(self, 
            dataTuples, learnRate, trainOnlyDigit=-1):   
        
        #Funciton Initialization ------------------------------
        if self.__class__.__name__ == RvBaseNeuralNetwork.__name__:   
            self.Caculate_Sum_LayersCostDerivations = \
                self.__Caculate_Sum_LayersCostDerivations_Normal
            self.Total_Cost = \
                self.__Total_Cost_EnDeCoder
            self.Accuracy = \
                self.__Accuracy_Normal  
        else:        
            self.Caculate_Sum_LayersCostDerivations = \
                self._RvBaseNeuralNetwork__Caculate_Sum_LayersCostDerivations_Normal
            self.Total_Cost = \
                 self._RvBaseNeuralNetwork__Total_Cost_EnDeCoder
            self.Accuracy = \
                 self._RvBaseNeuralNetwork__Accuracy_Normal          
#        else:
#            self.Caculate_Sum_LayersCostDerivations = 
#                self._RvBaseNeuralNetwork__Caculate_Sum_LayersCostDerivations_Normal
#            self.Total_Cost = self._RvBaseNeuralNetwork__Total_Cost_Normal
#            self.Accuracy = self._RvBaseNeuralNetwork__Accuracy_Normal
        
                
        dataTuples = np.array(dataTuples)
        self.__Update_LayersNeurons_Weights_Biases(dataTuples, learnRate, trainOnlyDigit)
        
    
    
 
    # 預測某張圖的數字 ------------------------------------
    def Plot_Output(self, oneInputX, saveFn=""):
        #output = self._RvBaseNeuralNetwork__Caculate_OutputActivations_FeedForward(oneInput)
        output = self.Get_OutputValues(oneInputX)
        rf.Plot_Digit( [output.transpose(),0], -1,-1, saveFn)
        return output        
       
    
    
        
        
        
        
#%%  ***************************************************************************

class RvNeuralNetwork(RvBaseNeuralNetwork, object):
 
    """=============================================================
    Constructor :
    ============================================================="""       
    def __init__(self,  *args):
        super().__init__(*args) #加上此，產生所有 parent 的 members 
        #self.__newMember = xxxxx
        if Debug and self.__class__.__name__=='RvNeuralNetwork':  
            self.Show_LayersInfo()  
    
    """=============================================================
    Private :
    ============================================================="""    
    
    def __Update_List_Cost_Accuracy(self, sTitle, input_data,
            lmbda, list_cost, list_accuracy, createLabelY=False, plotOutput=False):
        if None==input_data or None==list_cost: return
        cost = 0.0
        cost = self.Total_Cost(input_data, lmbda, createLabelY, plotOutput)
        list_cost.append(cost)
        accuracy = self.Accuracy(input_data, not createLabelY)
        list_accuracy.append(accuracy)       
        
        return list_cost, list_accuracy

    
    def __Update_Best_Cost_Accuracy(self, fnNetworkData1, sTitle, loop, t00, t0, 
            inpupt_dataNum, list_cost, list_accuracy, worstAccuracyRatio, bestAccuracyRatio, incCost, incAccuRatio):
        
        n_test = inpupt_dataNum
        cost = list_cost[-1]
        accuracy = list_accuracy[-1]
        
        incCost += cost
        accuRatio = accuracy/n_test
        incAccuRatio += accuRatio    
        
        
        s1 = ""    
        if accuRatio > bestAccuracyRatio: 
            s1 = "<-- Best"
            bestAccuracyRatio = accuRatio
            
            
        if accuRatio < worstAccuracyRatio:
            worstAccuracyRatio = accuRatio
            
            
        if ""!=fnNetworkData1:
            fnNw = "{}_Best.nnf".format(fnNetworkData1)
            fileAccu = self.Get_NetworkFileData(fnNw)
            if (bestAccuracyRatio>fileAccu):         
              self.Train_TimeSec = time.time()-t0
              self.BestAccuracyRatio = bestAccuracyRatio
              self.Train_Loop = self.CurLoop+1
              self.Save_NetworkData(fnNw)  
        
            
        if self.DebugLog:
            print("  {}: Cost({:.4f}), Accuracy({:.4f}) {}".
                format(sTitle, cost, accuRatio, s1))              
            dt = time.time()-t0
            sdT = timedelta(seconds=int(dt))
            #sTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sTime = datetime.now().strftime('%m-%d, %H:%M:%S"')
            print("  Time={}\". {}".format(sdT,sTime))
      
            dt = time.time()-t00
            elapseT = timedelta(seconds=int(dt))
            remainT = dt*(loop/(self.CurLoop+1) - 1)
            remainT = timedelta(seconds=int(remainT))
            print("  Elapse={}\", Remain={}\"".format(elapseT,remainT))
                    
        return worstAccuracyRatio, bestAccuracyRatio, incCost, incAccuRatio


      
    def __ShowInfo_Cost_Accuracy(self, sTitle, n_data, lst_cost, lst_accuracy):
        
        if (n_data<=0):return
        if len(lst_cost)<=0: return
        
        print("\nCost_{}: {:.4f} -> {:.4f}".
              format(sTitle, lst_cost[0],lst_cost[-1]) )
        print("Accu_{}: {:.4f} -> {:.4f}".
              format(sTitle, lst_accuracy[0]/n_data,lst_accuracy[-1]/n_data) )
                
        
    def __Plot_Cost_Accuracy(self, fnDebug, sTitle, test_cost, test_accuracy,
                    loops, learnRate, lmbda ):                
        cMinLoopToPlotDot = 30     
        loop = len(loops)
                    
        rf.Set_FigureText(plt, "{} Cost (lr={:.4f}, lmbda={:.4f})".
                format(sTitle, learnRate, lmbda), "Loop", "Coast")                
        if loop<cMinLoopToPlotDot:
            plt.plot(loops, test_cost, "ro-")
        else:
            plt.plot(loops, test_cost, "r-")
        plt.savefig("{}_{}Cost.png".format( fnDebug, sTitle), format = "png")
        plt.show()                                                
        rf.Set_FigureText(plt, "{} Accuracy (lr={:.4f}, lmbda={:.4f})".
                format(sTitle, learnRate, lmbda), "Loop", "Accuracy")        
        if loop<cMinLoopToPlotDot:
            plt.plot(loops, test_accuracy, "bo-")
        else:
            plt.plot(loops, test_accuracy, "b-")
        plt.savefig("{}_{}AccuracyRatio.png".format(fnDebug, sTitle), format = "png")
        plt.show()        
                
        
    def __Train(self, training_data, loop, samplingStep, learnRate,
            test_data=None, lmbda=0.0, blInitialWeiBias=True, labelY_trainOnlyDigit=-1,
            doShuffle=True): 
        
        if blInitialWeiBias:
            self.BestAccuracyRatio = 0.0
        self.WorstAccuracyRatio = 1.0
        self.AverageAccuracyRatio = 0.0
        self.AverageCost = 0.0
        self.Train_Loop = 0
        self.Train_LearnRate = learnRate
        self.Train_Lmbda = lmbda
        self.Train_TimeSec = 0.0
                    
            
        if (None==training_data): return 
        
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 4)
        callFr = calframe[1][3]
        
        self._RvBaseNeuralNetwork__Show_TrainingInfo("{}()".format(callFr),
            training_data, labelY_trainOnlyDigit)
        
        
        #  如果是一樣的訓練集，則不初始化，可以累積訓練精度
        if blInitialWeiBias:
            for lyr in self.NeuralLayers:
                lyr.Initial_Neurons_Weights_Biases()
        
        
        #測試資料 test_data[10000] = [ x[784], y[1] ], [ x[784], y[1] ],..... 10000筆   
        worstAccuracyRatio = 1.0
        bestAccuracyRatio = self.BestAccuracyRatio        
        
        if (self.Get_LayersNum()<1):        
            return worstAccuracyRatio,bestAccuracyRatio 
        
        
        sConvLyr = "_CNN" if ([]!=self.Get_ConvolutionLayerID()) else ""        
        sDropOut = "_DropOut_{}".format(self.NetEnumDropOut.name) if self.NetEnableDropOut else ""        
        if (""!=sConvLyr) and (gFilterShareWeights):
            sConvLyr = "{}ShareWeis".format(sConvLyr)
            
        fnNetworkData1 = "{}{}{}{}".format(self.LogPath, self.__class__.__name__, 
            sConvLyr, sDropOut) 
        
        s2 = ""
        
        if self.SaveVideo:
            self.Set_VideoOutputPath ( "{}{}\\".format(self.LogPath,"Training" ) )       
            rfi.Delete_Files(self._RvBaseNeuralNetwork__VideoOutputPath, [".jpg",".png"])
            
        
        t0 = time.time()
        
        # 初始化參數 ----------------------------------------------------
        n_train = len(training_data)
        n_test = len(test_data) if None!=test_data else 0
        test_cost, test_accuracy, training_cost, training_accuracy = [], [], [], []  
        incAccuRatio, incCost =  0.0, 0.0
        loop = max(1, loop)
        
        if self.DebugLog:     
            # 定義繪圖的字型資料 
            font = {'family': 'serif',
                    'color':  'red',
                    'weight': 'normal',
                    'size': 12, }                          
                
            fnDebug = "{}{}_{}".format(self.LogPath, "DebugLog", 
                  self.NeuralLayers[-1].ClassActivation.__name__)  
            print("\nTraining({}), Sampling({}), Testing({}) ".
                  format( n_train, int(n_train/samplingStep)+1, n_test ))
            print("loop({}), stepNum({}), learnRate({}), lmbda({})\n".
                  format(loop, samplingStep, learnRate, lmbda))                
            print("#loop : Cost(), Accuracy()".format(loop) )        
                
        t00 = time.time()
        loops = np.arange(0,loop,1)
        for j in range(loop):   
            self.CurLoop = j
            
            t0 = time.time() 
                    
            if doShuffle:
              self.RandomState.shuffle(training_data) #隨機打亂測試樣本順序
            # 在 500000 筆訓練資料內，從0開始，累加10取樣，
            # 共 500000/10 = 1000筆mini_trainingData[]
            # [0~10], [10~20], [20~30]......[49990~49999]
            mini_trainingData = [
                training_data[k:k+samplingStep] 
                for k in range(0, n_train, samplingStep)]
            
            
            # 每次新 loop 時，才更新 DropOutMask -------------------
            # 只有在每次新的 loop 時，每間格 2 次才更新 DropOut 的神經元
            if self.NetEnableDropOut and ((j+1)%gDropOutPerLoop==0) and (loop-1 != j): #不是最後一筆
                s2 = "DropOut({}, Ratio={})".format(self.NetEnumDropOut.name, self.NetDropOutRatio)
                # 只更新隱藏層，輸出層[:=1]不更新-------------
                for lyr in self.NeuralLayers[:-1]:
                    lyr.Set_DropOut(True, self.NetEnumDropOut, self.NetDropOutRatio)    
            else: s2 = ""
                        
            
            if self.DebugLog: print("\n#{}: {}".format(j, s2))
                            
            # 利用取樣集合 mini_trainingData[]，逐一以每個小測試集，　更新 weights 和 biases 
            for mini_batch in mini_trainingData:
                self._RvBaseNeuralNetwork__Update_LayersNeurons_Weights_Biases( \
                       mini_batch, learnRate, labelY_trainOnlyDigit)
                            
            #if 測試集開始之前，要先關掉 DropOut,測試集不能使用 DropOut
            for lyr in self.NeuralLayers: lyr.Set_DropOut(False)
                
                
            # 根據更新後的 Weights, Biases 重新計算 training_data的 cost, accuracy
            training_cost, training_accuracy = self.__Update_List_Cost_Accuracy(\
                'Train', training_data, lmbda, 
                training_cost, training_accuracy, createLabelY=False, 
                plotOutput=(self.DoPlotOutput and None==test_data))             
                
            # 輸入 test_data 預測結果 ---------------------------------------  
            if (test_data):
                test_cost, test_accuracy = self.__Update_List_Cost_Accuracy(\
                  'Test', test_data, lmbda, 
                   test_cost, test_accuracy, createLabelY=True, 
                   plotOutput=(self.DoPlotOutput and None!=test_data))   
                if self.DebugLog:
                  print("  {}: Cost({:.4f}), Accuracy({:.4f})".
                      format('Train', training_cost[-1], training_accuracy[-1]/n_train))
                worstAccuracyRatio, bestAccuracyRatio, incCost, incAccuRatio = \
                  self.__Update_Best_Cost_Accuracy(fnNetworkData1, 'Test', 
                      loop, t00, t0, 
                      n_test, test_cost, test_accuracy,
                      worstAccuracyRatio, bestAccuracyRatio, incCost, incAccuRatio)
                  
                if self.DoPloatActivations:
                    aId = random.randint(0,len(test_data)-1)
                    lyrsNeuActs = self.Get_LayersNeuronsActivations(test_data[aId][0])
                    #先劃出 Input
                    self.Plot_LayerNeuronsActivations(-1, test_data[aId][0])
                    # 在劃出所有層的 Activions
                    self.Plot_LayersNeuronsActivations(lyrsNeuActs)
                      
            else:
                worstAccuracyRatio, bestAccuracyRatio, incCost, incAccuRatio = \
                  self.__Update_Best_Cost_Accuracy(fnNetworkData1, 'Train', 
                      loop, t00, t0, 
                      n_train, training_cost, training_accuracy,
                      worstAccuracyRatio, bestAccuracyRatio, incCost, incAccuRatio)
                
                if self.DoPloatActivations:
                    aId = random.randint(0,len(training_data)-1)
                    lyrsNeuActs = self.Get_LayersNeuronsActivations(training_data[aId][0])
                    #先劃出 Input
                    self.Plot_LayerNeuronsActivations(-1, training_data[aId][0])
                    # 在劃出所有層的 Activions
                    self.Plot_LayersNeuronsActivations(lyrsNeuActs)
                   
                      
                    
        if self.DebugLog:           
            self.__ShowInfo_Cost_Accuracy('Train', n_train, training_cost, training_accuracy)
            self.__ShowInfo_Cost_Accuracy('Test', n_test, test_cost, test_accuracy)
            
        if self.DoPlotTraingTrend:
            rf.DrawFigures(plt, fnDebug, font, learnRate, lmbda, loops, training_cost,
                  test_cost, n_train, n_test,training_accuracy,test_accuracy)
        
            if (test_data):              
                self.__Plot_Cost_Accuracy(fnDebug, 'Test', test_cost, test_accuracy,
                    loops, learnRate, lmbda, )
            else:        
                self.__Plot_Cost_Accuracy(fnDebug, 'Train', training_cost, training_accuracy,
                    loops, learnRate, lmbda, )
            
        
        self.WorstAccuracyRatio = worstAccuracyRatio
        self.BestAccuracyRatio = bestAccuracyRatio
        self.AverageAccuracyRatio = incAccuRatio/loop
        self.AverageCost = incCost/loop
        
        self.Train_TimeSec = time.time()-t0
        
        
        if self.DoPloatWeights:
#            path = self.Get_VideoOutputPath()+"WeightsImage\\"
#            rfi.Delete_Files(path, [".jpg",".png"])
#            rfi.ForceDir(path)
#            fn = path+"WeiImg.jpg"
            fn = ""
            self.Plot_LayersNeuronsWeights(fn)
        
        if self.SaveVideo:        
            sampleNum=loop
            durationSec = min(1.0, 10/sampleNum)
            aviFn = "{}{}".format(self._RvBaseNeuralNetwork__VideoOutputPath, 
                "{}.avi".format(self.__class__.__name__), 1.0)
            if ru.ImageFilesToAvi(self._RvBaseNeuralNetwork__VideoOutputPath, aviFn, durationSec):
                os.system(r'start ' + aviFn)
        
        
        
        
    
    """=============================================================
    Public :
    ============================================================="""      
    #---------------------------------------------------------------------
    # Main Functions
    #--------------------------------------------------------------------- 
    def Train(self, training_data, loop, samplingStep, learnRate,
            test_data=None, lmbda=0.0, blInitialWeiBias=True, trainOnlyDigit=-1):
        
        #Funciton Initialization ------------------------------
        self.Caculate_Sum_LayersCostDerivations = \
            self._RvBaseNeuralNetwork__Caculate_Sum_LayersCostDerivations_Normal
        self.Total_Cost = self._RvBaseNeuralNetwork__Total_Cost_Normal
        self.Accuracy = self._RvBaseNeuralNetwork__Accuracy_Normal
        
        # 最後一層必須強制為 sigmoid，限制輸出(0.0 ~ 1.0)之間
        self.NeuralLayers[-1].Set_EnumActivation(nm.EnumActivation.afSigmoid)
        
        self.SaveVideo = False
        self.__Train(training_data, loop, samplingStep, learnRate,
                              test_data, lmbda, blInitialWeiBias, trainOnlyDigit)
        
        return self.WorstAccuracyRatio, self.BestAccuracyRatio     
    
    # 預測結果--------------------------------------------------------------
    def Evaluate_Accuracy(self, test_data):
        correctNum = self._RvBaseNeuralNetwork__Get_CorrectNum(test_data)
        n_test = len(test_data)
        return correctNum, n_test
    

    # 預測某張圖的數字 ------------------------------------
    def Predict_Digit(self, oneImgDigit, plotDigit=False):
        #caculation = self._RvBaseNeuralNetwork__Caculate_OutputActivations_FeedForward(oneImgDigit[0])
        caculation = self.Get_OutputValues(oneImgDigit[0])
#        print(caculation)
        # oneImgDigit[0] = pixels[784]
        # oneImgDigig[1] = label or labels[10]
        label = -1
        if type(oneImgDigit[1])==np.ndarray:
#            for i in range(0,len(oneImgDigit[1])):
#                if oneImgDigit[1][i]==1.0: 
#                    label=i
#                    break
            label = np.argmax(oneImgDigit[1]) # 取得陣列中最大數所在的index
        elif type(oneImgDigit[1])==np.int64:
            label = oneImgDigit[1]
        else:
            label = -1          
            
        result = -1
        value = 0.0
        
        if type(caculation)==np.ndarray:
#            maxVal = 0.0
#            for i in range(0,len(caculation)):
#                if caculation[i]>maxVal: 
#                    maxVal = caculation[i]
#                    result=i
            result = np.argmax(caculation) # 取得陣列中最大數所在的index
            value = caculation[result]
        elif type(caculation)==np.int64:
            result = caculation
        else:
            result = -1     
            
        if plotDigit: rf.Plot_Digit(oneImgDigit)
        
        return label, result, value[0]
    
    
    
    
        
#%%  ***************************************************************************

class RvNeuralEnDeCoder(RvNeuralNetwork, object):
    
    """=============================================================
    Static:
    ============================================================="""  
    @staticmethod
    def Load_DigitImages(digitImgPath, imgPxls):
        if not rfi.PathExists(digitImgPath): return # os.path.isdir(digitImgPath): return
        if imgPxls<=0:return
        imgW = int(np.sqrt(imgPxls))
        
        digitImgs = []
        for i in range(10):
            fn = digitImgPath + "{}.jpg".format(i)
            image = np.array(Image.open(fn))          
            # 將 cv2 的 BGR 轉成 image的 RGB--------------------------
            #image = ru.CvBGR_To_RGB(image)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #gray = cv2.bitwise_not(gray)  #反相
            gray = cv2.resize(gray, (imgW, imgW))  
#            if Debug: 
#                plt.title('result={},  Label={}'.format(i,i))
#                plt.imshow(gray, cmap='gray')
#                plt.show()
            #gray = cv2.resize(gray, (1,imgPxls) )/255.0 圖形錯誤
            gray = gray.reshape((imgPxls,1) )/255.0
            if Debug: 
                rf.Plot_Digit([gray.transpose(),0], i,i)
            digitImgs.append(gray)
            
        return digitImgs
    
    
    """=============================================================
    Constructor :
    ============================================================="""       
    def __init__(self,  *args):
        super().__init__(*args) #加上此，產生所有 parent 的 members 
        #self.__newMember = xxxxx
        
        # Encoder, Decoder 所有層必須為 sigmoid  ---------------------------
        for lyr in self.NeuralLayers:
            lyr.Set_EnumActivation(nm.EnumActivation.afSigmoid)
            
        if Debug and self.__class__.__name__=='RvNeuralEnDeCoder':  
            self.Show_LayersInfo()  
    
    
    
    """=============================================================
    Private :
    ============================================================="""     
    
       
    
    
    """=============================================================
    Public :
    ============================================================="""     
    def Get_Encoder_Decoder(self):
        bottleNeckLyr = int(len(self.NeuralLayers)/2 - 1)
        encoderStLyrId = 0
        decoderStLyrId = bottleNeckLyr+1
        
        encoder = RvNeuralNetwork( [ lyr.Create_LayerObj(lyr) 
            for lyr in self.NeuralLayers[encoderStLyrId:bottleNeckLyr+1] ] )
        encoder.WorstAccuracyRatio = self.WorstAccuracyRatio
        encoder.BestAccuracyRatio = self.BestAccuracyRatio
        
        decoder = RvNeuralNetwork([lyr.Create_LayerObj(lyr) 
            for lyr in self.NeuralLayers[decoderStLyrId:] ] )  
        decoder.WorstAccuracyRatio = self.WorstAccuracyRatio
        decoder.BestAccuracyRatio = self.BestAccuracyRatio      
            
        return encoder, decoder    
    
       
       
    def Build_Encoder_Decoder(self, training_data, loop, samplingStep, learnRate,
          lmbda=0.0, blInitialWeiBias=True, trainOnlyDigit=-1):  
        
        if self.NeuralLayers[0].Get_InputNum() != self.NeuralLayers[-1].Get_NeuronsNum():
          if Debug: print("Input Dimention should equal Output Dimention here")
          return 
        
        #Funciton Initialization ------------------------------
        self.Caculate_Sum_LayersCostDerivations = \
                self._RvBaseNeuralNetwork__Caculate_Sum_LayersCostDerivations_EnDeCoder
        self.Total_Cost = self._RvBaseNeuralNetwork__Total_Cost_EnDeCoder
        self.Accuracy = self._RvBaseNeuralNetwork__Accuracy_EnDeCoder
                
        
        # 最後一層必須強制為 sigmoid，限制輸出(0.0 ~ 1.0)之間
        self.NeuralLayers[-1].Set_EnumActivation(nm.EnumActivation.afSigmoid)
                    
        for lyr in self.NeuralLayers:
            if lyr._RvNeuralLayer__EnumActivation != af.afSigmoid:
                print("Use Sigmoid for all layers or it will be overfitting")
                break            
            
        self._RvNeuralNetwork__Train(training_data, loop, samplingStep, learnRate,
               None, lmbda, blInitialWeiBias, trainOnlyDigit)
        
        return self.Get_Encoder_Decoder()
    
    
                
                
    def Build_Encoder_Decoder_AssignOutputY(self, training_data, loop, samplingStep, learnRate,
          lmbda=0.0, blInitialWeiBias=True, trainOnlyDigit=-1):  
        
        if len(training_data[0][1]) != self.NeuralLayers[-1].Get_NeuronsNum():
          if Debug: print("training Data LableY Dimension should equal to Output Layer Neurons Num.")
          return 
        
        #Funciton Initialization ------------------------------
        self.Caculate_Sum_LayersCostDerivations = \
          self._RvBaseNeuralNetwork__Caculate_Sum_LayersCostDerivations_EnDeCoder_AssignOutputY
        self.Total_Cost = self._RvBaseNeuralNetwork__Total_Cost_EnDeCoder_AssignOutputY
        self.Accuracy = self._RvBaseNeuralNetwork__Accuracy_EnDeCoder_AssignOutputY
                
        # 最後一層必須強制為 sigmoid，限制輸出(0.0 ~ 1.0)之間
        self.NeuralLayers[-1].Set_EnumActivation(nm.EnumActivation.afSigmoid)        
            
        for lyr in self.NeuralLayers:
            if lyr._RvNeuralLayer__EnumActivation != af.afSigmoid:
                print("Use Sigmoid for all layers or it will be overfitting")
                break
            
#        self.__Build_Encoder_Decoder(training_data, loop, samplingStep, learnRate,
#          lmbda, blInitialWeiBias, trainOnlyDigit)
        
        self._RvNeuralNetwork__Train(training_data, loop, samplingStep, learnRate,
               None, lmbda, blInitialWeiBias, trainOnlyDigit)
        
        return self.Get_Encoder_Decoder()
    
    
    
    
    
        
#%%  ***************************************************************************

class RvNeuralDiscriminator(RvNeuralNetwork, object):
    
    """=============================================================
    Static:
    ============================================================="""  
    @staticmethod    
    def Create_FakeData_Generator(generator, fakeDataNum, minVal=0.2, maxVal=0.8):        
        if None==generator or fakeDataNum<=0:return None
        
        randomState =  np.random.RandomState(int(time.time())) #generator.RandomState
        inputNum = generator.Get_InputNum()        
#        encode = randomState.randn(inputNum,1)   
#        encode = np.maximum(0.2, np.minimum(0.8, encode))                          
#        output = generator.Get_OutputValues(encode)    
        
        fakeId = 0
        FakeDigit_set = [ 
            tuple([ 
               np.array(generator.Get_OutputValues(
                  np.maximum( minVal, np.minimum(maxVal, randomState.randn(inputNum,1)) )) ),  
               np.array([[fakeId]])  ]) 
            for i in range(fakeDataNum) ]
    
        return FakeDigit_set
    

    @staticmethod
    def Create_FakeData_RandomNoise(lstRealData, fakeDataNum ):
        randomState = np.random.RandomState(int(time.time()))
        fakeId=0
        FakeDigit_set = [ 
            tuple([ randomState.randn(len(lstRealData[0][0]),1) , [[fakeId]] ]) 
            for aTuple in range(fakeDataNum) ]
        return FakeDigit_set
        
        

    @staticmethod
    def Create_RealData(lstRealData, dataNum=-1 ):
        
        if dataNum==-1: stepNum = len(lstRealData)
        else: stepNum = min(len(lstRealData), dataNum)
        
        #lstRealCopy = lstRealData.copy()
        
        randomState = np.random.RandomState(int(time.time()))
        
        randomState.shuffle(lstRealData)
        # 利用取樣集合 mini_trainingData[]，逐一以每個小測試集，　更新 weights 和 biases 
        TrueDigit_sets = [ lstRealData[k:k+stepNum] 
            for k in range(0, len(lstRealData), stepNum)]
        selId = randomState.randint(len(TrueDigit_sets))
        
        trueId=1
        #fakeId=0
        
        """image[:,:] = [[ min(pixel + dVal, 255) 
            for pixel in row] for row in image[:,:]]"""
        
        MixDigit_set = [ 
            tuple([ aTuple[0], [[trueId]]  ]) 
            for aTuple in TrueDigit_sets[selId] ]
        
#        FakeDigit_set = [ 
#            tuple([ randomState.randn(len(aTuple[0]),1) , [[fakeId]] ]) 
#            for aTuple in TrueDigit_sets[0] ]
#            
#        MixDigit_set = MixDigit_set + FakeDigit_set
#        
#        randomState.shuffle(MixDigit_set)
        
        return MixDigit_set

            
    
    """=============================================================
    Constructor :
    ============================================================="""       
    def __init__(self,  *args):
        super().__init__(*args) #加上此，產生所有 parent 的 members 
        
        # Encoder, Decoder 所有層必須為 ReLu, 最後一層sigmoid  ---------------------------
#        for lyr in self.NeuralLayers:
#            lyr.Set_EnumActivation(nm.EnumActivation.afSigmoid)
        self.NeuralLayers[-1].Set_EnumActivation(nm.EnumActivation.afSigmoid)  
        
        #self.__newMember = xxxxx
        if Debug and self.__class__.__name__=='RvNeuralDiscriminator':  
            self.Show_LayersInfo()  
    
    
    
    """=============================================================
    Private :
    ============================================================="""     
    
    
    
    """=============================================================
    Public :
    ============================================================="""     
    def Train_Discriminator(self, training_data, loop, samplingStep, learnRate,
          lmbda=0.0, blInitialWeiBias=True, traninOnlyLabelY=-1, doShuffle=True):  
                  
        if 1 != self.NeuralLayers[-1].Get_NeuronsNum():
          if Debug: print("Output Dimention should be 1 for Discriminator")
          return None, None
        
        #Funciton Initialization ------------------------------
        self.Caculate_Sum_LayersCostDerivations = \
                self._RvBaseNeuralNetwork__Caculate_Sum_LayersCostDerivations_Discriminator
        self.Total_Cost = self._RvBaseNeuralNetwork__Total_Cost_Discriminator
        self.Accuracy = self._RvBaseNeuralNetwork__Accuracy_Discriminator
            
        # 最後一層必須強制為 sigmoid，限制輸出(0.0 ~ 1.0)之間
        self.NeuralLayers[-1].Set_EnumActivation(nm.EnumActivation.afSigmoid)
                        
        if traninOnlyLabelY==0: sTrueFake='False'
        elif traninOnlyLabelY==1: sTrueFake='True'
        else: sTrueFake='True & False'
        
        if Debug:
            print("Train_Discriminator({})".format(sTrueFake))
        
        return self._RvNeuralNetwork__Train(training_data, loop, samplingStep, learnRate,
            None, lmbda, blInitialWeiBias, traninOnlyLabelY, doShuffle)
      
        
        
        

    
    
#%% ***************************************************************************
    
class RvNeuralGAN(RvNeuralNetwork, object):
    """=============================================================
    Static:
    ============================================================="""     
#    @staticmethod
#    def LayersNeurons_To_RvNeuralLayers( layersNeuronsNum ):
#        return [ RvNeuralLayer(inputNeusNum, lyrNeusNum) 
#            for inputNeusNum, lyrNeusNum in \
#                zip(layersNeuronsNum[:-1], layersNeuronsNum[1:]) ]
        
    
    
    """=============================================================
    Constructor :
    ============================================================"""        
    # 代替 overload constructor  
    def __init__(self,  *args):           
        self.__Initial_Members()
        
        super().__init__() #加上此，產生所有 parent 的 members 
        
        nArgs = len(args)
        if nArgs>0:            
            if isinstance(args[0], list):  
                if isinstance(args[0][0], object): #RvNeuralNetwork): 
                    self.__Create_FromNetworksList(args[0])
            elif isinstance(args[0], str): # networkFileName
                self.__Create_FromFile(*args)              
            elif isinstance(args[0], object): # netObj
                self.__Create_FromNetworks(*args)            
            else:
                print("Need [Generator, Discriminator, Encoder]") 
        else:
            print("Need [Generator, Discriminator,se Encoder]")    
            
        #self.__newMember = xxxxx      
        if Debug and self.__class__.__name__=='RvNeuralGAN':  
            self.Show_LayersInfo()  
            
    
    
    """=============================================================
    Private :
    ============================================================="""  
    def __Create_FromNetworks(self, *args):
        nArgs = len(args)
        if (nArgs<3): return
        # 需要三個 network [Generator, Discriminator, Encoder]
        allIsNetObj = True
        for i in range(nArgs):
            if not isinstance(args[0],object): # RvNeuralNetwork):  
                allIsNetObj = False
                break
        if allIsNetObj:
            """ RvNeuralNetwork() 會Create 新的 Network
            self.DiscriminatorLayerStartIndex = \
              len(RvNeuralNetwork(args[0]).NeuralLayers)-1
            self.NeuralLayers = \
              RvNeuralNetwork(args[0]).NeuralLayers +\
              RvNeuralNetwork(args[1]).NeuralLayers
            self.Encoder = RvNeuralNetwork(args[2])"""
            self.DiscriminatorLayerStartIndex = \
              len((args[0]).NeuralLayers)-1
            self.NeuralLayers = \
              (args[0]).NeuralLayers +\
              (args[1]).NeuralLayers
            self.Encoder = (args[2])
            
                                        
              
    def __Create_FromNetworksList(self, lstNetworks):
        if isinstance(lstNetworks, list) and len(lstNetworks)>=3:
        # 需要三個 network [Generator, Discriminator, Encoder]
           self.__Create_FromNetwork(\
             lstNetworks[0], lstNetworks[1], lstNetworks[3] )
        


    def __Initial_Members(self):
        self.Encoder = None
        self.DiscriminatorLayerStartIndex = 0


    
    """=============================================================
    Public :
    ============================================================="""  
    def Get_Generator_Discriminator(self):
        dsmtrStLyr = self.DiscriminatorLayerStartIndex
        generatorStLyrId = 0
        
        generator = RvNeuralNetwork( [ lyr.Create_LayerObj(lyr) 
            for lyr in self.NeuralLayers[generatorStLyrId:dsmtrStLyr+1] ] )
        generator.WorstAccuracyRatio = self.WorstAccuracyRatio
        generator.BestAccuracyRatio = self.BestAccuracyRatio
        
        discriminator = RvNeuralDiscriminator([lyr.Create_LayerObj(lyr) 
            for lyr in self.NeuralLayers[dsmtrStLyr+1:] ] )  
        discriminator.WorstAccuracyRatio = self.WorstAccuracyRatio
        discriminator.BestAccuracyRatio = self.BestAccuracyRatio      
        
        
        generator.DoPlotTraingTrend = self.DoPlotTraingTrend
        generator.DoPloatActivations = self.DoPloatActivations
        generator.DoPloatWeights = self.DoPloatWeights
        generator.DoPlotOutput = self.DoPlotOutput
        generator.SaveVideo = self.SaveVideo
        
        discriminator.DoPlotTraingTrend = self.DoPlotTraingTrend
        discriminator.DoPloatActivations = self.DoPloatActivations
        discriminator.DoPloatWeights = self.DoPloatWeights
        discriminator.DoPlotOutput = self.DoPlotOutput
        discriminator.SaveVideo = self.SaveVideo
            
        return generator, discriminator    
        
        
        
        
    def Train_GAN(self, true_data, fakeDataNum, loop, samplingStep, learnRate,
          lmbda=0.0, fake_data=None, atReality=0.9):  
                  
        if 1 != self.NeuralLayers[-1].Get_NeuronsNum():
          if Debug: print("Output Dimention should be 1 for Discriminator")
          return None, None
      
        
        NoMoreUpdate = False
        updateWithTrueData = True
        minCodeVal, maxCodeVal = 0.0, 1.0
        
        MorphingAlongSameShape = False
        if MorphingAlongSameShape: #降低重新產生數字的標準 
            minDiscriminatorVal = 0.5  # 只有真實度到這個值，才能跳出
        else:
            minDiscriminatorVal = atReality 
        
        
        #Funciton Initialization ------------------------------
        self.Caculate_Sum_LayersCostDerivations = \
                self._RvBaseNeuralNetwork__Caculate_Sum_LayersCostDerivations_GAN
        self.Total_Cost = self._RvBaseNeuralNetwork__Total_Cost_GAN
        self.Accuracy = self._RvBaseNeuralNetwork__Accuracy_GAN
            
        # 最後一層必須強制為 sigmoid，限制輸出(0.0 ~ 1.0)之間
        self.NeuralLayers[-1].Set_EnumActivation(nm.EnumActivation.afSigmoid)
        
        dsmtrStLyr = self.DiscriminatorLayerStartIndex
        
        samplingStep = min(samplingStep, fakeDataNum)
        
        
        
        # 產生假 Imgage -------------------------------------
        generator, discriminator = self.Get_Generator_Discriminator()
        
        if (None==fake_data):
          genImages = RvNeuralDiscriminator.Create_FakeData_Generator(\
            generator, fakeDataNum, minCodeVal, maxCodeVal)
        else:
          genImages = fake_data          
        imgNum= len(genImages)
        
        trueSampleNum = imgNum #min(500, len(true_data))  # imgNum
          
#        if Debug:
#          for i in range(5):
#            #先劃出 Input
#            self.Plot_LayerNeuronsActivations(-1, genImages[i][0])
          
        if Debug:
            fakeImages = [ [] for img in range(imgNum) ]
            pxls = len(genImages[0][0])
            pxlW = int(np.sqrt(pxls))
            pxls = pxlW*pxlW
            nCol = 5
            nRow = int(imgNum/nCol)+1
            pltInchW = Get_PlotWidthInch(pxlW, nCol) #digitFigs = {title : [] for title in range(10)}
            
            vdoPath = "{}\\{}\\".format(self.LogPath , "GAN")
            self.Set_VideoOutputPath(vdoPath) 
            rfi.Delete_Files( vdoPath, [".jpg",".png"])
        
            for i in range(imgNum):
                fakeImages[i] = \
                  np.array(genImages[i][0][:pxls].transpose()).reshape(pxlW,pxlW)*255
            saveFn = "{}{}".format(vdoPath, 
                "img{:03}_accur{:.4f}.jpg".format(0,0.0 ))  
            pltFn.Plot_Images( np.array(fakeImages),nRow,nCol, 
                ["FakeInput[{}]".format(imgNum)], saveFn, pltInchW)
        
        
        
        #aId = random.randint(0,len(genImages)-1)
                
        loopDiscriminator = 1
        loopGan = 1
        priGanOutputs = [0.0 for i in range(imgNum)]
                
        self.DoPlotTraingTrend = False
        self.DoPloatActivations = False
        self.DoPloatWeights = False
        self.DoPlotOutput = False
        self.SaveVideo = False
        self.DebugLog = False
        
        
        for iloop in range(loop):   
            print("GAN loop: {}/{}".format(iloop,loop))        
            
            genCodes = [ 
                tuple([  self.Encoder.Get_OutputValues(imgTupple[0]), [[0]]  ])  
                for imgTupple in genImages]
            
            # 先固定 discriminator， 以 label=1 來更新 generator，產生接近真實圖像
            for iLyr in range(len(self.NeuralLayers)):
                self.NeuralLayers[iLyr].DoUpdateNeuronsWeightsBiases = False   
            for iLyr in range(0, dsmtrStLyr+1):
                  self.NeuralLayers[iLyr].DoUpdateNeuronsWeightsBiases = True
            # 開始以假 genCodes[] 當作是真實圖片，訓練 GAN model
            labelY = 1   # 要傳拷貝本進去 genCodes.copy()，以免順序被打亂
            self._RvNeuralNetwork__Train(genCodes.copy(), loopGan, samplingStep, learnRate,
                None, lmbda, False, labelY, doShuffle=True) 
            
#            if updateWithTrueData:
#                labelY = 1 
#                trueCodes = [ 
#                    tuple([  self.Encoder.Get_OutputValues(imgTupple[0]), [[0]]  ])  
#                    for imgTupple in true_data[0:imgNum]]
#                self._RvNeuralNetwork__Train(trueCodes, loopGan, samplingStep, learnRate,
#                    None, lmbda, False, labelY, doShuffle=True) 
            
            
              # 輸入假資料訓練 Discriminator -----------------------------------                             
            generator, discriminator = self.Get_Generator_Discriminator()            
            # 以true_data更新 discriminator 所有Weights ------------------------------
            if updateWithTrueData:
                self.RandomState.shuffle(true_data)  
                # Fake 和 Real 數量需一致，才會有較好結果，不會偏重任一方，
                mix_data = true_data[0:trueSampleNum] + genImages
                self.RandomState.shuffle(mix_data)  
                discriminator.DoPlotOutput = False
                discriminator.Train_Discriminator(\
                    mix_data, loopDiscriminator, samplingStep, learnRate, lmbda,  False,
                    -1,  doShuffle=True)
            else:#加強假資料判斷-------------------------------------
                # 以fake_data更新 discriminator 所有Weights ------------------------------   
                discriminator.DoPlotOutput = False
                # 要傳拷貝本進去 genImages.copy()，以免順序被打亂
                discriminator.Train_Discriminator(\
                        genImages.copy(), loopDiscriminator, samplingStep, learnRate, lmbda,  False,
                        -1,  doShuffle=True)                    
            # 重新整合 Generator, Discriminator-------------------------------
            self.NeuralLayers = generator.NeuralLayers + discriminator.NeuralLayers
            
                
#            if Debug:
#                lyrsNeuActs = self.Get_LayersNeuronsActivations(genCodes[aId][0])
#                self.Plot_LayerNeuronsActivations(dsmtrStLyr, lyrsNeuActs[dsmtrStLyr])
                
                       
             
            # 取得 Discriminator Layer 的圖像當作 Input Image 給 Encoder，
            # 產生下次新的 code 給 Generator  將 dsmtryStLyr 的影像值，轉成 code，當作新的輸入
            iCode=0
            allBiggerThanReality = True
            imgUpdated = False
            for code in genCodes:     
               if priGanOutputs[iCode]<atReality:
                   allBiggerThanReality=False                     
               lyrsNeuActs = self.Get_LayersNeuronsActivations(code[0])   
               dsmtrOutputVal = lyrsNeuActs[-1][0][0]
               
               if dsmtrOutputVal>=priGanOutputs[iCode]: #update only better
                   genImages[iCode] = tuple( [ lyrsNeuActs[dsmtrStLyr], [[0]]  ]) 
                   priGanOutputs[iCode] = dsmtrOutputVal
                   imgUpdated = True
               elif dsmtrOutputVal<minDiscriminatorVal:# Discriminator太小，稀疏，重新產生
                  fakeInput = generator.Get_OutputValues(\
                    np.maximum( minCodeVal, 
                      np.minimum(maxCodeVal, 
                        self.RandomState.randn(generator.Get_InputNum(),1)
                        #genCodes[iCode][0]*1.1
                                )
                              ) )                   
                  genImages[iCode] = tuple([ fakeInput, genImages[iCode][1]])   
                  priGanOutputs[iCode] = dsmtrOutputVal
                  imgUpdated = True
                  
               if MorphingAlongSameShape: #降低重新產生數字的標準 
                  minDiscriminatorVal = min(minDiscriminatorVal,dsmtrOutputVal )
                  
               iCode+=1
               
               
            if not imgUpdated:
                NoMoreUpdate = True
                print("No more updates, stop at loop \"{}\"\n".format(iloop))
                break            
#            if allBiggerThanReality:
#                print("All better then atReality({}), stop at loop \"{}\"\n".format(atReality, iloop))
#                break
                
#            # Add Noise----------------------------
#            genImages = [ 
#                tuple( [ RvBaseNeuralNetwork.Add_Noise(imgTuple[0], 0.2), [[0]] ])
#                for imgTuple in genImages ]
                
            
            if Debug:
                sTitles = ["" for i in range(imgNum)]
                for i in range(imgNum):
                    fakeImages[i] = \
                      np.array(genImages[i][0][:pxls].transpose()).reshape(pxlW,pxlW)*255
                    sTitles[i] = "({:.4f})".format(priGanOutputs[i])  
                saveFn = "{}{}".format(vdoPath, 
                    "img{:03}_accur{:.4f}.jpg".format(iloop+1,priGanOutputs[i] ))  
                pltFn.Plot_Images( np.array(fakeImages),nRow,nCol, 
                    sTitles, saveFn, pltInchW)
                
          
                
                
            
            
        # 回復更新所有權種
        for iLyr in range(len(self.NeuralLayers)):
            self.NeuralLayers[iLyr].DoUpdateNeuronsWeightsBiases = True
            
#             
        self.DoPlotTraingTrend = Debug
        self.DoPloatActivations = Debug
        self.DoPloatWeights = Debug        
        self.DoPlotOutput = Debug
        self.SaveVideo = Debug
        self.DebugLog = Debug
                
        
        if self.SaveVideo:
            durationSec = 1.0 #min(1.0, 10/loop)
            aviFn = "{}{}".format(vdoPath, "Gan.avi")        
            if ru.ImageFilesToAvi(vdoPath, aviFn, durationSec ):
                rfi.OpenFile(aviFn)
        
      
        generator, discriminator = self.Get_Generator_Discriminator()
        
               
        return generator, discriminator, genImages, NoMoreUpdate
    
    
    
    
    
    
    
        
#%%        
        
        
        
        