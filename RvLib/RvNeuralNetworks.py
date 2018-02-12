# -*- coding: utf-8 -*-
# coding: utf-8 　　　←表示使用 utf-8 編碼，加上這一行遇到中文註解才能編譯成功

#%%
"""
Created on Thu Feb  8 14:36:42 2018
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


RvNeuralNetworks.py

 1. 隨機梯度下降法 SGD() 更新 Weights, Biases
 2. 二種 Cost Classes :
     Cost_Quadratic, Cost_CrossEntropy 
  
"""


#%% Libraries

# Standard library----------------------------------------------
import sys
import os
import json
from enum import Enum
import time
#from datetime import datetime


# Third-party libraries------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
#matplotlib.use("Agg")
#import matplotlib.animation as animation


# prvaite libraries---------------------------------------------
import RvActivationCost as ac
from RvActivationCost import EnumDropOutMethod as drpOut
import RvMiscFunctions as rf
from RvGVariable import *


#%%  Directive 定義 ----------------------------------------------------
Debug = True
Debug_Plot = Debug
SaveVideo = Debug

gDropOutRatio = 0.5
gDropOutPerEpcho = 2



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
            else:
                print("Need InputNum and NeuronNum")        
        else:
            print("Need InputNum and NeuronNum")    
        if Debug: print("{} : Input({}), Neurons({})".
            format(self.__class__.__name__, self.Get_InputNum(), self.Get_NeuronsNum() ))
        
        
    def __Create_1Args(self, inOutNums,
                 enumActivation=ac.ActivationFunction.afReLU):  
        if len(inOutNums)<2: 
            print("Need InputNum and NeuronNum") 
            while (len(inOutNums)<2): inOutNums.append(1)   
        self.Initial_Neurons_Weights_Biases(inOutNums[0], inOutNums[1])   
        self.__EnumActivation = enumActivation 
        self.ClassActivation,self.ClassCost = \
            ac.Get_ClassActivation(enumActivation)  
            
            
    def __Create_4Args(self, inputNeuronsNum, oneLyrNeuronsNum,
                 enumActivation=ac.ActivationFunction.afReLU):   
        self.Initial_Neurons_Weights_Biases(inputNeuronsNum, oneLyrNeuronsNum)  
        self.__EnumActivation = enumActivation 
        self.ClassActivation,self.ClassCost = \
            ac.Get_ClassActivation(enumActivation)  
#        if Debug:          
#            print("Biase = {}\n".format(self.NeuronsBias) )
#            print("Weight = {}\n".format(self.NeuronsWeights) )
#            print("Cost Class : \"{}\"".format(self.ClassCost.__name__)) 
#            print("Activation Class : \"{}\"".
#                format(self.ClassActivation.__name__))  
            
    
    """=============================================================
    Static:
    ============================================================="""
    @staticmethod
    #### 從檔案讀取 RvNeuralLayer 參數 ----------------------------------------------
    def Create_RvNeuralLayer(filename):
        neuLyr = RvNeuralLayer([])
        neuLyr.Load_Neurons_Parameters(filename)
        return neuLyr
    
    
    
    """=============================================================
    Private :
    ============================================================="""    
    def __Initial_Members(self):     
        self.__RandomState = np.random.RandomState(int(time.time()))
        self.__DoDropOut = False
        self.__DropOutRatio = gDropOutRatio
        self.__EnumDropOut = drpOut.eoNone
        self.ClassActivation,self.ClassCost = \
            ac.Activation_ReLU, ac.Cost_Quadratic
    
            
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
        return len(self.NeuronsWeights)
    
    def Get_NeuronsValuesZ(self, prLyrInputX):
        # outputZ = Sum(wi*xi)+b
        return np.dot(self.NeuronsWeights, prLyrInputX) + self.NeuronsBiases
    
    def Get_NeuronsActivations(self, lyrNeusZs):
        return self.ClassActivation.activation(lyrNeusZs)
    
    def Get_LayerData(self, lyrIndex):
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
#        self.Initial_Neurons_Weights_Biases(data["InputNum"],data["NeuronsNum"])
#        # 要將 NeuronsWeights, NeuronsBiases 轉為 array, 否則會在 np.dot() 時候，
#        # 因為 list 和 array dot() 造成運算速度緩慢
#        self.NeuronsWeights = np.asarray(data["NeuronsWeights"])
#        self.NeuronsBiases = np.asarray(data["NeuronsBiases"] )       
        
        self.Update_LayerData(data["InputNum"],data["NeuronsNum"], \
           data["NeuronsWeights"], data["NeuronsBiases"])
         
        enumValue = data["EnumActivationValue"]
        enumActivation = ac.ActivationFunction( enumValue )
        self.__EnumActivation = enumActivation 
        self.ClassActivation,self.ClassCost = \
            ac.Get_ClassActivation(enumActivation)       
            
#        enumValue = data["EnumDropOutValue"]
#        enumDropOut = ac.EnumDropOutMethod( enumValue )
#        self.__EnumDropOut = enumDropOut
            
#        cls = getattr(sys.modules[__name__], data["ClassCost"])  
#        self.ClassCost = cls             
                    

    def Set_DropOut(self, doDropOut, \
            enumDropOut=drpOut.eoSmallActivation, \
            ratioDropOut=0.5):   
        self.__EnumDropOut = enumDropOut 
        self.__DoDropOut = doDropOut
        if doDropOut:  # a和 d(a) 都要 dropOut 
            self.__DropOutRatio = ratioDropOut
        
    
    
    
    
    # ----------------------------------------------------------
    # Main Functions 
    # ----------------------------------------------------------          
    def Initial_Neurons_Weights_Biases(self,inputNeuronsNum, oneLyrNeuronsNum):  
        """
        以 [L1, L2, L3] = [784, 50, 10] 為例
        Biases = [ [ # L2, 50個神經元, 50 x 1 array-----------------------------
                     [b1], 
                     ....   # L2, 50x1 個 bias
                     [b50] ],   
    
                   [ # L3, 10個神經元, 10 x 1 array ----------------------------
                     [b1],     
                     ...    # L3, 10x1 個 bias
                     [b10] ] ] 
        Weights = [ [ # L2, 50個神經元, 50 x 784 array--------------------------
                     [w1 ... w784], 
                     ....   # L2, 50x784 個 weight
                     [w1 ... w784] ],   
    
                    [ # L3, 10個神經元, 10 x 50 array --------------------------
                     [w1 ... w50],     
                     ...    # L3, 10x50 個 weight
                     [w1 ... w50] ] ] 
        """
        self.NeuronsBiases = self.__RandomState.randn(oneLyrNeuronsNum, 1)  
        self.NeuronsWeights = \
          self.__RandomState.randn(oneLyrNeuronsNum, inputNeuronsNum)/np.sqrt(inputNeuronsNum)
      
        
        
    # 建立一個和 self.NeuronsBiases 一樣的結構，並給予初值 ---------------
    def Create_ArrayOf_NeuronsBiases(self, initialValue):
        return np.full(self.NeuronsBiases.shape, initialValue)
    
    # 建立一個和 self.NeuronsBiases 一樣的結構，並給予初值 ---------------
    def Create_ArrayOf_NeuronsWeights(self, initialValue):
        return np.full(self.NeuronsWeights.shape, initialValue)
    
    def Caculate_dCost_OutputLayer(self, oneLyrNeusZs, oneLyrNeusActvs, \
            oneLyrNeusLabels,preLyrNeusActvs):
        # 計算最後一層輸出層的 cost 對 weight, bias 的偏微分-------------------
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
        oneLyrNeus_dCost_dWeis = np.dot(oneLyrNeusErrs, \
            preLyrNeusActvs.transpose()) 
        return oneLyrNeusErrs, oneLyrNeus_dCost_dWeis, oneLyrNeus_dCost_dBiases    
    
    
    def Caculate_dCost_HiddenLayer(self, preLyrNeusActvs, curLyrNeusZs, \
                nxtLyrNeusWeis, nxtLyrNeusErrs):
        # 非輸出層的 backPropagationagation，從倒數第二層開始 -------------------------------------------
        curLyrNeus_dA_dZ = self.ClassActivation.derivation(curLyrNeusZs)
        
        if self.__DoDropOut: # a和 d(a) 都要 dropOut 
            # 每次重算 DropOutMask 效果比較好-----------------
            curLyrNeus_dA_dZ = Get_NoneDropOutValues(curLyrNeus_dA_dZ, self.__DropOutRatio) 
            
        # error = dC/dA
        # dC/dB = error * dA/dZ
        # dC/dW = error*a * dA/dZ
        # 當前層的誤差 = 下一層誤差反推回來的誤差權重 * 當前層的激活值偏微分
        curLyrNeusErrs = np.dot(nxtLyrNeusWeis.transpose(),  
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
            oneInput = ac.ClassDropOut.Get_NonDropOutValues( self.__EnumDropOut,
                oneInput, self.__DropOutRatio)
            #oneInput = Get_BigValues(oneInput, self.__DropOutRatio)
            #oneInput = Get_NoneDropOutValues(oneInput, self.__DropOutRatio) 
            
        return oneLyrNeurosZs, oneInput   
                    
    def Update_LayerData(self, inputNum, lyrNeursNum, lyrNeursWeights, lyrBiases):
        self.Initial_Neurons_Weights_Biases(inputNum, lyrNeursNum)        
        # 要將 NeuronsWeights, NeuronsBiases 轉為 array, 否則會在 np.dot() 時候，
        # 因為 list 和 array dot() 造成運算速度緩慢
        self.NeuronsWeights = np.asarray(lyrNeursWeights)
        self.NeuronsBiases = np.asarray(lyrBiases)
        
    
    # ----------------------------------------------------------
    # File IO Functions ------
    # ----------------------------------------------------------
                
    def Load_Neurons_Parameters(self, filename):
        if not os.path.isfile(filename): return
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
        
class RvCovolutionLayer(RvNeuralLayer):
    
    """=============================================================
    Constructor:
    ============================================================="""
    def __init__(self, inputData, filter_shape, image_shape, poolsize=(2, 2), \
            enumActivation=ac.ActivationFunction.afReLU):  
        """
        :type rng: numpy.random.RandomState 
        :param rng: a random number generator used to initialize weights 
 
        3、input: 輸入特徵圖數據，也就是n幅特徵圖片 
 
        4、參數 filter_shape: (number of filters, num input feature maps, 
                              filter height, filter width) 
        num of filters：是卷積核的個數，有多少個卷積核，那麼本層的out feature maps的個數 
        也將生成多少個。num input feature maps：輸入特徵圖的個數。 
        然後接著filter height, filter width是卷積核的寬高，比如5*5,9*9…… 
        filter_shape是列表，因此我們可以用filter_shape[0]獲取卷積核個數 
 
        5、參數 image_shape: (batch size, num input feature maps, 
                             image height, image width)， 
         batch size：批量訓練樣本個數 ，num input feature maps：輸入特徵圖的個數 
         image height, image width分別是輸入的feature map圖片的大小。 
         image_shape是一個列表類型，所以可以直接用索引，訪問上面的4個參數，索引下標從 
         0~3。比如image_shape[2]=image_heigth  image_shape[3]=num input feature maps 
 
        6、參數 poolsize: 池化下採樣的的塊大小，一般為(2,2) 
        """  
        self.__Initial_Members()
        assert len(inputData)<=0
        assert image_shape[1] == filter_shape[1] #判斷輸入特徵圖的個數是否一致，如果不一致是錯誤的  
                
#        self.Initial_Neurons_Weights_Biases(inOutNums[0], inOutNums[1])   
        self.EnumActivation = enumActivation
        self.ClassActivation,self.ClassCost = \
            ac.Get_ClassActivation(enumActivation)  
        
    
    """=============================================================
    Static:
    ============================================================="""
#    @staticmethod
#    #### 從檔案讀取 Network參數 ----------------------------------------------
#    def Create_Layer(filename):        
#        return
    
    
    
    """=============================================================
    Private :
    ============================================================="""    
    def __Initial_Members(self):  # override          
        self.__RandomState = np.random.RandomState(int(time.time()))
        print("Class({}).__Initial_Members".format(self.__class__.__name__))
        
    
        
    """=============================================================
    Public :
    ============================================================="""
    # ----------------------------------------------------------
    # Get Functions 
    # ----------------------------------------------------------
    
    
    
    # ----------------------------------------------------------
    # Set Functions 
    # ---------------------------------------------------------- 
    
    
    
    
    
    # ----------------------------------------------------------
    # Main Functions 
    # ----------------------------------------------------------          
    
    
    # ----------------------------------------------------------
    # File IO Functions ------
    # ----------------------------------------------------------
    
    
    
    
    
#%% ***************************************************************************
    
class RvNeuralNetwork(object):    
    """=============================================================
    Static:
    ============================================================="""
    @staticmethod
    def LayersNeurons_To_RvNeuralLayers( layersNeuronsNum ):
        return [ RvNeuralLayer(inputNeusNum, lyrNeusNum) 
            for inputNeusNum, lyrNeusNum in \
                zip(layersNeuronsNum[:-1], layersNeuronsNum[1:]) ]
    
    @staticmethod
    def Create_Network(fnNetData):
        if (not os.path.isfile(fnNetData)): return None # 看網路參數檔案存在否
        
        f = open(fnNetData, "r")
        data = json.load(f)
        f.close()       
                
        """data = {
            'ClassName': self.__class__.__name__,
            'LayersNeurons' : self.Get_LayersNeurons(),
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
            'Email': cRasVectorEmail }"""
        
        lyrsNeurs = data['LayersNeurons']
        net = RvNeuralNetwork( lyrsNeurs )  # ( [784,50,10] )
        for i in range(1, len(lyrsNeurs)):
            net.NeuralLayers[i-1].Set_LayerData(data['NeuralLayers'][i-1])
            
        # 2018/02/12 新增
        key1, key2 = "EnumDropOutValue", "DropOutRatio"
        if (key1 in data) and (key2 in data):
            enumDropOut = ac.EnumDropOutMethod( data[key1] )            
            net.Set_DropOutMethod(enumDropOut, data[key2])
            
        return net
    
    
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
            else:
                print("Need InputNum and NeuronNum")        
        else:
            print("No Arguments")    
            
        if Debug: 
            self.Show_LayersInfo()  
            
    
    
    """=============================================================
    Private :
    ============================================================="""  
    def __Create_LayerObjects(self, lstLayersObjs):
        self.NeuralLayers = lstLayersObjs
        
    def __Create_LayerNeurons(self, lstLayersNeurons):
        self.__Create_LayerObjects( \
            RvNeuralNetwork.LayersNeurons_To_RvNeuralLayers(lstLayersNeurons))            
          
    def __Initial_Members(self):   
        self.__RandomState = np.random.RandomState(int(time.time()))
        self.NetEnableDropOut = False
        self.NetEnumDropOut = drpOut.eoSmallActivation
        self.NetDropOutRatio = gDropOutRatio
        self.Motoring_TrainningProcess = Debug        
        self.WorstAccuracyRatio = 1.0
        self.BestAccuracyRatio = 0.0
        self.AverageAccuracyRatio = 0.0           
        path = ".\\TmpLogs\\{}\\".format(self.__class__.__name__)
        if not os.path.isdir(path): os.mkdir(path)
        self.LogPath = path
    
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
#            if Debug:print("Z = \n{}\nActvation =\n{}\n".
#                  format(oneLyrNeurosZs,oneLyrNeusActvs))
        return lyrsNeusZs, lyrsNeusActvs
    
    
    """----------------------------------------------------------------------------
    # 計算單筆資料 對 所有神經元 Cost 偏微分的加總
    """
    def  __LayersNeurons_BackPropagation(self, oneInputX, labelY):         
        # Step1 : 正向計算所有神經元的 valueZ, activation        
        lyrsNeusZs, lyrsNeusActvs = self.__LayersNeurons_FeedForward(oneInputX)                
        #初始化一份和　self.biase結構、維度(多層，每層數量不同)一樣的 0值。
        lyrsNeus_dCost_dBiases = self.__Create_ArrayOf_LayersNeuronsBiases(0.0)
        lyrsNeus_dCost_dWeis = self.__Create_ArrayOf_LayersNeuronsWeights(0.0)
        # 指定前一筆輸入值          
        if len(lyrsNeusActvs)>1: priLyrNeusActvns = lyrsNeusActvs[-2]
        else: priLyrNeusActvns = oneInputX         
        
        # Step2 : 先計算最後一層輸出層的 cost 對 weight, bias 的偏微分，然後往前一直更新  
        oneLyrNeusErrs, lyrsNeus_dCost_dWeis[-1], lyrsNeus_dCost_dBiases[-1] = \
            self.NeuralLayers[-1].Caculate_dCost_OutputLayer( \
              lyrsNeusZs[-1], lyrsNeusActvs[-1], labelY, priLyrNeusActvns)            
            
        # Step3 : 非輸出層的 backPropagationagation，從倒數第二層開始 
        nLyr = len(self.NeuralLayers)        
        for lyrId in range(nLyr-2, -1, -1 ):  #range(sId,eId,step)#倒數第二層開始 
            # 每次求出的 oneLyrNeusErrs 在丟到函式內，當作下一層的 Error
            if lyrId==0: # input layer
                lastLyrNeusAs = oneInputX
            else:
                lastLyrNeusAs =lyrsNeusActvs[lyrId-1]            
                
            oneLyrNeusErrs, lyrsNeus_dCost_dWeis[lyrId], lyrsNeus_dCost_dBiases[lyrId] = \
                self.NeuralLayers[lyrId].Caculate_dCost_HiddenLayer( \
                    lastLyrNeusAs, lyrsNeusZs[lyrId], \
                    self.NeuralLayers[lyrId+1].NeuronsWeights, oneLyrNeusErrs)                
        return lyrsNeus_dCost_dWeis, lyrsNeus_dCost_dBiases
    
    
    
    """----------------------------------------------------------------------------
    # 計算所有資料 對 所有神經元 Cost 偏微分的加總
    """
    def __Caculate_Sum_LayersCostDerivations(self, inputDatas):
        # 為所有層的所有神經元配置空間，儲存 dCost 對所屬 weight 和 bias 的偏微分  
        sum_lyrsNeus_dCost_dBias = self.__Create_ArrayOf_LayersNeuronsBiases(0.0)
        sum_lyrsNeus_dCost_dWei = self.__Create_ArrayOf_LayersNeuronsWeights(0.0)    
        # 遍覽 inputDatas[] 內每筆資料，計算每筆input產生的偏微分，加總起來 ---------------
        for eachX, eachY in inputDatas: #每筆inputDatas[] = [x[784], y[10]]
            # 計算 cost 對單筆 inputDatas[]的　所有 biases 和 weights 的偏微分
            lyrsNeus_dCost_dWei, lyrsNeus_dCost_dBias = \
                self.__LayersNeurons_BackPropagation(eachX, eachY)
            # 逐一累加到所屬的 lyrsNeus_dCost_dBias[layer, neuron]　
            sum_lyrsNeus_dCost_dWei = [nw+dnw for nw, dnw in \
                zip(sum_lyrsNeus_dCost_dWei, lyrsNeus_dCost_dWei)]
            sum_lyrsNeus_dCost_dBias = [nb+dnb for nb, dnb in \
                zip(sum_lyrsNeus_dCost_dBias, lyrsNeus_dCost_dBias)]
                        
        return sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias
    
    
            
        
    # 計算 輸出層的 Activations ------------------------------------------------------
    def __Caculate_OutputActivations_FeedForward(self, oneInput):  #前面加 "__xxxfuncname()" 表示 private
        for lyr in self.NeuralLayers:
            oneInput = lyr.Get_NeuronsActivations(lyr.Get_NeuronsValuesZ(oneInput) )
        return oneInput

   
    # 預測值，傳入一筆測試資料 ------------------------------------------------------
    def __Get_CorrectNum(self, test_data):  #前面加 "__xxxfuncname()" 表示 private
        #測試資料 test_data[10000] = [ x[784], y[1] ], [ x[784], y[1] ],..... 10000筆   
        # 找到最大值所在的 index=數字結果 ---------------------
        test_results = \
            [(np.argmax(self.__Caculate_OutputActivations_FeedForward(x)), y)
             for (x, y) in test_data] # x,y 分別代表 test_data[0], test_data[1]
        return sum(int(x == y) for (x, y) in test_results)
        
    # 計算 y=a 辨識正確的data 數量-----------------------------------------
    def __Accuracy(self, inputDatas, convert=False):
        if convert:
            # 將 計算結果 a 和 label 放在 array[a,y], np.argmax(y):找出 y[]中最大值所在的 idx
          results = \
             [(np.argmax(self.__Caculate_OutputActivations_FeedForward(x)), np.argmax(y))
              for (x, y) in inputDatas]
        else:
          results = \
             [(np.argmax(self.__Caculate_OutputActivations_FeedForward(x)), y)
              for (x, y) in inputDatas]
        #隨機畫出測試的數字 ------    
        #rf.Plot_Digit(inputDatas[self.__RandomState.randint(0,len(inputDatas))])
        # 瀏覽所有結果，如果
        return sum(int(x == y) for (x, y) in results)
    
    
    def __CreateLabelsY(self, numY, labelId):
        e = np.zeros((numY, 1))
        e[labelId] = 1.0
        return e

    # 計算每個輸出的 cost ------------------------------------------------
    def __Total_Cost(self, inputDatas, lmbda, createLabelY=False):
        cost = 0.0
        n_Data = len(inputDatas)
        for x, y in inputDatas:
            finalLyrNeuronsActvns = self.__Caculate_OutputActivations_FeedForward(x)
            if createLabelY: y = self.__CreateLabelsY(10, y)
            # 以最後一層的 classCost計算 -------------------
            cost += self.NeuralLayers[-1].ClassCost.costValue(finalLyrNeuronsActvns, y)/n_Data
        # Regularization = lmbda/2n * Sum(wei^2)
        cost += 0.5*(lmbda/n_Data)*sum(
            np.linalg.norm(lyr.NeuronsWeights)**2 for lyr in self.NeuralLayers)
        return cost    
    
           
    def __Write_NetworkData(self, pF):          
        data = {
            'ClassName': self.__class__.__name__,
            'LayersNeurons' : self.Get_LayersNeurons(),
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
        json.dump(data, pF)
               
            
    
        
    """
    # 根據抽樣訓練集 mini_batch[]，利用梯度下降 來更新當前網路的所有 biases[], weights[]
    # 新的 nBiases = oBiases[] - learnRate * d(Cost)/d(bias)
    # 新的 nWeis = oWeis[] - learnRate * d(Cost)/d(wei)
    """
    def __Update_LayersNeurons_Weights_Biases(self, trainLoop, inputDatas, learnRate): 
        n_Data = len(inputDatas) 
        # 計算所有 layers 的所有 Neurons 的 Cost偏微分, dCost/dWei, dCost/dBia 
        lyrsNeus_dCost_dWei_sum, lyrsNeus_dCost_dBias_sum = \
            self.__Caculate_Sum_LayersCostDerivations(inputDatas)        
        
        # 計算新的 nWei = oWei - learnRate * nw,   nw = d(Cost)/d(wei) 
        lrDivN = learnRate/n_Data
        for lyr, oneLyrNWs in zip(self.NeuralLayers, lyrsNeus_dCost_dWei_sum):
            lyr.NeuronsWeights = np.asarray(
                #lyrWs - learnRate*(oneLyrNWs/n_Data)
                lyr.NeuronsWeights - lrDivN*oneLyrNWs)
        # 計算新的 nbias = obias - learnRate * nb,   nb = d(Cost)/d(bias) 
        for lyr, oneLyrNBiases in \
            zip(self.NeuralLayers, lyrsNeus_dCost_dBias_sum):
            lyr.NeuronsBiases = np.asarray( 
                #lyrBiases - learnRate*(oneLyrNBiases/n_Data)
                lyr.NeuronsBiases - lrDivN*oneLyrNBiases)
        
    

    """=============================================================
    Public :
    ============================================================="""
        
    #---------------------------------------------------------------------
    # Get Functions
    #---------------------------------------------------------------------
    def Get_LayersNum(self):
        return len(self.NeuralLayers)

    def Get_LayersNeurons(self):
        if self.Get_LayersNum()<=0: return []
        lyrsNeus = [ lyr.Get_NeuronsNum() for lyr in self.NeuralLayers ]
        lyrsNeus.insert(0, self.NeuralLayers[0].Get_InputNum())
        return lyrsNeus
    
    def Get_LayersNeuronsWeightsNum(self):        
        if self.Get_LayersNum()<=0: return []
        lyrsNeusWeis = [ lyr.Get_NeuronsNum()*lyr.Get_InputNum() 
            for lyr in self.NeuralLayers ]
        lyrsNeusWeis.insert(0, 0)
        return lyrsNeusWeis


    #---------------------------------------------------------------------
    # Set Functions
    #---------------------------------------------------------------------
    def Set_DropOutMethod(self, enumDropOut, ratioDropOut=0.5):   
        self.NetDropOutRatio = ratioDropOut
        self.NetEnumDropOut = enumDropOut    
    

    #---------------------------------------------------------------------
    # Display Functions
    #---------------------------------------------------------------------
    def Show_LayersInfo(self):    
        if [] == self.NeuralLayers:return
        print("\n{}:\n  LayersNeurons = {}".
            format(self.__class__.__name__,  self.Get_LayersNeurons() ) )      
        print("  LayersWeighsNum = {}".format(self.Get_LayersNeuronsWeightsNum()) )              

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
            filename="{}_NetData.txt".format(self.__class__.__name__)            
        pf = open(filename, "w")
        self.__Write_NetworkData(pf)
        pf.close()        
        
        
        

    #---------------------------------------------------------------------
    # Main Functions
    #--------------------------------------------------------------------- 
    def Train(self, training_data, loop, samplingStep, learnRate,
            test_data=None, lmbda=0.0, blInitialWeiBias=True):  
        
        print("\n************************************************")
        print("Train() with Stochastic Gradient Desent ********")
        print("DropOut={}, DropRatio={}, DropOutMethod={}".
              format(self.NetEnableDropOut, self.NetDropOutRatio, self.NetEnumDropOut))
        print("Random DropOut doesn't help in MNIST training")
        print("************************************************")
        self.Show_LayersInfo()
        
        cMinLoopToPlotDot = 30    
        
        #  如果是一樣的訓練集，則不初始化，可以累積訓練精度
        if blInitialWeiBias:
            for lyr in self.NeuralLayers:
                lyr. Initial_Neurons_Weights_Biases(lyr.Get_InputNum(), lyr.Get_NeuronsNum())
        
        
        #測試資料 test_data[10000] = [ x[784], y[1] ], [ x[784], y[1] ],..... 10000筆   
        worstAccuracyRatio, self.WorstAccuracyRatio = 1.0, 1.0
        bestAccuracyRatio, self.BestAccuracyRatio = 0.0, 1.0
        self.AverageAccuracyRatio = 0.0
        self.AverageCost = 0.0
        
        if (not test_data) or (self.Get_LayersNum()<1):        
            return worstAccuracyRatio,bestAccuracyRatio 
        
        self.Train_Loop = loop
        self.Train_LearnRate = learnRate
        self.Train_Lmbda = lmbda
        s1, s2 = "", ""
        
        t0 = time.time()
        
        # 初始化參數 ----------------------------------------------------
        n_train = len(training_data)
        n_test = len(test_data)
        test_cost, test_accuracy, training_cost, training_accuracy = [], [], [], []  
        accuRatio, cost, incAccuRatio, incCost = 0.0, 0.0, 0.0, 0.0
        
        if Debug:        
            fn = "{}{}_{}".format(self.LogPath, "DebugLog", \
                  self.NeuralLayers[-1].ClassActivation.__name__)  
            print("\n所有訓練資料({}), 取樣資料({}), 測試資料({}) ".
                  format( n_train, int(n_train/samplingStep)+1, n_test ))
            print("loop({}), stepNum({}), learnRate({}), lmbda({})\n".
                  format(loop, samplingStep, learnRate, lmbda))                
            print("#loop : Cost(), Accuracy()".format(loop) )        
                
        
        loops = np.arange(0,loop,1)
        for j in range(loop):                          
            self.__RandomState.shuffle(training_data) #隨機打亂測試樣本順序
            # 在 500000 筆訓練資料內，從0開始，累加10取樣，
            # 共 500000/10 = 1000筆mini_trainingData[]
            # [0~10], [10~20], [20~30]......[49990~49999]
            mini_trainingData = [
                training_data[k:k+samplingStep] 
                for k in range(0, n_train, samplingStep)]
            
            
            # 每次新 loop 時，才更新 DropOutMask -------------------
            # 只有在每次新的 loop 時，每間格 2 次才更新 DropOut 的神經元
            if self.NetEnableDropOut: # and ((j+1)%gDropOutPerEpcho==0) and (loop-1 != j): #不是最後一筆
                s2 = "DropOut({}, Ratio={})".format(self.NetEnumDropOut.name, self.NetDropOutRatio)
                # 只更新隱藏層，輸出層[:=1]不更新-------------
                for lyr in self.NeuralLayers[:-1]:
                    lyr.Set_DropOut(True, self.NetEnumDropOut, self.NetDropOutRatio)    
            else: s2 = ""
                        
            
            if Debug: print("#{}: {}".format(j, s2))
                            
            # 利用取樣集合 mini_trainingData[]，逐一以每個小測試集，　更新 weights 和 biases 
            for mini_batch in mini_trainingData:
                self.__Update_LayersNeurons_Weights_Biases(j, mini_batch, learnRate)
                            
            #if self.NetEnableDropOut:
            for lyr in self.NeuralLayers:
                lyr.Set_DropOut(False)
                
                
            # 根據更新後的 Weights, Biases 重新計算 training_data的 cost, accuracy
            if self.Motoring_TrainningProcess:
                train_cost = self.__Total_Cost(training_data, lmbda)
                training_cost.append(train_cost)
                train_accuracy = self.__Accuracy(training_data, convert=True)
                training_accuracy.append(train_accuracy)
                print("\tTrain: Cost({:.4f}), Accuracy({:.4f})".
                      format(train_cost, train_accuracy/n_train))       
                
            # 輸入 test_data 預測結果 ---------------------------------------            
            #accuracy, n_test = self.Evaluate_Accuracy(test_data)  
            cost = self.__Total_Cost(test_data, lmbda, createLabelY=True)
            test_cost.append(cost)
            accuracy = self.__Accuracy(test_data)
            test_accuracy.append(accuracy)
            
            incCost += cost
            accuRatio = accuracy/n_test
            incAccuRatio += accuRatio
            
            if accuRatio > bestAccuracyRatio: 
                bestAccuracyRatio = accuRatio
                s1 = "<-- Best"
            else:
                s1 = ""
            
            if accuRatio < worstAccuracyRatio:
                worstAccuracyRatio = accuRatio
                
            if Debug:
                print("\tTest : Cost({:.4f}), Accuracy({:.4f}) {}".
                    format(cost, accuRatio, s1))           
            
                     
                                        
                    
        if Debug:           
            if self.Motoring_TrainningProcess:          
                # 定義繪圖的字型資料 
                font = {'family': 'serif',
                        'color':  'red',
                        'weight': 'normal',
                        'size': 12,
                    }                                
                print("\nCost_Train: {:.4f} -> {:.4f}".
                      format(training_cost[0],training_cost[-1]) )
                print("Accu_Train: {:.4f} -> {:.4f}".
                      format(training_accuracy[0]/n_train,training_accuracy[-1]/n_train) )
                
                print("Cost_Test: {:.4f} -> {:.4f}".
                      format(test_cost[0],test_cost[-1]) )
                print("Accu_Test: {:.4f} -> {:.4f}".
                      format(test_accuracy[0]/n_test, test_accuracy[-1]/n_test) )
                
                rf.DrawFigures(plt, fn, font, learnRate, lmbda, loops, training_cost,
                          test_cost, n_train, n_test,training_accuracy,test_accuracy)
            if Debug_Plot:                        
                rf.Set_FigureText(plt, "Test Cost (lr={:.4f}, lmbda={:.4f})".
                        format(learnRate, lmbda), "Loop", "Coast")                
                if loop<cMinLoopToPlotDot:
                    plt.plot(loops, test_cost, "ro-")
                else:
                    plt.plot(loops, test_cost, "r-")
                plt.savefig("{}_TestCost.png".format(fn), format = "png")
                plt.show()                                                
                rf.Set_FigureText(plt, "Test Accuracy (lr={:.4f}, lmbda={:.4f})".
                        format(learnRate, lmbda), "Loop", "Accuracy")        
                if loop<cMinLoopToPlotDot:
                    plt.plot(loops, test_accuracy, "bo-")
                else:
                    plt.plot(loops, test_accuracy, "b-")
                plt.savefig("{}_TestAccuracyRatio.png".format(fn), format = "png")
                plt.show()                   
#          
#            if SaveVideo:     
#                fn1 =  "{}DNN_Cost_Accuracy.mp4".format(self.LogPath)        
#                Writer = animation.writers['ffmpeg']
#                writer = Writer(fps=15, metadata=dict(artist='Daniel Lu'), bitrate=1800) 
#                fig = plt.figure()
#                im_ani = animation.ArtistAnimation(fig, figs, interval=50, repeat_delay=3000,
#                    blit=True)
#                im_ani.save(fn1)                
                
            
        self.WorstAccuracyRatio = worstAccuracyRatio
        self.BestAccuracyRatio = bestAccuracyRatio
        self.AverageAccuracyRatio = incAccuRatio/loop
        self.AverageCost = incCost/loop
        
        self.Train_TimeSec = time.time()-t0
        
        return worstAccuracyRatio,bestAccuracyRatio     
    
    
    
    # 預測結果--------------------------------------------------------------
    def Evaluate_Accuracy(self, test_data):
        correctNum = self.__Get_CorrectNum(test_data)
        n_test = len(test_data)
        return correctNum, n_test
    

    # 預測某張圖的數字 ------------------------------------
    def Predict_Digit(self, oneImgDigit, plotDigit=False):
        caculation = self.__Caculate_OutputActivations_FeedForward(oneImgDigit[0])
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
        
        if type(caculation)==np.ndarray:
#            maxVal = 0.0
#            for i in range(0,len(caculation)):
#                if caculation[i]>maxVal: 
#                    maxVal = caculation[i]
#                    result=i
            result = np.argmax(caculation) # 取得陣列中最大數所在的index
        elif type(caculation)==np.int64:
            result = caculation
        else:
            result = -1     
            
        if plotDigit: rf.Plot_Digit(oneImgDigit)
        
        return label, result
        
    
        
#%%  ***************************************************************************

class RvConvolutionNeuralNetwork(RvNeuralNetwork):
    
    def __init__(self,  *args):
        print(self.__class__.__name__)
    
    
    
#%% ***************************************************************************
def Get_NoneDropOutMask(dim, ratioDrop):
    if ratioDrop < 0. or ratioDrop >= 1:#ratioDropout是概率值，必須在0~1之間  
        raise Exception('Dropout ratioDropout must be in interval [0, 1[.')  
    ratioRetain = 1. - ratioDrop  
    #我們通過binomial函數，生成與x一樣的維數向量。binomial函數就像拋硬幣一樣，我們可以把每個神經元當做拋硬幣一樣  
    #硬幣 正面的概率為p，n表示每個神經元試驗的次數  
    #因為我們每個神經元只需要拋一次就可以了所以n=1，size參數是我們有多少個硬幣。
    rng = np.random.RandomState(1234) #int(time.time())) #(1234)

    x = np.ones((dim,1))
    retain_norms=rng.binomial(n=1, p= ratioRetain, size=x.shape)#即將生成一個0、1分佈的向量，0表示這個神經元被屏蔽，不工作了，也就是dropout了  
    
    return retain_norms



def Get_DropOutMask1(dim, ratioDrop):
    if ratioDrop < 0. or ratioDrop >= 1:#ratioDropout是概率值，必須在0~1之間  
        raise Exception('Dropout ratioDropout must be in interval [0, 1[.')  
    ratioRetain = 1. - ratioDrop  
    #我們通過binomial函數，生成與x一樣的維數向量。binomial函數就像拋硬幣一樣，我們可以把每個神經元當做拋硬幣一樣  
    #硬幣 正面的概率為p，n表示每個神經元試驗的次數  
    #因為我們每個神經元只需要拋一次就可以了所以n=1，size參數是我們有多少個硬幣。
    rng = np.random.RandomState(1234) #int(time.time())) #(1234)

    x = np.ones((dim,1))
    retain_norms=rng.binomial(n=1, p= ratioRetain, size=x.shape)#即將生成一個0、1分佈的向量，0表示這個神經元被屏蔽，不工作了，也就是dropout了  
    
    ones = np.ones(x.shape, dtype=int)
    drop_norms = np.bitwise_xor(ones, retain_norms)  
    return ratioDrop, drop_norms, ratioRetain, retain_norms

def Get_Valuses_ByMask(x, norms, ratio):
    x *= norms# 0、1與x相乘，我們就可以屏蔽某些神經元，讓它們的值變為0  
    # print("x*norms = \n\t{}".format(x))      
    x /= ratio  
    # print("x*norms/remainRatio = \n\t{}".format(x))  
    return x  
    
    
def Get_NoneDropOutValues(x, ratioDrop):  
    if ratioDrop < 0. or ratioDrop >= 1:#ratioDropout是概率值，必須在0~1之間  
        raise Exception('Dropout ratioDropout must be in interval [0, 1[.')  
    ratioRetain = 1. - ratioDrop  
    #我們通過binomial函數，生成與x一樣的維數向量。binomial函數就像拋硬幣一樣，我們可以把每個神經元當做拋硬幣一樣  
    #硬幣 正面的概率為p，n表示每個神經元試驗的次數  
    #因為我們每個神經元只需要拋一次就可以了所以n=1，size參數是我們有多少個硬幣。  
    rng = np.random.RandomState(1234) #int(time.time())) #(1234)

    norms=rng.binomial(n=1, p= ratioRetain, size=x.shape)#即將生成一個0、1分佈的向量，0表示這個神經元被屏蔽，不工作了，也就是dropout了  
    # print("norms = \n\t{}".format(norms)) 
    
    x *= norms# 0、1與x相乘，我們就可以屏蔽某些神經元，讓它們的值變為0  
    # print("x*norms = \n\t{}".format(x))  
    
    x /= ratioRetain  
    # print("x*norms/remainRatio = \n\t{}".format(x))  
    return x  


#保留較大權種，特徵特別明顯的 ----------------------
def Get_BigValues(x, minValue=0.5):
    
    for a in x:
        a = (a>minValue)*a
    return x