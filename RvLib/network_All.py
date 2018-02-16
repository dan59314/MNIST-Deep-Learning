# -*- coding: utf-8 -*-
# coding: utf-8 　　　←表示使用 utf-8 編碼，加上這一行遇到中文註解才能編譯成功

"""
Network_All.py

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


~~~~~~~~~~
 1. 隨機梯度下降法 SGD() 更新 Weights, Biases
 2. 三種 Cost Classes :
     Cost_Quadratic, Cost_CrossEntropy
 
 語法:    
  a 的平方 =  a**2  or math.pow(a,2)     
  np.linalg.norm( [a,b,c] ) = np.sqrt( a^2+b^2+c^2 )
  
  列舉型態 ------------------------------------------
  from enum import Enum
  
  class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3    

  print(Color.RED) -> Color.RED
  print(repr(Color.RED)) ->  <Color.RED: 1>
  type(Color.RED) ->  <enum 'Color'>
  isinstance(Color.GREEN, Color) ->  True
  print(Color.RED.name) ->  RED
  print(Color(2).name) -> GREEN
  for color in Color:print(color)
  
  zip()用法 ----------------------------------------
  a = [1,2,3]
  b = [4,5,6]
  c = [4,5,6,7,8]
  zipped = zip(a,b)
  print(zipped)
  >>[(1, 4), (2, 5), (3, 6)]
  print( zip(a,c) )
  >>[(1, 4), (2, 5), (3, 6)]
  print(zip(*zipped))
  >>[(1, 2, 3), (4, 5, 6)]
  
"""

#%%

#### Libraries
# Standard library
import os
import json
import random
from enum import Enum

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
#matplotlib.use("Agg")
#import matplotlib.animation as animation

import RvActivationCost as ac
import RvMiscFunctions as rf

#%%  Directive 定義 ----------------------------------------------------
Debug = True
Debug_Plot = True
SaveVideo = True


           

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

     
    
    
#%%
class Network_All(object):
    
    """
    =============================================================
    Constructor :
    ============================================================="""
    def __init__(self, lyrNeurons, 
                 enumInitWeisBias=Enum_WeiBiasInit.iwbLarge,
                 enumActivation=ac.EnumActivation.afSigmoid):
                   
        self.__Assign_FuncInitWeightsBiases(enumInitWeisBias)
        self.Assign_ClassActivation(enumActivation)
        
        self.__Func_Initial_Weights_Biases(lyrNeurons)
        self.__Initial_Members()             
       
        
    """
    =============================================================
    Static:
    ============================================================="""
    @staticmethod
    #### 從檔案讀取 Network參數 ----------------------------------------------
    def Create_Network(filename):
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        
        net = Network_All([0,0,0])
        
        net.__Initial_Network( data["LayerNeurons"], data["Weights"],
                          data["Biases"]) 
        
#        clsCost = getattr(sys.modules[__name__], data["Cost"])  
#        net.__Assign_ClassCost(clsCost)    
        
        enumInitWeisBias = data["EnumInitWeisBias"]  
        net.__Assign_FuncInitWeightsBiases(enumInitWeisBias)
        enumActivation = data["EnumActivation"]  
        net.Assign_ClassActivation(enumActivation) 
            
            
        return net
    
    
    """
    =============================================================
    Private :
    ============================================================="""
    # 初始化網路參數資料 ------------------------------------------------
    def __Initial_Weights_Biases_StdcostDerivation(self, lyrNeurons=[0,0,0]):  #前面加 "__xxxfuncname()" 表示 private
          # [784, 30, 10] 代表有三層，其中第一層有 784 輸入，第二層有30神經元，第三層有10個輸出
        self.LayerNum = len(lyrNeurons)
        self.LayersNeurons = lyrNeurons        
        print("\nLayerNeurons = {}\n__Initial_Weights_Biases_StdcostDerivation".
              format(lyrNeurons) )     
        
        """
        以 [L1, L2, L3] = [784, 50, 10] 為例
        Biases = [ [ # L2, 50個神經元, 50 x 1 array------------------------------------
                     [b1], 
                     ....   # L2, 50x1 個 bias
                     [b50] ],   
    
                   [ # L3, 10個神經元, 10 x 1 array ----------------------------------
                     [b1],     
                     ...    # L3, 10x1 個 bias
                     [b10] ] ] 
        """
        # np.random.randn(m,n) 配置 mxn維 array
        self.LayersNeuronsBiases = [ np.random.randn(nCurLyr, 1) # 使用lyrNeurons[]取道的值，配置(lyrNeurons[]*1])陣列 
                        # 從index=1 開始，遍覽 lyrNeurons[] 所有 值
                        for nCurLyr in lyrNeurons[1:]]  #[1:] 表示從 1開始，0不作
        """
        Weights = [ [ # L2, 50個神經元, 50 x 784 array------------------------------------
                     [w1 ... w784], 
                     ....   # L2, 50x784 個 weight
                     [w1 ... w784] ],   
    
                    [ # L3, 10個神經元, 10 x 50 array ----------------------------------
                     [w1 ... w50],     
                     ...    # L3, 10x50 個 weight
                     [w1 ... w50] ] ] 
        """
        self.LayersNeuronsWeights = [ np.random.randn( nCurLyr, nPreLyr)/np.sqrt(nPreLyr)
                        # x 從 lyrNeurons[0]取到lyrNeurons[lyrNum-2],[:-1]表示最後一筆不取
                        # y 從 lyrNeurons[1]取到lyrneurons[lyrNum-1]                         
                        # lyrNeurons[:-1] = [784, 50]
                        # lyrNeurons[1:]  = [50, 10]
                        # zip(lyrNeurons[:-1],  lyrNeurons[1:])] 
                        # => [x,y] = [784,50] , [50,10]
                        for nPreLyr, nCurLyr in zip(lyrNeurons[:-1], # 0 ~ lyr-2
                                        lyrNeurons[1:])] # 1 ~ lyr-1
                        
    def __Initial_Weights_Biases_Large(self, lyrNeurons=[0,0,0]):  #前面加 "__xxxfuncname()" 表示 private
          # [784, 30, 10] 代表有三層，其中第一層有 784 輸入，第二層有30神經元，第三層有10個輸出
        self.LayerNum = len(lyrNeurons)
        self.LayersNeurons = lyrNeurons        
        print("\nLayerNeurons = {}\n__Initial_Weights_Biases_Large".
              format(lyrNeurons) )     
        
        self.LayersNeuronsBiases = [np.random.randn(nCurLyr, 1) 
                        for nCurLyr in lyrNeurons[1:]]  #[1:] 表示從 1開始，0不作
        self.LayersNeuronsWeights = [np.random.randn(nCurLyr, nPreLyr)
                        for nPreLyr, nCurLyr in zip(lyrNeurons[:-1], 
                                        lyrNeurons[1:])]
            
    # 產生正態分布初始值 --------------------------------
    def __Initial_Weights_Biases_Normal(self, lyrNeurons=[0,0,0]):  #前面加 "__xxxfuncname()" 表示 private
          # [784, 30, 10] 代表有三層，其中第一層有 784 輸入，第二層有30神經元，第三層有10個輸出
        self.LayerNum = len(lyrNeurons)
        self.LayersNeurons = lyrNeurons        
        print("\nLayerNeurons = {}\n__Initial_Weights_Biases_Large".
              format(lyrNeurons) )     
        
        self.LayersNeuronsBiases = [np.random.normal(
                        loc=0.0, scale=1.0, size=(nCurLyr,1))     
                        # 從index=1 開始，遍覽 lyrNeurons[] 所有 值
                        for nCurLyr in lyrNeurons[1:]]  #[1:] 表示從 1開始，0不作
        self.LayersNeuronsWeights = [np.random.normal(
                        loc=0.0, scale=np.sqrt(1.0/nPreLyr), size=(nCurLyr,nPreLyr))   
                        # nPreLyr 從 lyrNeurons[0]取到lyrNeurons[lyrNum-2],[:-1]表示最後一筆不取
                        # nCurLyr 從 lyrNeurons[1]取到lyrneurons[lyrNum-1]
                        for nPreLyr, nCurLyr in zip(lyrNeurons[:-1], 
                                        lyrNeurons[1:])]
#        #耗費時間                  
#        plt.subplot(142)
#        for lyr in range(0, 1): #len(self.LayersNeuronsWeights)):   
#            plt.hist(self.LayersNeuronsWeights[lyr], 142, normed=True)                      
                        
    
    
    def __Initial_Members(self): 
        path = ".\\TmpLogs\\{}\\".format(self.__class__.__name__)
        if not os.path.isdir(path): os.mkdir(path)
        self.LogPath = path
        self.Motoring_TrainningProcess = False       
        self.WorstAccuracyRatio = 1.0
        self.BestAccuracyRatio = 0.0
        self.AverageAccuracyRatio = 0.0
        
    # 根據已存在的 網路參數初始化此網路 --------------------------------------    
    def __Initial_Network(self, lyrNeurons, weights, biases):
        self.__Func_Initial_Weights_Biases(lyrNeurons)
        self.__Initial_Members()    
        self.LayersNeuronsWeights = [np.array(w) for w in weights]
        self.LayersNeuronsBiases = [np.array(b) for b in biases]
        
    
    def __Assign_FuncInitWeightsBiases(self, enumInitWeiBias):
        self.EnumInitWeisBias = enumInitWeiBias
        # __Initial_Weights_Biases_StdcostDerivation()的效果比__Initial_Weights_Biases_Large()準確率大幅提升
        if enumInitWeiBias==Enum_WeiBiasInit.iwbStdError:
            self.__Func_Initial_Weights_Biases = self.__Initial_Weights_Biases_StdcostDerivation 
        elif enumInitWeiBias==Enum_WeiBiasInit.iwbLarge:
            self.__Func_Initial_Weights_Biases = self.__Initial_Weights_Biases_Large 
        elif enumInitWeiBias==Enum_WeiBiasInit.iwbNormal:
            self.__Func_Initial_Weights_Biases = self.__Initial_Weights_Biases_Normal 
        else:
            print("{} not found.".format(enumInitWeiBias))
            self.__Func_Initial_Weights_Biases = self.__Initial_Weights_Biases_StdcostDerivation            
        print("WeiBias_Initialization : \"{}\"".format(enumInitWeiBias))  
            
            
    def __LayersNeurons_FeedForward(self, oneInputX):
        
        # 輸入 x[] 是上一層所有神經元的 激活值 ----------------
        lyrNeusActv = oneInputX # input[784]
        lyrsNeusActv = [oneInputX] # list to store all the activations, layer by layer
        
        lyrsNeusZ = [] # 儲存所有層，所有神經元的加總值 z = Sum(wi*xi)+b
        
        for lyrWs,lyrBs in zip(self.LayersNeuronsWeights, self.LayersNeuronsBiases): # biases[lyr,nuro], weights[lyr,nuros,input]
            # b[lyr1], w[lyr1, 1~n]
            """
            第一筆 -----------------------------------
            w1= (50x784)  。  actn=(1x784)    +    bias = (50x1)   ->  z = (50x1)
             [                  [                   [                   [
               [w1..w784],        [x1..x784]          [b1],               [z1],
              ...50個...    。     ...1筆資料  +     ...50個...     =>     ...50個...
               [w1..w784]                             [b50]               [z50]
             ]                  ]                   ]                   ]
            """
            lyrZs = self.__NeuronValueZ(lyrWs, lyrNeusActv, lyrBs) #np.dot(w, activation)+b  # 神經元值 z = Sum(wi*xi)+b             
            
            # lyrsNeusZ = [  [..50個..], [..10個..] ]
            lyrsNeusZ.append(lyrZs)  # 加入新的 z 到 lyrsNeusZ[lyr,neuron]
            
            # activation : [..784個..] -> [..50個..] -> [..10個..]
            lyrNeusActv = self.ClassActivation.activation(lyrZs) # 神經元的激活值 sigmoid(), ReLU(), tanh()
            # activations = [  [..784個..], [..50個..], [..10個..] ]
            lyrsNeusActv.append(lyrNeusActv) # 加入新的激活值到 activations[lyr,neuron]
            
        return lyrsNeusZ, lyrsNeusActv
    
        
    
    def __NeuronValueZ(self, w, x, b):
        """
        w (mxn)     。  x (1xn)     +    b (mx1)   =   z (mx1)  
        [               [               [               [
         [w1..wn],        [x1..xn]        [b1],           [z1],
         .. m個..   。      ..1筆    +     ..m個    =      ..m個
         [w1..wn]                         [bm]            [zm]
        ]               ]               ]               ]
        
        """
        # outputZ = Sum(wi*xi)+b
        return np.dot(w, x)+b
        
    # 計算 輸出值 ------------------------------------------------------
    def __Feedforward(self, a):  #前面加 "__xxxfuncname()" 表示 private
        for b, w in zip(self.LayersNeuronsBiases, self.LayersNeuronsWeights):
            a = self.ClassActivation.activation( self.__NeuronValueZ(w,a,b) )
        return a

   

     # 輸入x[784], 輸出 y[10]，單筆訓練資料　
    def __BackPropagation(self, oneInputX, labelY):  #前面加 "__xxxfuncname()" 表示 private
        
        #初始化一份和　self.biase結構、維度(多層，每層數量不同)一樣的 0值。
        lyrsNeus_dCost_dBias = [np.zeros(b.shape) for b in self.LayersNeuronsBiases]
        lyrsNeus_dCost_dWei = [np.zeros(w.shape) for w in self.LayersNeuronsWeights]
        
        
        # 遍覽 每層的每個神經元，逐一求出 z, a，存在 lyrsNeusZ[lyr,neurons], activations[lyr,neurons]-----------
        """
        以 [L1, L2, L3] = [784, 50, 10] 為例
        Biases = [ [ # L2, 50個神經元, 50 x 1 array------------------------------------
                     [b1], 
                     ....   # L2, 50x1 個 bias
                     [b50] ],   
    
                   [ # L3, 10個神經元, 10 x 1 array ----------------------------------
                     [b1],     
                     ...    # L3, 10x1 個 bias
                     [b10] ] ] 
       
        Weights = [ [ # L2, 50個神經元, 50 x 784 array------------------------------------
                     [w1 ... w784], 
                     ....   # L2, 50x784 個 weight
                     [w1 ... w784] ],   
    
                    [ # L3, 10個神經元, 10 x 50 array ----------------------------------
                     [w1 ... w50],     
                     ...    # L3, 10x50 個 weight
                     [w1 ... w50] ] ] 
        """
        # 計算所有神經元的 activations 值 --------------------------
        lyrsNeusZ, lyrsNeusActv = self.__LayersNeurons_FeedForward( oneInputX)
        
       
        # 計算最後一層輸出層的 cost 對 weight, bias 的偏微分-------------------
        """
        輸出層的　backward pass，反向傳播 backPropagationogation --------------------------
        costDerivation = 2*(a-y) * d(z)
        lyrsNeusZ[-1]=(10x1)     lyrActs[-1]=(10x1)    labelY=(10x1)  =>  lyrNeusErr=(10x1)
          [                   [                   [                 [
            [z1],               [a1],               [y1],             [err1]
            ..10個              ..10個              ..10個             ..10個  
            [z10]               [a10]               [y10]             [err10]
          ]                   ]                   ]                 ]  
        """
        lyrNeusErr = (self.ClassCost).errorValue(lyrsNeusZ[-1], lyrsNeusActv[-1], labelY) 
        
        # cost(bias) = costDerivation 對 bias 微分 ->   d(costDerivation)/d(bias)=costDerivation ＃當前層的 costDerivation
        lyrsNeus_dCost_dBias[-1] = lyrNeusErr
        
        # const(weight) = costDerivation 對 weight微分 ->  d(costDerivation)/d(weight)=a(L-1)*costDerivation 
        # = 前一層 a*當前層costDerivation
        lyrsNeus_dCost_dWei[-1] = np.dot(lyrNeusErr, lyrsNeusActv[-2].transpose()) # [-2]表示倒數第二筆
        
        
        
        
        # 非輸出層的 backPropagationagation，從倒數第二層開始 -------------------------------------------
        for l in range(2, self.LayerNum): # 從倒數第二層開始, Lyr[lyrN-2], Lyr[LyrN-3]....
            lyrNeusZ = lyrsNeusZ[-l] #倒數第二層開始，L[-2], L[-3], L[-4]... L[1]
            lyrNeusdA = self.ClassActivation.derivation(lyrNeusZ)
            # 當前層(-l)的 costDerivation = 下一層(-l+1)的costDerivation[-l+1,neuron]*weight[-l+1,neuron]*lyrNeusdA
            # 當前層的誤差 = 下一層誤差反推回來的誤差權重 * 當前層的激活值
            lyrNeusErr = np.dot(self.LayersNeuronsWeights[-l+1].transpose(), lyrNeusErr) * lyrNeusdA
            # 當前層的 cost(bias) = 當前層的 costDerivation
            lyrsNeus_dCost_dBias[-l] = lyrNeusErr
            # 當前層的 cost(weight) = 前一層的a*當前層costDerivation
            lyrsNeus_dCost_dWei[-l] = np.dot(lyrNeusErr, lyrsNeusActv[-l-1].transpose())
        return (lyrsNeus_dCost_dBias, lyrsNeus_dCost_dWei)
    
    
    

    def __Caculate_Sum_LayersCostDerivations(self, inputDatas):
        """
        以 [L1, L2, L3] = [784, 50, 10] 為例
        lyrsNeus_dCost_dBias = 
                 [ [  # L2, 50個神經元, 50 x 1 array------------------------------------
                     [dc1], 
                     ....   # L2, 50x1 個 bias
                     [dc50] ],   
    
                   [ # L3, 10個神經元, 10 x 1 array ----------------------------------
                     [dc1],     
                     ...    # L3, 10x1 個 bias
                     [dc10] ] ] 
       
        lyrsNeus_dCost_dWei = 
                 [ [ # L2, 50個神經元, 50 x 784 array------------------------------------
                     [dw1 ... dw784], 
                     ....   # L2, 50x784 個 weight
                     [dw1 ... dw784] ],   
    
                    [ # L3, 10個神經元, 10 x 50 array ----------------------------------
                     [dw1 ... dw50],     
                     ...    # L3, 10x50 個 weight
                     [dw1 ... dw50] ] ] 
        """
        sum_lyrsNeus_dCost_dBias = [ np.zeros(b.shape) for b in self.LayersNeuronsBiases ]
        sum_lyrsNeus_dCost_dWei = [ np.zeros(w.shape) for w in self.LayersNeuronsWeights ]        
        
        # 遍覽 inputDatas[] 內每筆資料，計算每筆input產生的偏微分，加總起來 ---------------
        for eachX, eachY in inputDatas: #每筆inputDatas[] = [x[784], y[10]]
            # 計算 cost 對單筆 inputDatas[]的　所有 biases 和 weights 的偏微分
            # each_dCost_dBias =  d(Cost)/d(Bias)
            lyrsNeus_dCost_dBias, lyrsNeus_dCost_dWei = self.__BackPropagation(eachX, eachY)
            # 每筆inputDatas[] 所有的 biases[], weights[] 的偏微分結果，
            # 逐一累加到所屬的 lyrsNeus_dCost_dBias[layer, neuron]　
            sum_lyrsNeus_dCost_dBias = [nb+dnb for nb, dnb in zip(sum_lyrsNeus_dCost_dBias, lyrsNeus_dCost_dBias)]
            sum_lyrsNeus_dCost_dWei = [nw+dnw for nw, dnw in zip(sum_lyrsNeus_dCost_dWei, lyrsNeus_dCost_dWei)]
                        
        return sum_lyrsNeus_dCost_dWei, sum_lyrsNeus_dCost_dBias
    
    
        
    # 根據抽樣訓練集 mini_batch[]，利用梯度下降 來更新當前網路的所有 biases[], weights[]
    # 新的 nBiases = oBiases[] - learnRate * d(Cost)/d(bias)
    # 新的 nWeis = oWeis[] - learnRate * d(Cost)/d(wei)
    def __Update_LayersNeurons_Weights_Biases(self, inputDatas, learnRate): #前面加 "__xxxfuncname()" 表示 private
        # 配置一份和 self.LayersNeuronsBiases, self.LayersNeuronsWeights 一樣結構的陣列，
        # 用來儲存各層個神經元的 cost 對所屬 bias, weight 的偏微分
        lyrsNeus_dCost_dWei_sum, lyrsNeus_dCost_dBias_sum = \
            self.__Caculate_Sum_LayersCostDerivations(inputDatas)
        
        
        n_Data = len(inputDatas)
        # 計算新的 nWei = oWei - learnRate * nw,   nw = d(Cost)/d(wei) 
        """
        以 L2 來說 [784, 50, 10]
        lyrWs=(50x784)  -  learnRate*(lyrNWs/n_Data)         = lyrNWs(50x784)
         [                                                      [
           [w1..w784],     [lr*dw1/n_Data..lr*dw784/n_Data]       [w1..w784],
           ..50..個     -   ..50個..                               ..50個..
           [w1..w784]      [lr*dw1/n_Data..lr*dw784/n_Data]       [w1..w784]
         ]                                                      ]
        """
        lrDivN = learnRate/n_Data
        self.LayersNeuronsWeights = [
                #lyrWs - learnRate*(lyrNWs/n_Data)
                lyrWs - lrDivN*lyrNWs
                for lyrWs, lyrNWs in zip(self.LayersNeuronsWeights, lyrsNeus_dCost_dWei_sum)]
        # 計算新的 nbias = obias - learnRate * nb,   nb = d(Cost)/d(bias) 
        self.LayersNeuronsBiases = [ 
                #lyrBiases - learnRate*(lyrNBiases/n_Data)
                lyrBiases - lrDivN*lyrNBiases
                for lyrBiases, lyrNBiases in zip(self.LayersNeuronsBiases, lyrsNeus_dCost_dBias_sum)]
        
    
    # 預測值，傳入一筆測試資料 ------------------------------------------------------
    def __Get_CorrectNum(self, test_data):  #前面加 "__xxxfuncname()" 表示 private
        #測試資料 test_data[10000] = [ x[784], y[1] ], [ x[784], y[1] ],..... 10000筆   
        # 找到最大值所在的 index=數字結果 ---------------------
        test_results = [(np.argmax(self.__Feedforward(x)), y)
                        for (x, y) in test_data] # x,y 分別代表 test_data[0], test_data[1]
        return sum(int(x == y) for (x, y) in test_results)
        
    # 計算 y=a 辨識正確的data 數量-----------------------------------------
    def __Accuracy(self, data, convert=False):
        if convert:
            # 將 計算結果 a 和 label 放在 array[a,y]
            results = [(np.argmax(self.__Feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.__Feedforward(x)), y)
                        for (x, y) in data]
        #隨機畫出測試的數字 ------    
        #rf.Plot_Digit(data[np.random.randint(0,len(data))])
        # 瀏覽所有結果，如果
        return sum(int(x == y) for (x, y) in results)
    
    
    # 計算每個輸出的 cost ------------------------------------------------
    def __Total_Cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.__Feedforward(x)
            if convert: y = vectorized_result( 10, y)
            cost += self.ClassCost.costValue(a, y)/len(data)
        # Regularization = lmbda/2n * Sum(wei^2)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.LayersNeuronsWeights)
        return cost    
    
    
    
    """
    =============================================================
    Public :
    ============================================================="""     
        
    
    def Initial_Weights_Biases(self):  #前面加 "__xxxfuncname()" 表示 private        
        self.__Func_Initial_Weights_Biases(self.LayersNeurons)
                     
    
   
    def Assign_ClassActivation(self, enumActivation):
        self.EnumActivation = enumActivation
        
        self.ClassActivation, self.ClassCost = \
            ac.Get_ClassActivation(enumActivation)
            
        print("Activation Class : \"{}\"".format(self.ClassActivation.__name__))  
        print("Cost Class : \"{}\"".format(self.ClassCost.__name__)) 


    # Stochastic Gradient Desent 隨機梯度下降 -------------------------------
    def SGD(self, training_data, loop, samplingStep, learnRate,
            test_data=None, lmbda=0.0, blInitialWeiBias=True):  
        
        print("\n************************************************")
        print("SGD() Stochastic Gradient Desent ****************")
        print("Activation Class : \"{}\"".format(self.ClassActivation.__name__))  
        print("Cost Class : \"{}\"".format(self.ClassCost.__name__)) 
        print("WeiBias_Initialization : \"{}\"".format(self.EnumInitWeisBias))  
        print("************************************************")

        
        cMinLoopToPlotDot = 30
    
        
        #  如果是一樣的訓練集，則不初始化，可以累積訓練精度
        if blInitialWeiBias:
            self.__Func_Initial_Weights_Biases(self.LayersNeurons)
        
        
        #測試資料 test_data[10000] = [ x[784], y[1] ], [ x[784], y[1] ],..... 10000筆   
        worstAccuracyRatio = 1.0
        bestAccuracyRatio = 0.0
        self.WorstAccuracyRatio = worstAccuracyRatio
        self.BestAccuracyRatio = bestAccuracyRatio
        self.AverageAccuracyRatio = 0.0
        self.AverageCost = 0.0
        
        if not test_data: 
            return worstAccuracyRatio,bestAccuracyRatio        
        
        fn = "{}{}".format(self.LogPath, self.ClassCost.__name__)
        
        # 定義繪圖的字型資料 
        font = {'family': 'serif',
                'color':  'red',
                'weight': 'normal',
                'size': 12,
                }    
        
        n_train = len(training_data)
        n_test = len(test_data)
        
        if Debug:
            print("\n所有訓練資料({}), 取樣資料({}), 測試資料({}) ".
                  format( n_train, int(n_train/samplingStep)+1, n_test ) )
            print("loop({}), stepNum({}), learnRate({}), lmbda({})\n".
                  format(loop, samplingStep, learnRate, lmbda))                 
            print("#loop : Cost(), Accuracy()".format(loop) )            
        
        test_cost, test_accuracy = [], []
        training_cost, training_accuracy = [], []     
        
        accuRatio, cost, incAccuRatio, incCost = 0.0, 0.0, 0.0, 0.0
        
        loops = np.arange(0,loop,1)
        for j in range(loop):   
            print("#{}: ".format(j))
                
            random.shuffle(training_data) #隨機打亂測試樣本順序
            # 在 500000 筆訓練資料內，從0開始，累加10取樣，
            # 共 500000/10 = 1000筆mini_trainingData[]
            # [0~10], [10~20], [20~30]......[49990~49999]
            mini_trainingData = [
                training_data[k:k+samplingStep] 
                for k in range(0, n_train, samplingStep)]
                        
            # 利用取樣集合 mini_trainingData[]，逐一以每個小測試集，　更新 weights 和 biases 
            for mini_batch in mini_trainingData:
                self.__Update_LayersNeurons_Weights_Biases(mini_batch, learnRate)
                
            # 根據更新後的 Weights, Biases 重新計算 training_data的 cost, accuracy
            if self.Motoring_TrainningProcess:
                train_cost = self.__Total_Cost(training_data, lmbda)
                training_cost.append(train_cost)
                train_accuracy = self.__Accuracy(training_data, convert=True)
                training_accuracy.append(train_accuracy)
                print("   Train: Cost({:.4f}), Accu({:.4f})".
                      format(train_cost, train_accuracy/n_train))       
                
            # 輸入 test_data 預測結果 ---------------------------------------            
            #accuracy, n_test = self.Evaluate_Accuracy(test_data)  
            cost = self.__Total_Cost(test_data, lmbda, convert=True)
            test_cost.append(cost)
            accuracy = self.__Accuracy(test_data)
            test_accuracy.append(accuracy)
            
            incCost += cost
            accuRatio = accuracy/n_test
            incAccuRatio += accuRatio
            print("   Test : Cost({:.4f}), Accu({:.4f})".
                    format(cost, accuRatio))           
            
            if accuRatio > bestAccuracyRatio:bestAccuracyRatio = accuRatio
            if accuRatio < worstAccuracyRatio:worstAccuracyRatio = accuRatio
                      
                                            
                    
        if Debug:                
            if self.Motoring_TrainningProcess:            
                print("\nCost_Train: {:.4f} -> {:.4f}".format(training_cost[0],training_cost[-1]) )
                print("Accu_Train: {:.4f} -> {:.4f}".format(training_accuracy[0]/n_train,training_accuracy[-1]/n_train) )
                
                print("Cost_Test: {:.4f} -> {:.4f}".format(test_cost[0],test_cost[-1]) )
                print("Accu_Test: {:.4f} -> {:.4f}".format(test_accuracy[0]/n_test, test_accuracy[-1]/n_test) )
                
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
        
        return worstAccuracyRatio,bestAccuracyRatio     
    
    # 預測結果--------------------------------------------------------------
    def Evaluate_Accuracy(self, test_data):
        correctNum = self.__Get_CorrectNum(test_data)
        n_test = len(test_data)
        return correctNum, n_test
    
    
    # 將學習好的 Netework 參數存到檔案 ----------------------------------------------
    def Save_NetworkData(self, filename=""):  
        if ""==filename: 
            filename="{}_{}.txt".format(self.__class__.__name__, 
                      self.ClassCost.__name__)
        
        data = {
                "BestAccuracyRatio": self.BestAccuracyRatio,
                "ClassName": self.__class__.__name__,
                "LayerNeurons": self.LayersNeurons,
                "Weights": [w.tolist() for w in self.LayersNeuronsWeights],
                "Biases": [b.tolist() for b in self.LayersNeuronsBiases],
                "Cost": str(self.ClassCost.__name__),
                "EnumInitWeisBias": str(self.EnumInitWeisBias),
                "EnumActivation": str(self.EnumActivation)
                }
        
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        
    
    #### 從檔案讀取 Network參數 ----------------------------------------------
    def Update_Network(self, filename):
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        
        self.__Initial_Network( data["LayerNeurons"], data["Weights"],
                          data["Biases"]) 
#        
#        clsCost = getattr(sys.modules[__name__], data["Cost"])  
#        self.__Assign_ClassCost(clsCost)    
        enumInitWeisBias =  data["EnumInitWeisBias"]  
        self.__Assign_FuncInitWeightsBiases(enumInitWeisBias)
        enumActivation = data["EnumActivation"]  
        self.Assign_ClassActivation(enumActivation) 
            
            
    # 預測某張圖的數字 ------------------------------------
    def Predict_Digit(self, oneImgDigit, plotDigit=False):
        caculation = self.__Feedforward(oneImgDigit[0])
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
        
        
#%%
def softmax(z):
    e = np.exp(z - np.max(z))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2


def vectorized_result(m,j):
    e = np.zeros((m, 1))
    e[j] = 1.0
    return e

    