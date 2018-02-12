# -*- coding: utf-8 -*-

#%%

"""
Created on Fri Feb  2 10:35:38 2018

@author: dan59314
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
"""

#%%

from enum import Enum

import numpy as np



#%% Activation ---------------------------------------------------------

class EnumDropOutMethod(Enum):
    eoNone = 0
    eoRandom = 1
    eoSmallActivation = 2
    
class ClassDropOut(object):
    """=============================================================
    Static:
    ============================================================="""
    @staticmethod
    #### 從檔案讀取 RvNeuralLayer 參數 ----------------------------------------------
    def Get_NonDropOutValues(enumDropOut, x, ratio=0.5):
        if (enumDropOut==EnumDropOutMethod.eoRandom):
            return ClassDropOut.__Get_NonDropOut_Random(x,ratio)
        elif (enumDropOut==EnumDropOutMethod.eoSmallActivation):
            return ClassDropOut.__Get_NonSmallActivation(x,ratio)
        else:
            return x
    
    @staticmethod
    def __Get_NonDropOut_Random(x,ratioDrop):
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
        
    @staticmethod
    def __Get_NonSmallActivation(x,minValue):
        for a in x:
            a = (a>minValue)*a
        return x
    
    
class ActivationFunction(Enum):
    afSigmoid = 1
    afReLU = 2
    afTanh = 3


class Activation_Sigmoid(object):     
    @staticmethod
    def activation(z): # = sigmoid(z)
        """The sigmoid function."""
        # signmod(z)=1/(1+e^-z)  
        return 1.0/(1.0+np.exp(-z))
    
    @staticmethod
    def derivation(z): # = sigmoid(z) * (1-sigmoid(z))
        """Derivative of the sigmoid function."""
        #return sigmoid(z)*(1-sigmoid(z))
        a = Activation_Sigmoid.activation(z)    
        return a * (1.0-a)
    

class Activation_ReLU(object): 
    @staticmethod
    def activation(z): # = ReLU(z)
        return z * (z > 0)
    
    @staticmethod
    def derivation(z): # = 1. * (ReLU(z) > 0)
        #return 1. * (ReLU(z) > 0)
        a = Activation_ReLU.activation(z)
        return 1.0 * (a>0)
    
class Activation_Tanh(object): 
    @staticmethod
    def activation(z): # = tanh(z)
        return np.tanh(z)
    
    @staticmethod
    def derivation(z): # = 1. - tanh(z) * tanh(z)
        #return 1. - tanh(z) * tanh(z)
        a = Activation_Tanh.activation(z)
        return 1.0 - a*a
    
    
#%% Cost  ----------------------------------------------------------------

class CostFunction(Enum):
    cfQuadratic = 1
    cfCrossEntropy = 2


class Cost_Quadratic(object):
    @staticmethod
    def costValue(a, y):
        # 傳回 cost = (a - y)^2 / 2
        # a=[a1,a2,a3....], y=[y1,y2,y3...]
        # cost = 0.5 * ( sqrt( (a1-y1)^2 + (a2-y2)^2 + ..... ) )^2
        # = 0.5 * np.linalg.norm(a-y)^2
        return 0.5*np.linalg.norm(a-y)**2  # 平方

    @staticmethod
    def errorValue(z, a, y):
        # 輸出層的　backward pass，反向傳播 backPropagationogation --------------------------
        # 先計算 errorValue = d(Cost)/d(a) = d( (a-y)^2)/d(a) = 2*(a-y) * a',   ----式 1 
        # sigmoid()微分 -> d(Sigmoid(z)) = d( 1/(1+e^-z) ) = sigmod(z)*(1-sigmoid(z)) --式2
        # 因為 sigmoid(z)=a  ->  所以 式2 ->    a'=a*(1-a) 代入式 1 ->
        # errorValue = 2*(a-y) * (a*(1-a))
        # return (a-y) * sigmoid_derivate(z)
        return (a-y) * Activation_Sigmoid.derivation(z)
    

class Cost_CrossEntropy(object): 
    @staticmethod
    def costValue(a, y):
        # 傳回 cost = -Sum( y*ln(a) + (1-y)*ln(1-a) ) / n
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))


    @staticmethod
    def errorValue(z, a, y):
        # 傳回 errorValue = (a-y)
        return (a-y)



#%% Function Define

def Get_ClassCost(enumCost):   
    if enumCost==CostFunction.cfQuadratic:
        return Cost_Quadratic
    elif enumCost==CostFunction.cfCrossEntropy:
        return Cost_CrossEntropy
    else: 
        return Cost_CrossEntropy       
    
    
def Get_ClassActivation(enumActivation):
        # __Initial_Weights_Biases_StderrorValue()的效果比__Initial_Weights_Biases_Large()準確率大幅提升
    if enumActivation==ActivationFunction.afSigmoid:
         # Sigmoid 會造成 訓練停滯，所以必須使用 CrossEntropy 加大
        return Activation_Sigmoid, Get_ClassCost(CostFunction.cfCrossEntropy)
    elif enumActivation==ActivationFunction.afReLU:
        # ReLU沒有限制Z 在 0~1 之間，所以不能使用 CrossEntropy 加大
        return Activation_ReLU, Get_ClassCost(CostFunction.cfQuadratic)
    elif enumActivation==ActivationFunction.afTanh:
        return Activation_Tanh, Get_ClassCost(CostFunction.cfQuadratic)
    else:
        print("{} not found.".format(enumActivation))
        # ReLU沒有限制Z 在 0~1 之間，所以不能使用 CrossEntropy 加大
        return Activation_ReLU, Get_ClassCost(CostFunction.cfQuadratic)


def softmax(z):
    e = np.exp(z - np.max(z))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2     