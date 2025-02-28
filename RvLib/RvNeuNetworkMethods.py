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
    #__name__ = "DropOut"
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
    def __Get_NonSmallActivation(x,minValue): # MaxOut, 取較大神經元值的weight保留下來
        for a in x:
            a = (a>minValue)*a
        return x
    
#%% Activation Function
        
class EnumActivation(Enum):
    afSigmoid = 1
    afReLU = 2
    afTanh = 3
    afSeLU = 4


class Activation_Sigmoid(object):        
    #__name__ = "Sigmoid"   
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
    #__name__ = "ReLU"
    @staticmethod
    def activation(z): # = ReLU(z)
        return z * (z > 0)
        # return z if (z>0) else 0.01*z  # leaky ReLU
    
    @staticmethod
    def derivation(z): # = 1. * (ReLU(z) > 0)
        #return 1. * (ReLU(z) > 0)
        a = Activation_ReLU.activation(z)
        return 1.0 * (a>0)
        # return 1 if (a>0) else a  # leaky ReLU
    

class Activation_SeLU(object):     
    #__name__ = "SeLU"
    @staticmethod
    def activation(z): # = SeLU(z)
        alpha = 1.673263242354377
        lmbda = 1.050700987355480
        
        for vz in z:                
            if (vz>0): 
              vz = vz 
            else:
              vz = alpha*np.exp(vz)-alpha
        return lmbda*z  
         
    
    
    @staticmethod
    def derivation(z): # = 1. * (SeLU(z) > 0)
        #return 1. * (SeLU(z) > 0)
        a = Activation_SeLU.activation(z)
        return 1.0 * (a>0)
        # return 1 if (a>0) else a  # leaky SeLU
    
class Activation_Tanh(object): 
    #__name__ = "Tanh"
    @staticmethod
    def activation(z): # = tanh(z)
        return np.tanh(z)
    
    @staticmethod
    def derivation(z): # = 1. - tanh(z) * tanh(z)
        #return 1. - tanh(z) * tanh(z)
        a = Activation_Tanh.activation(z)
        return 1.0 - a*a
    
    
#%% Cost  ----------------------------------------------------------------

class EnumCost(Enum):
    cfQuadratic = 1
    cfCrossEntropy = 2


class Cost_Quadratic(object):    
    #__name__ = "Quadratic"
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
    #__name__ = "CrossEntropy"
    @staticmethod
    def costValue(a, y):
        # 傳回 cost = -Sum( y*ln(a) + (1-y)*ln(1-a) ) / n
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))


    @staticmethod
    def errorValue(z, a, y):
        # 傳回 errorValue = (a-y)
        return (a-y)


#%%  Filter Method
class EnumCnvFilterMethod(Enum):
    fmNone = 0
    fmAverageSum = 1
    
class ClassCnvoutionFilter(object):    
    #__name__ = "Sigmoid"
    """=============================================================
    Static:
    ============================================================="""
    @staticmethod
    #### 從檔案讀取 RvNeuralLayer 參數 ----------------------------------------------
    def Get_CnvFilterValue(oneInputPiece, oneFilter, enumCnvFilter=EnumCnvFilterMethod.fmAverageSum):
        if (enumCnvFilter==EnumCnvFilterMethod.fmAverageSum):
            return ClassCnvoutionFilter.__Get_CnvFilterValuese_fmAverageSum( \
                oneInputPiece, oneFilter)
        else:
            return ClassCnvoutionFilter.__Get_CnvFilterValuese_fmAverageSum( \
                oneInputPiece, oneFilter)
    
    @staticmethod
    def __Get_CnvFilterValues_fmAverageSum(inputPiece_1D, filter_1D):  
        assert( len(inputPiece_1D) == len(filter_1D) )
        num = len(inputPiece_1D)
        return np.dot(inputPiece_1D, filter_1D)/num





#%%  Filter Method
class EnumPoolingMethod(Enum):
    pmNone = 0
    pmMaxValue = 1
    pmAverageSum = 2
    
class ClassPooling(object):    
    #__name__ = "Sigmoid"
    """=============================================================
    Static:
    ============================================================="""
    @staticmethod
    #### 從檔案讀取 RvNeuralLayer 參數 ----------------------------------------------
    def Get_PoolValues(inputX, enumPool=EnumPoolingMethod.pmMaxValue):        
        if (enumPool==EnumPoolingMethod.pmMaxValue):
            return ClassPooling.__Get_PoolingValue_pmMaxValue(inputX)
        if (enumPool==EnumPoolingMethod.pmAverageSum):
            return ClassPooling.__Get_PoolingValue_pmAverageSum(inputX)
        else:
            return ClassPooling.__Get_PoolValuese_pmMaxValue(inputX)
        
    @staticmethod
    def __Get_PoolValues_pmMaxValue(inputX):         
        return max(inputX)
    
    @staticmethod
    def __Get_PoolValues_pmAverageSum(inputX):   
        aSum = 0.0
        for a in inputX: aSum+=a
        return aSum/len(inputX)
    
    
    

#%% Function Define

def Get_ClassCost(enumCost):   
    if enumCost==EnumCost.cfQuadratic:
        return Cost_Quadratic
    elif enumCost==EnumCost.cfCrossEntropy:
        return Cost_CrossEntropy
    else: 
        return Cost_CrossEntropy       
    
    
def Get_ClassActivation(enumActivation):
        # __Initial_Weights_Biases_StderrorValue()的效果比__Initial_Weights_Biases_Large()準確率大幅提升
    if enumActivation==EnumActivation.afSigmoid:
         # Sigmoid 會造成 訓練停滯，所以必須使用 CrossEntropy 加大
        return Activation_Sigmoid, Get_ClassCost(EnumCost.cfCrossEntropy)
    elif enumActivation==EnumActivation.afReLU:
        # ReLU沒有限制Z 在 0~1 之間，所以不能使用 CrossEntropy 加大
        return Activation_ReLU, Get_ClassCost(EnumCost.cfQuadratic)
    elif enumActivation==EnumActivation.afTanh:
        return Activation_Tanh, Get_ClassCost(EnumCost.cfQuadratic)
    elif enumActivation==EnumActivation.afSeLU:
        # ReLU沒有限制Z 在 0~1 之間，所以不能使用 CrossEntropy 加大
        return Activation_SeLU, Get_ClassCost(EnumCost.cfQuadratic)
    else:
        print("{} not found.".format(enumActivation))
        # ReLU沒有限制Z 在 0~1 之間，所以不能使用 CrossEntropy 加大
        return Activation_ReLU, Get_ClassCost(EnumCost.cfQuadratic)


def softmax(z):
    e = np.exp(z - np.max(z))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2     
    
    
    
    
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