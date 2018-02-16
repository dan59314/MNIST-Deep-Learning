

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
import matplotlib.cm as cm


# prvaite libraries---------------------------------------------
import mnist_loader
import RvNeuralNetworks as rn
from RvNeuralNetworks import *
import RvAskInput as ri
import RvMiscFunctions as rf
import RvActivationCost as ac
import PlotFunctions as pltFn        
        
    
#%%
    
    
#使用 mnist_expanded.pkl.gz(250000筆） 準確率提高到 0.97 
fn = ".datamnist.pkl.gz"  #".datamnist_expanded.pkl.gz"
lstTrain, lstV, lstT =  mnist_loader.load_data_wrapper(fn)
lstTrain = list(lstTrain)
lstV = list(lstV)
lstT = list(lstT)
    
fnNetworkData1= ".\\{}_NetData_DontDelete.txt".format(rn.RvNeuralNetwork.__name__)
fnNetworkData2= ".\\{}_NetData_DropOut.txt".format(rn.RvNeuralNetwork.__name__)
fnNetworkData3= ".\\{}_NetData_CnvLyr.txt".format(rn.RvNeuralNetwork.__name__)
#
accur1,t1 = pltFn.Predict_Digits_FromNetworkFile(fnNetworkData1, lstT, False)
accur2,t2 = pltFn.Predict_Digits_FromNetworkFile(fnNetworkData2, lstT, False)
accur3,t3 = pltFn.Predict_Digits_FromNetworkFile(fnNetworkData3, lstT)
print("FullConnected Layer:\n  Accu:{}, Time:{:.3} sec\n".format(accur1,t1))
print("DropOut Layer:\n  Accu:{}, Time:{:.3} sec\n".format(accur2,t2))
print("Convolution Layer:\n  Accu:{}, Time:{:.3} sec\n".format(accur3,t3))



"""
if (os.path.isfile(fnNetworkData1)):
    print("\n讀取網路參數檔案以新建網路 - Create_NetworkData():\n")
    net = rn.RvNeuralNetwork.Create_Network(fnNetworkData1)    
    
    # 預測所有測試集------------------------------------------------
    correctNum,n_test  = net.Evaluate_Accuracy(lstT)    
    print("\n預測結果: 正確率({:.3f}),  {}/{}(正確/總數)".
          format(correctNum/n_test, correctNum,n_test))    
         
     隨機取樣 預測 -------------------------------------------------------------
    n_test = len(lstT)  
    np.random.shuffle(lstT) #隨機打亂測試樣本順序
    num=2000
    sId=np.random.randint(0,n_test-num-1)
    eId=sId + num
    correctNum,n_test  = net.Evaluate_Accuracy(lstT[sId:eId])
    print("\n隨機取{}筆預測 - Evaluate_Accuracy():\n結果 {:.3f}: {}/{} (正確/總數)".
          format(num, correctNum/n_test, correctNum,n_test))
#    
    # 預測多筆--------------------------------------
    Predict_Digits(net, lstT)        
    
    # 預測一筆 -------------------------------------
    aId = np.random.randint(0,len(lstT))
    label, result = net.Predict_Digit(lstT[aId], True)  
    print("({}): Label={}, Predict:{} -> {} ".
      format(aId,label,result, sResult[int(label==result)]))  
"""
    
    
    
    