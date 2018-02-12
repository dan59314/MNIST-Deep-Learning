

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

#%%

sResult = ["錯誤", "正確"]
    
    
    
#%%
def Predict_Digits(net, lstT, plotDigit=True):
    # 隨機測試某筆數字 ----------------------------------------------    
    start = time.time() 
    
    sampleNum=10000 # 不含繪圖，辨識 10000張，費時 1.3 秒，平均每張 0.00013秒
    plotNum = 5
    plotMod = int(sampleNum/plotNum) + 1
    correctNum=0    
    failNum = 0
    for i in range(0, sampleNum):
        doPlot = (i%plotMod == 0)
        aId = np.random.randint(0,len(lstT))
        label, result = net.Predict_Digit(lstT[aId], False)    
        if label==result: correctNum+=1
        else: 
            failNum+=1
            doPlot = (failNum<plotNum) 
        if doPlot and plotDigit:
            rf.Plot_Digit(lstT[aId])
            print("({}): Label={}, Predict:{} -> {} ".
              format(i, label,result, sResult[int(label==result)]))   
    
    dt = time.time()-start
    
    accurRatio = correctNum/sampleNum
    print("\n預測結果: 正確率({:.3f}),  {}/{}(正確/總數)".
          format(accurRatio, correctNum, sampleNum))    
    print("耗費時間(秒) : {:.3f} sec.\n".format(dt))    
    return accurRatio

    
def Predict_Digicts_ByNetworkData(fn, lstT, plotDigit=True):
    if (os.path.isfile(fn)):
        net = rn.RvNeuralNetwork.Create_Network(fn)
        return Predict_Digits(net, lstT, plotDigit)        
        
        
    
#%%
    
    
#使用 mnist_expanded.pkl.gz(250000筆） 準確率提高到 0.97 
fn = ".datamnist.pkl.gz"  #".datamnist_expanded.pkl.gz"
lstTrain, lstV, lstT =  mnist_loader.load_data_wrapper(fn)
lstTrain = list(lstTrain)
lstV = list(lstV)
lstT = list(lstT)
    
fnNetworkData1= ".\\{}_NetData_DontDelete.txt".format(rn.RvNeuralNetwork.__name__)
fnNetworkData2= ".\\{}_NetData_DropOut.txt".format(rn.RvNeuralNetwork.__name__)

print("DropOut Layer :\n")
accur2 = Predict_Digicts_ByNetworkData(fnNetworkData2, lstT, False)
print("FullConnected Layer :\n")
accur1 = Predict_Digicts_ByNetworkData(fnNetworkData1, lstT)
print("Accuracy :\n DropOut Layer:  {}".format(accur2))
print(" FullConnected Layer:  {}".format(accur1))



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
    
    
    
    