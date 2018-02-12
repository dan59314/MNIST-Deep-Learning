
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


Created on Fri Jan 19 11:01:35 2018

@author: dan59314


"""

#%% 快捷鍵
"""
 Ctrl + 1 : 註解 / 撤銷註解
 Ctrl + 4/5 : 區塊註解
 Ctrl + L : 跳轉到行號
 F5 : Run
 F11 : 全螢幕
 Tab : 
 Shift + Tab : 縮排
"""




#%% 語法 -------------------------------------------------
"""
  //GetTickCount()
    import time
    start = time.time()
    do_long_code()
    print "it took", time.time() - start, "second
    
  //如果指定目錄不存在就建立目錄  要不然的話就直接開檔案    
    import os
    path = "C:\\alarm"
    if not os.path.isdir(path):
        os.mkdir(path)
    SaveDirectory = os.getcwd() #印出目前工作目錄
    #組合路徑，自動加上兩條斜線 "\\"
    
    SaveAs = os.path.join(SaveDirectory,'ScreenShot_' + time.strftime('%Y_%m_%d_%H_%M_%S') + '.jpg')
    

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


# prvaite libraries---------------------------------------------
import mnist_loader
import network_All as netAll
import RvAskInput as ri
import RvActivationCost as ac
import RvMiscFunctions as rf

#%%
DoTraining = False
    

#%%
""" 
測試 mnist_loader.load_data_wrapper()
training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()

print("type(training_data) = {}".format(type(training_data)))
lstTrain = list(training_data)
print("type(lstTrain) = {}".format(type(lstTrain)), "\n")

print("訓練資料({})".format(len(lstTrain)))
#print(list(training_data)[0][0].shape)
#print(list(training_data)[0][1].shape)
print("每筆輸入資料維度(輸入pixels, 結果y) = ", lstTrain[0][0].shape)
print("每筆輸出結果維度 = ", lstTrain[0][1].shape)

lstV = list(validation_data)
print("驗證資料({})".format(len(lstV)))

lstT = list(test_data)
print("測試資料({})".format(len(lstT)))
print('\n')
"""

#使用 mnist.pkl.gz(50000筆） 準確率 0.96 
#使用 mnist_expanded.pkl.gz(250000筆） 準確率提高到 0.97 
fn = ".\data\mnist.pkl.gz"  #".\data\mnist_expanded.pkl.gz"
lstTrain, lstV, lstT =  mnist_loader.load_data_wrapper(fn)
lstTrain = list(lstTrain)
lstV = list(lstV)
lstT = list(lstT)


#%%
""" 
測試 network(),  SGD()
"""
path = ".\\TmpLogs\\"
if not os.path.isdir(path):
    os.mkdir(path)

fnNetworkData = "{}NetworkData".format(path)   
fnNetworkData1 = ""

# Create NetWork -------------------------------------------
#lyrNeurons = [784, lyrN, 10] # 30 層->0.95, 60->0.96, 100->0.976, 400->0.9779
#lyrNeurons = [784, 50, 50, 50, 10] #-> 0.9735, 282秒
lyrNeurons = [784, 50]  
lyrNeurons = ri.Ask_Add_Array_Int("輸入增加新層神經元數", lyrNeurons, 30)
lyrNeurons.append(10)
#print("LayersNeurons = {}\n".format(lyrNeurons))


# Ask Method to Initial Weights, Baises ----------------------------
enumInitWeisBias = ri.Ask_Enum("選取 Wight/Bias 初始化函數.", 
     netAll.Enum_WeiBiasInit,  netAll.Enum_WeiBiasInit.iwbStdError )
#print("Method Init Weis/Biases = {}".format(methodInitWeisBias) )

# Ask Activation  ------------------_----------
enumActivation = ri.Ask_Enum("選取 Activation 類別.", 
     ac.ActivationFunction,  ac.ActivationFunction.afReLU )
    

# Create NetWork ------------------------------------------------
net = netAll.Network_All(lyrNeurons, enumInitWeisBias, enumActivation) 
net.LogPath = path     




#%% 神經網路訓練 SGD()*********************************************************************
# Ask DoTraining----------------------------------------------------
DoTraining = ri.Ask_YesNo("要執行SGD()訓練嗎?", "y")
# 是否要重新學習 ---------------------------------------------    
if DoTraining:     
    
    monitoring = ri.Ask_YesNo("是否監看訓練過程?", "y")
    netAll.Debug_Plot =  ri.Ask_YesNo("要顯示預估過程繪圖嗎?", "n")
    
    #輸入網路參數 --    
    loop = 10  # loop影響正確率不大，10和 30都在 9成以上
    stepNum = 10 # stepNum越大，正確率越低　10->0.9,  100->0.5
    learnRate = 0.1  # 調整 learnRate 和 lmbda 的參數，會互相影響結果
    lmbda = 5.0     # 加上 lmbda(Regularization) 可以解決 overfitting 問題    
    
    
    #是否要計算最適合的值 ----------------------------------------------
    DoEvaluate_ProperParams = ri.Ask_YesNo("要自動預估適合的網路參數嗎?", "n")
    if DoEvaluate_ProperParams:
        loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_lmbda(
                net, net.SGD, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
        loop,stepNum,learnRate,lmbda = rf.Evaluate_BestParam_learnRate(
                net, net.SGD, lstTrain[:1000], lstV[:500], loop,stepNum,learnRate,lmbda)
    else:      
        loop,stepNum,learnRate,lmbda = rf.Ask_Input_SGD(loop,stepNum,learnRate,lmbda)
    
    print( "網路訓練參數  : Loop({}), stepNum({}), learnRatio({}), lmbda({})\n".format(loop,stepNum,learnRate,lmbda)  )


    net.Motoring_TrainningProcess = monitoring
    
    start = time.time() 
    
    # 執行梯度下降訓練函數 ------------------------------
    #net.SGD(lstTrain, loop, stepNum, learnRate, lstT, lmbda) #test data
    net.SGD(lstTrain, loop, stepNum, learnRate, lstV, lmbda)  #evaluate data
    
    dT = time.time()-start
    
    # 存出網路參數檔案
    fnNetworkData1 = "{}_{:.2f}.txt".format(fnNetworkData, net.BestAccuracyRatio) 
    net.Save_NetworkData(fnNetworkData1)           
    
    s1 = "\n日期 : {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + \
         "神經網路結構 : {}\n".format(lyrNeurons)  + \
         "網路訓練參數 : Loop({}), stepNum({}), learnRatio({:.4f}), lmbda({:.4f})\n".format(loop,stepNum,learnRate,lmbda)  + \
         "WeiBias初始化 : \"{}\"\n".format(enumInitWeisBias) + \
         "Activation函數 : \"{}\"\n".format(enumActivation) + \
         "準確度 : 最差({}), 最好({})\n".format(net.WorstAccuracyRatio, net.BestAccuracyRatio)  + \
         "耗費時間(秒) : {:.3f} sec.\n".format( dT ) 
    print(s1)
    
#    f = open("{}Demo_Histry_All.txt".format(path),'a') 
#    f.write(s1)
#    f.close()      
   
    
#%% 以之前學習好的網路參數檔案來計算準確率***********************************************

    
if os.path.isfile(fnNetworkData1): 
    print(fnNetworkData1)
    
    LoadNetworkData =ri.Ask_YesNo("Do you want Load NetWork File and Evaluate?", "n")
     
    if LoadNetworkData:      
#        if os.path.isfile(fnNetworkData1): # 看網路參數檔案存在否
        # 使用網路參數檔案來預測 ---------------------------------------------
        # 初始化網路參數檔案
        net.Initial_Weights_Biases()
        correctNum,n_test  = net.Evaluate_Accuracy(lstT)
        print("初始化網路參數檔案 - Initial_NetworkData():\n結果(正確/總數): {}/{} ".format(correctNum,n_test))
        
        # 從檔案更新網路參數
        net.Update_Network(fnNetworkData1)
        correctNum,n_test  = net.Evaluate_Accuracy(lstT)
        print("讀取網路參數檔案 - Update_NetworkData():\n結果(正確/總數): {}/{} ".format(correctNum,n_test))
        
        # 從檔案建立新網路
        net1 = netAll.Network_All.Create_Network(fnNetworkData1)
        correctNum,n_test  = net1.Evaluate_Accuracy(lstT)
        print("讀取網路參數檔案以新建網路 - Create_NetworkData():\n結果 {:.3f}: {}/{} (正確/總數)".
              format(correctNum/n_test, correctNum,n_test))
             
        # 隨機取樣
        n_test = len(lstT)  
        np.random.shuffle(lstT) #隨機打亂測試樣本順序
        num=2000
        sId=np.random.randint(0,n_test-num-1)
        eId=sId + num
        correctNum,n_test  = net1.Evaluate_Accuracy(lstT[sId:eId])
        print("隨機取{}筆預測 - Evaluate_Accuracy():\n結果 {:.3f}: {}/{} (正確/總數)".
              format(num, correctNum/n_test, correctNum,n_test))
    
    
        # 隨機測試某筆數字 ----------------------------------------------
        label, result = net.Predict_Digit(lstT[np.random.randint(0,len(lstT))])    
        print("Label={}, Predict:{}".format(label,result))

    
# 手寫數字辨識競賽 世界成績　　http://yann.lecun.com/exdb/mnist/