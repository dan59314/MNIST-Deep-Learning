 # -*- coding: utf-8 -*-


"""
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
import sys
sys.path.append('./RvLib')
import os
import mnist_loader
import network_All as netAll
import numpy as np
import time

import RvAskInput as ri
import RvMiscFunctions as rf


#%%
DoTraining = False

#%%
    

#%%
lstTrain, lstV, lstT =  mnist_loader.load_data_wrapper()
lstTrain = list(lstTrain)
lstV = list(lstV)
lstT = list(lstT)


#
#RandomPlotDigit = ask_YesNo("要隨機印出數字嗎?", "n")
## 是否要重新學習 ---------------------------------------------    
#if RandomPlotDigit:    
#    netAll.plot_Digit(lstTrain[np.random.randint(0,len(lstTrain))])
#    netAll.plot_Digit(lstV[np.random.randint(0,len(lstV))])
#    netAll.plot_Digit(lstT[np.random.randint(0,len(lstT))])


#%%

fnNetworkData = ".\\NetworkData_DontDelete.txt"   
if os.path.isfile(fnNetworkData): # 看網路參數檔案存在否
    
    
    # 從檔案建立新網路 --------------------------------------------------------
    net1 = netAll.Network_All.Create_Network(fnNetworkData)
    correctNum,n_test  = net1.Evaluate_Accuracy(lstT)
    print("讀取網路參數檔案以新建網路 - Create_NetworkData():\n結果 {:.3f}: {}/{} (正確/總數)".
          format(correctNum/n_test, correctNum,n_test))
    
         
    # 隨機取樣 預測 -------------------------------------------------------------
    n_test = len(lstT)  
    np.random.shuffle(lstT) #隨機打亂測試樣本順序
    num=2000
    sId=np.random.randint(0,n_test-num-1)
    eId=sId + num
    correctNum,n_test  = net1.Evaluate_Accuracy(lstT[sId:eId])
    print("隨機取{}筆預測 - Evaluate_Accuracy():\n結果 {:.3f}: {}/{} (正確/總數)".
          format(num, correctNum/n_test, correctNum,n_test))

    
    
    # 隨機測試某筆數字 ----------------------------------------------
    netAll.Debug_Plot = True
    
    start = time.time() 
    
    sResult = ["錯誤", "正確"]
    sampleNum=10000 # 不含繪圖，辨識 10000張，費時 1.3 秒，平均每張 0.00013秒
    plotNum = 5
    plotMod = int(sampleNum/plotNum) + 1
    correctNum=0    
    failNum = 0
    for i in range(0, sampleNum):
        doPlot = (i%plotMod == 0)
        aId = np.random.randint(0,len(lstT))
        label, result = net1.Predict_Digit(lstT[aId], False)    
        if label==result: correctNum+=1
        else: 
            failNum+=1
            doPlot = (failNum<plotNum) 
        if doPlot:
            rf.Plot_Digit(lstT[aId])
            print("({}): Label={}, Predict:{} -> {} ".
              format(i, label,result, sResult[(label==result)]))        
    print("Accuracy:{}, Correct/All = {}/{} \n耗費時間(秒) : {:.3f} sec.\n".
          format(correctNum/sampleNum, correctNum, sampleNum,
                  time.time()-start))    
    
# 手寫數字辨識競賽 世界成績　　http://yann.lecun.com/exdb/mnist/