# -*- coding: utf-8 -*-


#%%
"""
Created on Thu Feb  8 14:50:04 2018

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
import time
from datetime import datetime, timedelta


import numpy as np
import matplotlib.pyplot as plt


import RvAskInput as ri



#%%


def Ask_Input_SGD(loop,stepNum,learnRate,lmbda):  
    print("Now : loop({}), stepNum({}), learnRate({}), lmbda({})".
        format(loop,stepNum,learnRate,lmbda) )    
    print("Please Input...")
    loop        = ri.Ask_Input_Integer("\t loop",loop)
    stepNum     = ri.Ask_Input_Integer("\t stepNum",stepNum)
    learnRate   = ri.Ask_Input_Float("\t learnRate",learnRate) 
    lmbda       = ri.Ask_Input_Float("\t Improve overfitting problem(lmbda>0.0)\n\t lmbda",lmbda)
    print()
    return loop,stepNum,learnRate,lmbda
        

def Evaluate_BestParam_lmbda(net, funcTrain, lstTrain, lstV, loop,stepNum,learnRate,lmbda):
    # 取 cost 最低的 ---------------------------------------------
    cTestLoop = 3
    cRoundDecimal = 3
    cMinLnbdaInterval = 0.1
        
    oMonitoring = net.Motoring_TrainningProcess
    net.Motoring_TrainningProcess = False
    
    lmbdaRange = [0.0, 100.0]

    curCost = 0.0
#    priCost = 0.0
    
    lmbda0 = lmbdaRange[0]
    funcTrain(lstTrain, cTestLoop, 10, learnRate, lstV, lmbda0, True)  #evaluate data
    Cost0 = net.AverageCost
    Cost0 = round(Cost0,cRoundDecimal)
    print("lmbda({}) -> Cost({})".format(lmbda0, Cost0) )
    
    lmbda1 = lmbdaRange[1]
    funcTrain(lstTrain, cTestLoop, 10, learnRate, lstV, lmbda1, False)  #evaluate data
    Cost1 = net.AverageCost
    Cost1 = round(Cost1,cRoundDecimal)
    print("lmbda({}) -> Cost({})".format(lmbda1, Cost1) )
    
    midLmbda = lmbda0
    
    while True: 
        if lmbdaRange[1]<lmbdaRange[0]:
            break
        elif abs(lmbdaRange[1]-lmbdaRange[0])<cMinLnbdaInterval:
            break        
        midLmbda = round((lmbdaRange[0]+lmbdaRange[1])/2.0, cRoundDecimal)       
        funcTrain(lstTrain, cTestLoop, 10, learnRate, lstV, midLmbda, False) 
        curCost  = net.AverageCost            
        curCost = round(curCost,cRoundDecimal)
        print("lmbda({}) -> Cost({})".format(midLmbda, curCost) )        
        
        if curCost > Cost0:#取左邊到中間
            lmbda = midLmbda  
            lmbdaRange[1] = midLmbda
            Cost1 = curCost
        elif curCost < Cost1: # 取中間到右邊
            lmbda = midLmbda  
            lmbdaRange[0] = midLmbda
            Cost0 = curCost            
        else:
            break   
         
        """
        midLmbda += 0.05    
        funcTrain(lstTrain, cTestLoop, 10, learnRate, lstV, midLmbda, False) 
        curCost  = net.AverageCost            
        curCost = round(curCost,cRoundDecimal)
        if curCost > priCost:
            lmbda = midLmbda  
            priCost = curCost
        else:
            break;
        """
        
    net.Motoring_TrainningProcess = oMonitoring    
    return loop,stepNum,learnRate,lmbda
        
    
def Evaluate_BestParam_learnRate(net, funcTrain, lstTrain, lstV, loop,stepNum,learnRate,lmbda):
    # 取 cost 最低的 ---------------------------------------------
    cTestLoop = 3
    cRoundDecimal = 3
    cMinLearnRateInterval = 0.1
    
    oMonitoring = net.Motoring_TrainningProcess
    net.Motoring_TrainningProcess = False
    
    learnRateRange = [0.01, 50.0]

#    priCost = 0.0
    curCost = 0.0
    
    learnRate0 = learnRateRange[0]
    funcTrain(lstTrain, cTestLoop, 10, learnRate0, lstV, lmbda, True)  #evaluate data
    Cost0 = net.AverageCost
    Cost0 = round(Cost0,cRoundDecimal)
    print("learnRate({}) -> Cost({})".format(learnRate0, Cost0) )
    
    learnRate1 = learnRateRange[1]
    funcTrain(lstTrain, cTestLoop, 10, learnRate1, lstV, lmbda, False)  #evaluate data
    Cost1 = net.AverageCost
    Cost1 = round(Cost1,cRoundDecimal)
    print("learnRate({}) -> Cost({})".format(learnRate1, Cost1) )
    
    
    while True: 
        if learnRateRange[1]<learnRateRange[0]:
            break
        elif abs(learnRateRange[1]-learnRateRange[0]) < cMinLearnRateInterval:
            break
        
        midlearnRate = round((learnRateRange[0]+learnRateRange[1])/2.0, cRoundDecimal)         
        funcTrain(lstTrain, cTestLoop, 10, midlearnRate, lstV, lmbda, False) 
        curCost  = net.AverageCost
        curCost = round(curCost,cRoundDecimal)
        print("learnRate({}) -> Cost({})".format(midlearnRate, curCost) )   
        
        if curCost > Cost0:#取左邊到中間
            learnRate = midlearnRate  
            learnRateRange[1] = midlearnRate
            Cost1 = curCost
        elif curCost < Cost1: # 取中間到右邊
            learnRate = midlearnRate  
            learnRateRange[0] = midlearnRate
            Cost0 = curCost            
        else:
            break               
                 
    
    net.Motoring_TrainningProcess = oMonitoring    
    return loop,stepNum,learnRate,lmbda




def Set_FigureText(plot, sTitle, sX, sY):
    fig = plot.figure()
    fig.suptitle(sTitle)
    plot.xlabel(sX)
    plot.ylabel(sY)
    
def Plot_Digit(digit, result=-1, label=-1):
    # digit[0] = pixels[784]
    # oneImgDigig[1] = label or labels[10]
    pixels = np.array(digit[0]*255, dtype='uint8')
    pixels = pixels.reshape((28, 28))
#    print(type(digit[1]) )
    
    if label==-1:
        if type(digit[1])==np.ndarray:
            label = np.argmax(digit[1]) # 取得陣列中最大數所在的index
        elif type(digit[1])==np.int64:
            label = digit[1]
        else:
            label = -1
        
    # Plot
    plt.title('result={},  Label={}'.format(result, label))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    
    

def DrawFigures(plt, fn, font, learnRate, lmbda, loops, training_cost,
                  test_cost, n_train, n_test,training_accuracy,test_accuracy,
                  blShow=True, blSave=True, frId=None, toId=None):      
    
    cMinLoopToPlotDot = 30

    # 將觀察的數據繪圖出來 ----------------------------------------------
    Set_FigureText(plt, "Cost (lr={:.4f}, lmbda={:.4f})".
                   format(learnRate, lmbda),
                   "loop", "cost")
    
    loop = len(loops)
    
    # 劃出圖表 --------------------
    if loop<cMinLoopToPlotDot:
        plt.plot(loops[frId:toId], training_cost[frId:toId],"ro-", test_cost[frId:toId],"bo-")
    else:
        plt.plot(loops[frId:toId], training_cost[frId:toId],"r-", test_cost[frId:toId],"b-") # "ro"畫紅點， "b--"畫藍色虛線
    # 顯示 Train, Test 數量
    font['color'] = 'green'
    plt.text(loops[0],training_cost[-1], "Train={}, Test={}".
              format(n_train,n_test), **font)  # loop//2 = round(loop/2)
    # 顯示線段最後的文字
    font['color'] = 'red'
    plt.text(loops[-1],training_cost[-1], r'$training$', **font)
    font['color'] = 'blue'
    plt.text(loops[-1],test_cost[-1], r'$test$', **font)
    if True==blSave: 
        plt.savefig("{}_Cost.png".format(fn), format = "png")
    if True==blShow: 
        plt.show() 
    
    Set_FigureText(plt, "Accuracy (lr={:.4f}, lmbda={:.4f})".
                   format(learnRate,lmbda),
                   "loop", "accuracy")
    accur_train = np.divide(training_accuracy,n_train)
    accur_test = np.divide(test_accuracy,n_test)
    # 劃出圖表 --------------------
    if loop<cMinLoopToPlotDot:
        plt.plot(loops[frId:toId], accur_train[frId:toId], "ro-", accur_test[frId:toId], "bo-")
    else:
        plt.plot(loops[frId:toId], accur_train[frId:toId], "r-", accur_test[frId:toId], "b-")
    
    # 顯示 Train, Test 數量
    font['color'] = 'green'
    plt.text(loops[0],accur_train[-1], "Train={}, Test={}".
             format(n_train,n_test), **font)  # loop//2 = round(loop/2)
    # 顯示線段最後的文字
    font['color'] = 'red'
    plt.text(loops[-1], accur_train[-1], r'$training$', **font)
    font['color'] = 'blue'
    plt.text(loops[-1], accur_test[-1], r'$test$', **font)
    if True==blSave: 
        plt.savefig("{}_Accuracy.png".format(fn), format = "png")
    if True==blShow: 
        plt.show()
        
        
        
        
def Save_NetworkDataFile(net, fnNetworkData, loop,stepNum,learnRate,lmbda, dT):
    # 劃出 Neurons Weights
    iLyr=0
    print("Layer({}) : ".format(iLyr))
    aId = np.random.randint(0, len(net.NeuralLayers[iLyr].NeuronsWeights) )
    net.NeuralLayers[iLyr].Plot_NeuronsWeights([aId,aId+1])
    
    # 存出網路參數檔案
    sConvLyr, sDropOut = "", ""
    sConvLyr = "_CnvLyr" if ([]!=net.Get_ConvolutionLayerID()) else ""        
    sDropOut = "_DropOut_{}".format(net.NetEnumDropOut.name) \
        if net.NetEnableDropOut else ""        
    fnSave = "{}{}{}_{:.2f}.txt".format(fnNetworkData, 
        sConvLyr, sDropOut, net.BestAccuracyRatio) 
    net.Save_NetworkData(fnSave)    
    

    lyrsWeis = net.Get_LayersNeuronsWeightsNum()
    weiMultiple = 0
    for wei in lyrsWeis:
        weiMultiple += wei
        
    s1 = "\nDateTime : {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + \
         "{}, {}\n".format(sConvLyr, sDropOut) + \
         "LayersName: {}\nLayersNeurons: {}\n".format( 
            net.Get_LayersName(), net.Get_LayersNeurons() ) +\
         "LayersWeightsNum: {}\n".format(lyrsWeis) + \
         "WeightsMultiples: {}\n".format(weiMultiple ) + \
         "Hyper Pameters: Loop({}), stepNum({}), lr({:.4f}), lmbda({:.4f})\n".format(
            loop,stepNum,learnRate,lmbda)  + \
         "Accuracy : Worst({}), Best({})\n".format(net.WorstAccuracyRatio, net.BestAccuracyRatio)  + \
         "Elapsed Time(Seconds): {} sec.\n".format( timedelta(seconds=int(dT)) ) 
    print(s1)
    
    if dT>600: # 超過十分鐘，把結果記下來
        fn = ".\RvNeuralNetwork.log"
        f = open(fn,'a') 
        f.write(s1)
        f.close() 
        print("log added to... \"{}\"".format(fn))
    
    return fnSave



#%% Initial Functions


