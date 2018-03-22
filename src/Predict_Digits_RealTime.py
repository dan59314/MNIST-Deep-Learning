

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
sys.path.append('../data')
sys.path.append('../RvLib')
import os
import time
from datetime import datetime

import cv2
from skimage.filters import threshold_mean

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
import RvNeuNetworkMethods as nm
import PlotFunctions as pltFn   
import RvFileIO as rfi         
        

#%%
DrawContours = False
DrawRoiBox = True
SearchOne = False
DrawDigits = False


cSp = 3
cMinMulp = 0.05
cMaxMulp = 0.5
cPxls=2
cPadPxl=6

font = cv2.FONT_HERSHEY_SIMPLEX
fntSize = 4
fntThickness = 5

minSize = 50
maxSize = 400
#%%
def Get_MaxROI(contours, imgW,imgH):
    minDist = 1000000
    imgCX,imgCY= imgW/2, imgH/2
    
    idx =0 
    aId =-1
    for cnt in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        #roi=image[y:y+h,x:x+w]
        #cv2.imwrite(str(idx) + '.jpg', roi)
        #cv2.rectangle(image,(x,y),(x+w,y+h),(200,200,0),2)
        minWH=min(w,h)
        maxWH=max(w,h)
        cx,cy = (x+w/2), (y+h/2)
        dist = (cx-imgCX)**2 + (cy-imgCY)**2
        if (maxWH>minSize) and (maxWH<maxSize) and (dist<minDist):
            minDist = dist
#            minA = w*h
            aId = idx-1   
    return aId

def Padding_Array2D(input_2D, padSize, fillValue=0):
    if (padSize>0): # 每筆 input_2D[]左邊增加一筆
            # np.insert(array, 插入index, 插入值， 第幾維度)
        for ix in range(padSize): 
            input_2D=np.insert(input_2D, 0, fillValue, axis=1)
 
        #  np.append(array, 插入值，第幾個維度)
        for ix in range(padSize): 
            input_2D=np.insert(input_2D, len(input_2D[0]), fillValue, axis=1)

        for iy in range(padSize): 
            input_2D=np.insert(input_2D, 0, fillValue, axis=0)

        for iy in range(padSize): 
            input_2D=np.insert(input_2D, len(input_2D), fillValue, axis=0)
            
    return input_2D


def Predict_One(image, contours, contourId):
    ax,ay,aw,ah = 100, 200, 100, 100
    
    if contourId>=0:
        x,y,w,h = cv2.boundingRect(contours[contourId])
        ax,ay,aw,ah = x,y,w,h
        if DrawRoiBox: cv2.rectangle(image,(x,y),(x+w,y+h),(200,200,0),2)
        sp = 0 #int(max(w,h)*0.4)
        sy,ey,sx,ex = max(0,y-sp), min(imgH,y+h+sp),max(0,x-sp), min(imgW,x+w+sp)
        gray = image[sy:ey, sx:ex]
    else:
        gray = image

    # gray scale            
    gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)  
    #gray = threshold_mean(gray)
    
    #cv2.imshow('image',gray)
    gray = cv2.bitwise_not(gray)  #反相
    gray = cv2.resize(gray, (28-cPadPxl*2, 28-cPadPxl*2))  
    gray = Padding_Array2D(gray, cPadPxl, 0)
            
    # 加強對比 ---------------- 
#    iDark = np.average(gray[0:2][:])+30 #np.min(gray)+50 #       
#    for y in range(len(gray)):
#        for x in range(len(gray[0])):
#            gray[y][x] = 0 if gray[y][x]<iDark else min(255, gray[y][x]*1.5)
    ret, gray = cv2.threshold( gray, 127,255,cv2.THRESH_BINARY)  
    
    #OpenCV kernal structure 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(cPxls,cPxls)) 
    gray = cv2.dilate(gray,kernel)
    
    # retval, gray = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY);  
    #minus = 0.3 # np.linalg.norm(gray[0:5])
    x = gray.reshape((784,1))/255
    label = np.dtype('int64').type(6)
    inputX = np.asarray([x,label])
    #print(inputX.shape)
    inputX = tuple(inputX)
    
    #cv2.imshow('image',inputX)
    
    label, result, outputY = net.Predict_Digit(inputX, False)       

    if DrawDigits:        
        rf.Plot_Digit(inputX, result, label)
        print("Label={}, result:{}, actValue({:.5f}) ".format(label,result,outputY))  
    
    cv2.putText(image, str(result) ,(ax,ay+ah), font, 
        fntSize,(0,255,255), fntThickness,cv2.LINE_AA)
       

def Predict_All(image, contours, imgW,imgH):
    idx =0 
    for cnt in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        #roi=image[y:y+h,x:x+w]
        #cv2.imwrite(str(idx) + '.jpg', roi)
        #cv2.rectangle(image,(x,y),(x+w,y+h),(200,200,0),2)
        minWH=min(w,h)
        maxWH=max(w,h)
        if (maxWH>minSize) and (maxWH<maxSize):
            Predict_One(image, contours, idx-1)
                
#%%

fns, fn0s =  rfi.Get_FilesInFolder(".\\NetData\\", [".dnn",".cnn"])
if len(fns)>0:
  aId = ri.Ask_SelectItem("Select network file", fn0s, 0)    
  fn1= fns[aId]
  #fn1 = ".\RvNeuralNetwork_NetData_CnvLyr_ShareWeights.dnn"

  if (os.path.isfile(fn1)):
    net = rn.RvNeuralNetwork(fn1)
    if (None!=net):          
        camera = cv2.VideoCapture(0)
        while True:
            return_value,image = camera.read()  
            imgInfo = np.asarray(image).shape     
            if len(imgInfo)<2: break
            imgH=imgInfo[0]
            imgW=imgInfo[1]
            imgChannel=imgInfo[2]
            minSize = max(imgW,imgH) * cMinMulp
            maxSize = max(imgW,imgH) * cMaxMulp #maxImgWH-20
            
            # find out all contours ---------------------------
            binary =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(cPxls,cPxls)) 
#            binary = cv2.dilate(binary,kernel)
            ret, binary = cv2.threshold( binary, 127,255,cv2.THRESH_BINARY)   
            #binary = threshold_mean(binary)
            binary, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)  
            
            if DrawContours:
                cv2.drawContours(image,contours,-1,(0,0,255),3)  
            
            try: 
                # draw boundary box ---------------------------
                if SearchOne:
                    aId = Get_MaxROI(contours, imgW, imgH)
                    Predict_One(image, contours, aId)
                else:
                    Predict_All(image, contours, imgW, imgH)
                    
                cv2.imshow('image',image)
                    
                if cv2.waitKey(1) & 0xFF == 27:  #esc   ord('s'):
                    #cv2.imwrite('test.jpg',image)
                    break
            except ValueError:
                break
        
        camera.release()
        cv2.destroyAllWindows()
        
        
        

    