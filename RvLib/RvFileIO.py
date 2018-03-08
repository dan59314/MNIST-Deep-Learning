# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

/* -----------------------------------------------------------------------------
  Copyright: (C) Daniel Lu, RasVector Technology.

  Email : dan59314@gmail.com
  Web :     http://www.rasvector.url.tw/
  YouTube : http://www.youtube.com/dan59314/playlist
  GitHub : https://github.com/dan59314
  LinkedIn : https://www.linkedin.com/in/daniel-lu-238910a4/

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
#%%%
# Standard library----------------------------------------------
import sys
sys.path.append('../data')
sys.path.append('../RvLib')
import os
import time
from datetime import datetime


# Third-party libraries------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmn


# prvaite libraries---------------------------------------------
"""
import mnist_loader
import RvNeuralNetworks as rn
from RvNeuralNetworks import *
import RvAskInput as ri
import RvMiscFunctions as rf
import RvNeuNetworkMethods as nm
from RvNeuNetworkMethods import EnumDropOutMethod as drpOut
"""


#%%%  Function Section

def Get_FilesInFolder(sDir, fileExts=[]):
    sDir = os.path.abspath(sDir)
    fns = []
    fn0s = []
    if fileExts==[]:
        for file in os.listdir(sDir):
            fn0s.append(file)
            fns.append(os.path.join(sDir, file))
    else:
        for sExt in fileExts:
            for file in os.listdir(sDir):
                if file.endswith(sExt):
                    fn0s.append(file)
                    fns.append(os.path.join(sDir, file))
    return fns, fn0s




def Delete_Files(sDir, delFileExts=[], deleteSubDir=False):
    if(sDir == '//' or sDir == "\\"): return
    else:
        for root, dirs, files in os.walk(sDir, topdown=False):
            for name in files:
                if len(delFileExts)<=0:
                    os.remove(os.path.join(root, name))
                else:
                    for sExt in delFileExts:
                      if name.endswith(sExt): #(".jpg") 
                        os.remove(os.path.join(root, name))
            if deleteSubDir:
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
                
def ExtractFilePath(absFn):
    #return os.path.abspath(os.path.dirname(absFn))
    return os.path.split(absFn)[0]
        
    
def ExtractFileName(absFn):
    #return str(absFn).split('\\')[-1:][0]
    return os.path.split(absFn)[1]


def ExtractFileExt(absFn):
    #fn = ExtractFileName(absFn)
    #return str(fn).split('.')[1]    
    filename_w_ext = os.path.basename(absFn)
    filename, file_extension = os.path.splitext(filename_w_ext)
    return file_extension
    
def ForceDir(path):
    if not os.path.isdir(path):
        os.mkdir(path) 
        
def FileExists(absFn):
    return os.path.isfile(absFn)
    
def PathExists(path):
    return os.path.isdir(path)

def OpenFile(absFn):
    os.system(r'start ' + absFn)
    
    
    
#%%% Test Section






#%%% Main Section





