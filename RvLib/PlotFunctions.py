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
sys.path.append('./RvLib')
import os
import time
from datetime import datetime


# Third-party libraries------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmn


# prvaite libraries---------------------------------------------


#%%%  Function Section
def Plot_Figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    
    
    nImg = len(figures)
    nrows = int(nImg / ncols) + (1 * (nImg%ncols>0) )

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    
    for ind,title in zip(range(nImg), figures):
        figures[title] = np.array(figures[title], dtype='uint8')
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        #axeslist.ravel()[ind].set_title(title)
        #img.set_cmap('hot') #讓圖形呈現紅色調
        axeslist.ravel()[ind].set_axis_off()
        
    for ind in range(nImg,nrows*ncols):
        axeslist.ravel()[ind].imshow([[]], cmap=plt.gray())
        #axeslist.ravel()[ind].set_title(title)
        #img.set_cmap('hot') #讓圖形呈現紅色調
        axeslist.ravel()[ind].set_axis_off()
        
    plt.tight_layout() # optional

""" 
Examples:
number_to_stop = 8
figures = {}
for i in range(number_to_stop):
    index = random.randint(0, n_train-1)
    figures[y_train[index]] = X_train[index]

plot_figures(figures, 2, 4)
"""


def Plot_Images(images, nrows = 1, ncols=1, sTitle="", saveFn="", figW_Inch=10):
    """Plot a dictionary of figures.

    Parameters
    ----------
    images : Array[0..n] of image[w][h]
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns*rows +1):
        img = np.random.randint(10, size=(h,w))
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()
    """
    
#    for i in range(len(images)):
#        if images[i].shape[0]==1:
#            w = int(np.sqrt(images[i].shape[1]))
#            images[i].reshape(w,w)
    
    images = np.array(images, dtype='uint8')
    
    nImg = len(images)
#    pxlH = len(images[0][0])
#    pxlW = len(images[0][1])
    nrows = int(nImg / ncols) + (1 * (nImg%ncols>0) )
    """matplotlib.pyplot.figure(num=None, figsize=None, dpi=None, facecolor=None, 
        edgecolor=None, frameon=True, FigureClass=<class 'matplotlib.figure.Figure'>, 
        clear=False, **kwargs)
    """
    
    idx = 0
    fig=plt.figure(figsize=(figW_Inch,figW_Inch))
#    if ""!=sTitle: fig.suptitle(sTitle)
    for i in range(1, ncols*nrows +1):
        fig.add_subplot(nrows,ncols, i)
        plt.axis('off')
        if (idx>=nImg):
          plt.imshow([[]])
        else:
          plt.imshow(images[idx], cmap=plt.gray())
        idx+=1
        
    #img.set_cmap('hot')
    if ""!=saveFn: plt.savefig(saveFn, bbox_inches='tight') #要放在 plt.show()之前才能正確存出圖形  
   
    plt.show()
    



#%%% Test Section






#%%% Main Section





