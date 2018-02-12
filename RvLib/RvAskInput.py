# -*- coding: utf-8 -*-

#%%
"""
Created on Tue Jan 30 21:28:22 2018

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
def Ask_YesNo(sQuestion, default="y"):
    sInput = input("{}(Now={}) (y/n): ".format(sQuestion, default))
    
    try: 
        if ""==sInput:
            return "y"==default.lower() or "yes"==default.lower()
        elif "y"==sInput.lower() or "yes"==sInput.lower(): 
            return True
        else:
            return False
    except ValueError:
        return False
    
    
def Ask_Input_Integer(sAsk, defaultInt=0):
    sInput = input("{} (Now={}): ".format(sAsk,defaultInt) )
    
    try:  
        if ""!=sInput: 
            return int(sInput)
        else:
            return defaultInt
    except ValueError:
        return defaultInt    
       
    
    
def Ask_Input_Float(sAsk, defaultFloat=0.0):
    sInput = input("{} (Now={}): ".format(sAsk,defaultFloat) )
    
    try: 
        if ""!=sInput: 
            return float(sInput)
        else:
            return defaultFloat
    except ValueError:
        return defaultFloat
    

def Ask_Enum(sAsk, enum, default):
    print("\n{}".format(sAsk))
    for eu in enum: 
        print("{} = {}".format(eu, eu.value ))
    sInput = input("Input Index (Now={}, \"{}\"): ".format( default.value,default.name ) )
    try: 
        if ""!=sInput: 
            return enum(int(sInput))
        else:
            return default
    except ValueError:
        return default
        

    
def Ask_Add_Array_Int(sAsk, lst, default):
    print("{}\n press\"Enter\" exit.".format(sAsk))
    while (True):
        #sInput = input("目前神經網路所有層{}, 新隱藏層神經元數: %s".format(lst)%default ) #or default
        sInput = input("Input Number (Current={}): ".format(lst) ) #or default
        
        try: 
            if ""==sInput:            
                break
            else:
                lst.append(int(sInput))
        except ValueError:
            break
            
    return lst

        
    
#%%
        