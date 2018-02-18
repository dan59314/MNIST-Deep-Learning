# MNIST-Deep-Learning
Deep Learning codes for MNIST with detailed explanation 

  ---------------------------------------------------------------------------------

  Copyright: (C) Daniel Lu, RasVector Technology.

  Email : dan59314@gmail.com
  
  linkedin : https://www.linkedin.com/in/daniel-lu-238910a4/
  
  Web :     http://www.rasvector.url.tw/
  
  YouTube : http://www.youtube.com/dan59314/playlist
  
  Instructables : https://goo.gl/EwRGYA
  
  

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
  
  linkedin : https://www.linkedin.com/in/daniel-lu-238910a4/
  
  Web :     http://www.rasvector.url.tw/
  
  YouTube : http://www.youtube.com/dan59314/playlist
  
  Instructables : https://goo.gl/EwRGYA
  
  

  使用或修改軟體，請註明引用出處資訊如上。未經過作者明示同意，禁止使用在商業用途。
  
  
  ---------------------------------------------------------------------------------


Example :  
  1. Predict_Digit  
      Predict_Digit_RvNeuralNetwork.py
      
  2. Training
      Test_RvNeuralNetwork.py
      
Hints :
  1. RvNeuralNetwork.Set_DropOutMethod(enumDropOut, ratioDropOut)
      enumDropOut :
          EnumDropOutMethod.eoRandom = 1,  normal dropout
          eoSmallActivation = 2  -->　kind of MaxOut, dropout the neurons of small activation that is < ratioDropOut
          
  2. Ways to create network:
    
    Create non-convolutionLayer network [ 780, 50, 10] :
     
    net = rn.RvNeuralNetwork([784,50,10])
      
      
    create convolutionLayer network [ 780, cnvLyr, 50, 10] :
     
    lyrObjs.append( RvConvolutionLayer(
        inputShape, # eg. [pxlW, pxlH, Channel]
        filterShape, # eg. [pxlW, pxlH, Channel, FilterNum], 
        filterStride) )         
        
       lyrObjs.append( rn.RvNeuralLayer([lyrObjs[-1].Get_NeuronNum), 50))
       
       lyrObjs.append( rn.RvNeuralLayer( [50, 10])
       
      net = rn.RvNeuralNetwork(lyrObjs)
      
      net.Train(....)
      
      
  
  
------------------------------------------------------------------------------------      ｉ
Misc. Projects of 3D, Multimedia, Arduino Iot, CAD/CAM, Free Tools

https://github.com/dan59314

http://www.rasvector.url.tw/hot_91270.html
