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


## Example :  

  Train_NoConvLyr.py
  
  	Create and train a model for MNIST, then save the mode as a network file.
  
  Train_ConvLyr.py
  
    Same as above, but allow you to add a covolution layer    
  
  Load_And_Train.py
  
  	Load an saved network file(model) and keep training without restart all.
  
  Predict_Digits.py 
  
    Load traing data from MNIST data set, and randomlly predicit numbers insided.
  
  Predict_Digits_RealTime.py
  
    Capture image from camera, recognize digit(s) in realtime.    

[Recognizing One Digit Video](https://goo.gl/X8KAGz)

[![Recognizing One Digit](https://github.com/dan59314/MNIST-Deep-Learning/blob/master/images/Realtime_Predict.JPG)](https://goo.gl/X8KAGz?t=0s "One Digit Recognizing") 
		

[Recognizing Multiple Digits Video](https://youtu.be/FCE8azMDrMs)

[![Recognizing Multiple Digits](https://github.com/dan59314/MNIST-Deep-Learning/blob/master/images/Predict_MultiDigits.JPG)](https://youtu.be/FCE8azMDrMs?t=0s "One Digit Recognizing") 
        
------------------------------------------------------------------------------------
## What else you can do?

  1. Train your own hand-writing digits model.
  2. Train with input of other image set, like alphabat, patten, signs.... etc
  3. Tell me if you feel these codes useful.
  
-----------------------------------------------------------------------------------
      
## Hints :
  
  ### Methods in RvNeuralNetwork class:
  		Set_DropOutMethod()
  		Show_LayersInfo()
  		Train()
  		Evaluate_Accuracy()
  		Predict_Digit()
  		...
          
  ### Ways to create network:    
    
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

------------------------------------------------------------------------------------      
## Test result

Neural Network -> Accuracy

[784, 30, 10] -> 0.95

[784, 60, 10] -> 0.96

[784, 100, 10] -> 0.976

[784, 400, 10] -> 0.9779

3 Hidden Layers 

[784, 50, 50, 50, 10] -> 0.9735

Convolution Layer

[784, ConvLyr, 50, 10] -> 0.9801 ... tested 20 epochs

----------------------------------------------------------------------------------

![image](https://github.com/dan59314/MNIST-Deep-Learning/blob/master/images/Spyder01.jpg)
<img src="https://github.com/dan59314/MNIST-Deep-Learning/blob/master/images/Spyder01.jpg" width="640">

![image](https://github.com/dan59314/MNIST-Deep-Learning/blob/master/images/Spyder02.jpg)

![image](https://github.com/dan59314/MNIST-Deep-Learning/blob/master/images/train01.jpg)

![image](https://github.com/dan59314/MNIST-Deep-Learning/blob/master/images/train02.jpg)

![image](https://github.com/dan59314/MNIST-Deep-Learning/blob/master/images/train03.jpg)

![image](https://github.com/dan59314/MNIST-Deep-Learning/blob/master/images/Note01.jpg)

![image](https://github.com/dan59314/MNIST-Deep-Learning/blob/master/images/Note02.jpg)

![image](https://github.com/dan59314/MNIST-Deep-Learning/blob/master/images/Note03.jpg)

------------------------------------------------------------------------------------
## Misc. Projects of 3D, Multimedia, Arduino Iot, CAD/CAM, Free Tools

GitHub: https://github.com/dan59314

Email : dan59314@gmail.com

linkedin : https://www.linkedin.com/in/daniel-lu-238910a4/

Web : http://www.rasvector.url.tw/

YouTube : http://www.youtube.com/dan59314/playlist

Instructables : https://goo.gl/EwRGYA

Free Tools :	http://www.rasvector.url.tw/hot_91270.html
