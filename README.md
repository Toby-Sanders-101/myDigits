# myDigits
Everything needed to create an AI bot that can read images

## Overview
In this repository, there are all the necessary files to create an AI model that can read digits 0-9. It can be modified to read any characters, words or symbols (however they would need numerical identifers). It uses python 3.7 as the programming language and uses a deep neural network to train itself.

## Files
*dataMaker.py* ~ This file uses numpy, cv2 and other modules to create a .csv file filled with testing and training data for the neural network

*myDigitsDNN.py* ~ This file uses numpy and other modules to train a deep neural network using the data in the .csv file. It then uses testing data to track its accuracy

*useMyDigitsReader.py* ~ This file uses numpy, cv2 and other modules to process real-time images and attempt to predict what digit they are

*myDigitsReaderHelp.py* ~ This file is only used as a module to be imported by the other files so that it is easier to make changes without having to change all the other files


*myDigits_(size).csv* ~ This stores all the data for training and testing

*myParameters_(size).txt* ~ This stores all the parameters used by the DNN model

## Modules
matplotlib==3.3.4

numpy==1.25.1

opencv-python==4.3.0.38

PyAutoGUI==0.9.54


These modules can all be installed using `pip install (module)` in the terminal

## Usage
1. In order to use this repository, you must first clone it:
    - To do this using the terminal, navigate to whichever directory you want it eg `cd Documents/Projects`, then run `git clone https://github.com/Toby-Sanders-101/myDigits.git`
    - Or download the ZIP file and extract it to your chosen directory

1. Then, assuming all modules and subsidiaries are installed correctly, you should confirm that your camera works sufficiently. You can run *dataMaker.py* with lines: 15, 30-37 and 44 commented out to do this. This will also output your camera dimensions.

1. Next, choose what 'quality drop' you would like to use and change line 10 in *myDigitsReaderHelp.py* accordingly. Similarly, line 8 in *dataMaker.py*, lines 9 and 15-18 in *myDigitsDNN.py* and lines 8-9 in *useMyDigitsReader.py* may need to be altered.

1. Next, you may want to change: lines 16-19 in *myDigitsReaderHelp.py*; lines 11-12 in *myDigitsDNN.py* along with line 14 in *useMyDigitsReader.py*; or line 205 in *myDigitsDNN.py*. These changes will allow you to modify: the quality of the images processed; the number of possible predictions/outputs from the model and the ratio of testing data:training data; or the learning rate and number of iterations carried out by the network.

1. Now that you've customised your files, you should run *dataMaker.py*. The first time you do this, it will create a file named *myDigits_(size).csv*. You will need to input the digit that you are taking photos of eg input 4, then you can proceed to take photos of the number 4 and it will collect the data. If you would like to take photos of letters or characters instead, you will need to alter some of the files to accommodate for this. This may include using a dictionary or array.

1. Repeat this until the .csv file is sufficiently full of data.

1. Next, run *myDigitsDNN.py*. This will create a file named *myParameters_(size).txt* to store the parameters. Chances are the network won't train very well the first time; you may need to change layer_dims, learning_rate, num_iterations or the amount/quality of training data in order to maximise the accuracy of the model.

1. Once you've created a model that is of high enough standard, you can use *useMyDigitsReader.py* to test it with real world data. If it continues to work well, you can integrate the model into a larger program or just show it off to your other developer friends! (otherwise it's back to the drawing board).

## Further reading
3blue1brown neural network explanantion: [https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

OpenCV Python Tutorial: [https://www.geeksforgeeks.org/opencv-python-tutorial/?ref=lbp](https://www.geeksforgeeks.org/opencv-python-tutorial/?ref=lbp)

NumPy Tutorial: [https://www.geeksforgeeks.org/numpy-tutorial/?ref=lbp](https://www.geeksforgeeks.org/numpy-tutorial/?ref=lbp)

Matplotlib Tutorial: [https://www.geeksforgeeks.org/matplotlib-tutorial/](https://www.geeksforgeeks.org/matplotlib-tutorial/)

PyAutoGUI Documentation: [https://buildmedia.readthedocs.org/media/pdf/pyautogui/latest/pyautogui.pdf](https://buildmedia.readthedocs.org/media/pdf/pyautogui/latest/pyautogui.pdf)
