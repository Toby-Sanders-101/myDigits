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
1. In order to use this repository, you must first clone it. You can use `gh repo clone Toby-Sanders-101/myDigits` in the terminal to do this

1. Then, assuming all modules and subsidiaries are installed correctly, you should confirm that your camera works sufficiently. You can run *dataMaker.py* with lines: 15, 30-37 and 44 commented out to do this. This will also output your camera dimensions.

1. Next, choose what 'quality drop' you would like to use and change lines 
