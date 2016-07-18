#!/usr/bin/python

#For classification
#To print accuracy and Confusion matrix comparing predicted outputs and targets.

import numpy as np
from math import sqrt

#Load file containing predictions
y_pred = np.loadtxt('predicted.xyz',dtype = float)
#load file containing actual targets
y_actual = np.loadtxt('targets.xyz',dtype = float)
NTD = len(y_pred)
numClass = len(np.unique(y_actual))
conftes = np.zeros(shape=(numClass,numClass))

for i in range(NTD):
	pred = y_pred[i]
	actual = y_actual[i]
	conftes[actual-1][pred-1] += 1

#Printing confusion matrix
print conftes

#Calculating and printing accuracy
accuracy = np.trace(conftes)/NTD
print accuracy
