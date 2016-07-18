#!/usr/bin/python

# Multi Layer Perceptron - Classification with gradient
# decent and backpropogation for weights update using
# Sigmoidal activation function.

import numpy as np
from random import randint
from math import sqrt

# Load file containing training data set.
# Output must be in class coded label format.
NTrain = np.loadtxt('xyz.tra',dtype = float)
m,n = NTrain.shape
NTD = m

# Input parameters needed to be initialized.
# inp = no. of input neurons i.e. input features/dimensions.
# hid = no. of hidden neurons.
# out = no. of output neurons i.e. no. of output classes.
# lam = learning rate.
# epo = epoch cycles.
inp = 19
hid = 100
out = 7
lam = 1.e-02
epo = 1000

# input and output weight matrices initialized at random.
Wi = 0.001*(np.random.rand(hid,inp)*2.0-1.0)
Wo = 0.001*(np.random.rand(out,hid)*2.0-1.0)

# Train the network.
for ep in range(epo):
	sumerr = 0
	miscla = 0
	for sa in range(NTD):
		xx = np.array([NTrain[sa,0:inp]]).T			# Current sample
		tt = np.array([NTrain[sa,inp:]]).T			# Current traget
		Yh = 1/(1 + np.exp(-Wi.dot(xx)))			# Hidden output
		Yo = Wo.dot(Yh)						# Predicted output
		er = tt - Yo						# Error
		Wo = Wo + lam * (er.dot((Yh.T)))			# Update rule for output weights
		Wi = Wi + lam * (((Wo.T).dot(er))*Yh*(1-Yh))*(xx.T)	# Update rule for input weights
		sumerr = sumerr + np.sum(er**2)
		ca = np.where(tt==1)[0][0]				# Actual class
		cp = np.where(Yo==max(Yo))[0][0]			# Predicted class
		if ca!=cp:
			miscla += 1
	print [sumerr,miscla]

# Validate the network.
conftra = np.zeros(shape=(out,out))
res_tra = np.zeros(shape=(NTD,2))
for sa in range(NTD):
	xx = np.array([NTrain[sa,0:inp]]).T
	tt = np.array([NTrain[sa,inp:]]).T
	Yh = 1/(1 + np.exp(-Wi.dot(xx)))
	Yo = Wo.dot(Yh)
	ca = np.where(tt==1)[0][0]
	cp = np.where(Yo==max(Yo))[0][0]
	conftra[ca][cp] += 1
	res_tra[sa,:] = [ca,cp]
print "conftra:"
print conftra

# Test the network.
# Load file containing testing data set.
# Output must be in class lable format and not in class coded labels.
NFeature = np.loadtxt('xyz.tes',dtype = float)
m,n = NFeature.shape
NTD = m
conftes = np.zeros(shape=(out,out))
res_tes = np.zeros(shape=(NTD,2))
for sa in range(NTD):
	xx = np.array([NFeature[sa,0:inp]]).T
	ca = np.array([NFeature[sa,inp:]])
	Yh = 1/(1 + np.exp(-Wi.dot(xx)))
	Yo = Wo.dot(Yh)
	cp = np.where(Yo==max(Yo))[0][0]
	conftes[int(ca-1)][cp] += 1
	res_tes[sa,:] = [ca,cp]
print "conftes:"
print conftes
