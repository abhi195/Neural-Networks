#!/usr/bin/python

# Multi Layer Perceptron - Approximation with gradient
# decent and backpropogation for weights update using
# Sigmoidal activation function and least square
# error function.

import numpy as np
from random import randint
from math import sqrt

# Load file containing training data set.
NTrain = np.loadtxt('xyz.tra',dtype = float)
m,n = NTrain.shape
NTD = m

# Input parameters needed to be initialized.
# inp = no. of input neurons i.e. input features/dimensions.
# hid = no. of hidden neurons.
# out = no. of output neurons i.e. no. of output classes.
# lam = learning rate.
# epo = epoch cycles.
inp = 2
hid = 6
out = 1
lam = 1.e-02
epo = 9000


# input and output weight matrices initialized at random.
Wi = 0.001*(np.random.rand(hid,inp)*2.0-1.0)
Wo = 0.001*(np.random.rand(out,hid)*2.0-1.0)

# Train the network.
for ep in range(epo):
	sumerr = 0
	DWi = np.zeros(shape=(hid,inp))
	DWo = np.zeros(shape=(out,hid))
	for sa in range(NTD):
		xx = np.array([NTrain[sa,0:inp]]).T		# Current sample
		tt = np.array([NTrain[sa,inp:]]).T		# Current traget
		Yh = 1/(1 + np.exp(-Wi.dot(xx)))		# Hidden output
		Yo = Wo.dot(Yh)					# Predicted output
		er = tt - Yo					# Error
		DWo = DWo + lam * (er.dot((Yh.T)))		# Update rule for output weights
		DWi = DWi + lam * (((Wo.T)*er)*Yh*(1-Yh))*(xx.T)# Update rule for input weights
		sumerr = sumerr + np.sum(er**2)
	Wi = Wi + DWi
	Wo = Wo + DWo
	print sqrt(sumerr/NTD)

# Validate the network.
rmstra = np.zeros(shape=(out,1))
res_tra = np.zeros(shape=(NTD,2))
for sa in range(NTD):
	xx = np.array([NTrain[sa,0:inp]]).T
	tt = np.array([NTrain[sa,inp:]]).T
	Yh = 1/(1 + np.exp(-Wi.dot(xx)))
	Yo = Wo.dot(Yh)
	rmstra = rmstra + (tt-Yo)**2
	res_tra[sa,:] = [tt,Yo]
print "Validate:",sqrt(rmstra/NTD)

# Test the network.
# Load file containing testing data set.
NFeature = np.loadtxt('xyz.tes',dtype = float)
m,n = NTrain.shape
NTD = m
rmstes = np.zeros(shape=(out,1))
res_tes = np.zeros(shape=(NTD,2))
for sa in range(NTD):
	xx = np.array([NFeature[sa,0:inp]]).T
	ca = np.array([NTrain[sa,inp:]])
	Yh = 1/(1 + np.exp(-Wi.dot(xx)))
	Yo = Wo.dot(Yh)
	rmstes = rmstes + (ca-Yo)**2
	res_tes[sa,:] = [ca,Yo]
print "Test:",sqrt(rmstes/NTD)
