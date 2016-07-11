#!/usr/bin/python

import numpy as np
from random import randint
from math import sqrt

NTrain = np.loadtxt('xyz.tra',dtype = float)
m,n = NTrain.shape
NTD = m

inp = 2
hid = 6
out = 1
lam = 1.e-02
epo = 9000
	
Wi = 0.001*(np.random.rand(hid,inp)*2.0-1.0)
Wo = 0.001*(np.random.rand(out,hid)*2.0-1.0)

for ep in range(epo):
	sumerr = 0
	DWi = np.zeros(shape=(hid,inp))
	DWo = np.zeros(shape=(out,hid))
	for sa in range(NTD):
		xx = np.array([NTrain[sa,0:inp]]).T
		tt = np.array([NTrain[sa,inp:]]).T
		Yh = 1/(1 + np.exp(-Wi.dot(xx)))
		Yo = Wo.dot(Yh)
		er = tt - Yo
		DWo = DWo + lam * (er.dot((Yh.T)))
		DWi = DWi + lam * (((Wo.T)*er)*Yh*(1-Yh))*(xx.T)
		sumerr = sumerr + np.sum(er**2)
	Wi = Wi + DWi
	Wo = Wo + DWo
	print sqrt(sumerr/NTD)

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
