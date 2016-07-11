#!/usr/bin/python

import numpy as np
from random import randint
from math import sqrt

NTrain = np.loadtxt('xyz.tra',dtype = float)
m,n = NTrain.shape
NTD = m

inp = 19
hid = 100
out = 7
lam = 1.e-02
epo = 1000
	
Wi = 0.001*(np.random.rand(hid,inp)*2.0-1.0)
Wo = 0.001*(np.random.rand(out,hid)*2.0-1.0)

for ep in range(epo):
	sumerr = 0
	miscla = 0
	for sa in range(NTD):
		xx = np.array([NTrain[sa,0:inp]]).T
		# print "xx",xx
		tt = np.array([NTrain[sa,inp:]]).T
		# print "tt",tt
		Yh = 1/(1 + np.exp(-Wi.dot(xx)))
		Yo = Wo.dot(Yh)
		# print "Yo",Yo
		er = tt - Yo
		Wo = Wo + lam * (er.dot((Yh.T)))
		Wi = Wi + lam * (((Wo.T).dot(er))*Yh*(1-Yh))*(xx.T)
		sumerr = sumerr + np.sum(er**2)
		ca = np.where(tt==1)[0][0]
		cp = np.where(Yo==max(Yo))[0][0]
		if ca!=cp:
			miscla += 1
	print [sumerr,miscla]

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

NFeature = np.loadtxt('xyz.tes',dtype = float)
m,n = NFeature.shape
NTD = m
conftes = np.zeros(shape=(out,out))
res_tes = np.zeros(shape=(NTD,2))
for sa in range(NTD):
	xx = np.array([NFeature[sa,0:inp]]).T
	ca = np.array([NFeature[sa,inp:]])
	# print "xx",xx
	# print "ca",ca
	Yh = 1/(1 + np.exp(-Wi.dot(xx)))
	Yo = Wo.dot(Yh)
	cp = np.where(Yo==max(Yo))[0][0]
	# print "cp",cp
	conftes[int(ca-1)][cp] += 1
	res_tes[sa,:] = [ca,cp]
print "conftes:"
print conftes
