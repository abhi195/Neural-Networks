#!/usr/bin/python

import numpy as np
from random import randint
from math import sqrt,exp
import matplotlib.pyplot as plt

def coded_conversion(metrix,inp_size):
	coded_metrix = np.zeros(shape=(inp_size,len(np.unique(metrix))))
	coded_metrix += -1
	# print coded_metrix
	for i in range(inp_size):
		coded_metrix[i][metrix[i][0]-1]=1

	# print coded_metrix
	return coded_metrix

def get_dims(memberships, centroid_num):
	m = memberships.shape[0]
	row_dim = 0

	for i in range(m):
		if memberships[i][0]==centroid_num:
			row_dim += 1

	return row_dim

def computeCentroids(x, prev_centroids, memberships, k):
	m, n = x.shape
	centroids = np.zeros(shape=(k, n))

	for i in range(k):
		if not np.any(memberships==i):
			centroids[i,:] = prev_centroids[i,:]
		else:
			divisor = get_dims(memberships, i)
			prices = np.zeros(shape=(m,n))
			for j in range(m):
				if memberships[j][0]==i:
					prices[j,:]=x[j,:]
				else:
					prices[j,:]=0 #np.array([0,0])
			centroids[i,:] = (np.sum(prices,axis=0))/divisor
	return centroids


def findClosestCentroids(x, centroids):
	k = centroids.shape[0]
	m = x.shape[0]

	memberships = np.zeros(shape=(m,1))
	distances = np.zeros(shape=(m,k))

	for i in range(k):

		diffs = np.zeros(shape=(m,x.shape[1]))
		for j in range(m):
			diffs[j:] = x[j,:] - centroids[i,:]

		sqrdDiffs = diffs**2
		temp = np.array([np.sum(sqrdDiffs,axis=1)]).T
		for iter in range(m):
			distances[iter][i] = temp[iter][0]

		# distances[:,i] = np.array([np.sum(sqrdDiffs,axis=1)]).T
	# print distances
	for i in range(m):
		memberships[i][0] = np.where(distances==min(distances[i,:]))[1][0]

	return memberships

def KMeansInitCentroids(x,k):
	centroids = np.zeros(shape=(k,x.shape[1]))
	# print "centroids.shape",centroids.shape

	randidx = np.random.permutation(100)

	centroids = x[randidx[0:k],:]
	return centroids

def KMeans(x, initial_centroids, max_iters):
	k = initial_centroids.shape[0]

	centroids = initial_centroids
	prevCentroids = centroids
	# print type(prevCentroids)

	for i in range(max_iters):
		memberships = findClosestCentroids(x,centroids)
		centroids = computeCentroids(x, centroids, memberships, k)
		if (prevCentroids==centroids).all():
			break
		prevCentroids = centroids

	return centroids,memberships

if __name__ == '__main__':
	
	NTrain = np.loadtxt('xyz.tra',dtype = float)
	print "loadfile-shape:",NTrain.shape
	m,n = NTrain.shape
	NTD = m

	NTD=(NTD*3)/4

	inp = n-1
	print "Inp-features:",inp
	numRBFNeuronsPerClass = 5

	x_train = np.zeros(shape=(NTD,inp))
	y_train = np.zeros(shape=(NTD,1))
	# extract1 = [col for col in range(inp)]
	# extract2 = [col for col in range(inp,inp+1)]
	# print extract1
	# print extract2

	x_train = NTrain[0:NTD,0:inp]
	y_train = NTrain[0:NTD,inp:]
	# print "y_train",y_train

	numClass = len(np.unique(y_train))

	y_train_coded = coded_conversion(y_train,NTD)

	final_centroids = np.zeros(shape=(numRBFNeuronsPerClass*numClass,inp))
	# print "final_centroids.shape-before",final_centroids.shape
	counter = 0

	for c in range(numClass):
		extract = []
		for i in range(NTD):
			# print "y_train",y_train
			# print "c",c
			if int(y_train[i][0])==c+1:
				extract.append(i)

		# print "extract",extract

		Xc = x_train[extract,:]
		# print "Xc",Xc

		init_centroids = Xc[0:numRBFNeuronsPerClass,:]
		# print "init_centroids",init_centroids
		centers, memberships = KMeans(Xc, init_centroids, 100)
		# print "centers-->",centers

		for y in range(numRBFNeuronsPerClass):
			final_centroids[counter,:]=centers[y,:]
			counter += 1

	numRBFNeurons = final_centroids.shape[0]
	# print "final_centroids.shape",final_centroids.shape
	# print "final_centroids",final_centroids
	centers = final_centroids
	# print "centers.shape",centers.shape
	# for sa in range(NTD):
	# 	x_train[sa,:] = NTrain[sa,extract1]
	# 	y_train[sa,:] = NTrain[sa,extract2]

	# plt.plot(x_train, y_train, color="blue")
	# plt.show()

	# init_centroids = KMeansInitCentroids(x_train,numRBFNeurons)

	# centers,memberships = KMeans(x_train,init_centroids,100)
	# print "centers",centers
	# print memberships
	maxi = 0
	for i in range(numRBFNeurons-1):
		for j in range(i+1,numRBFNeurons):
			dist = centers[i,:] - centers[j,:]
			sqrdist = dist**2
			temp = np.sum(sqrdist)
			if temp>maxi:
				maxi = temp
	sigma = maxi/sqrt(numRBFNeurons)

	pseudo = np.zeros(shape=(NTD,numRBFNeurons+1))
	pseudo[:,numRBFNeurons] = 1
	for i in range(NTD):
		for j in range(numRBFNeurons):
			dist = x_train[i,:] - centers[j,:]
			sqrdist = dist**2
			divident = np.sum(sqrdist)
			gauss = divident / (2*(sigma**2))
			pseudo[i][j] = exp(-gauss)
	weight = np.linalg.pinv(pseudo).dot(y_train_coded)
	# print "weight",weight
	# print "weight.shape",weight.shape

	# Testing


	# NTest = np.loadtxt('her.tes',dtype = float)
	# m,n = NTest.shape
	# NTD = m

	# x_test = np.zeros(shape=(NTD,inp))
	# y_test = np.zeros(shape=(NTD,1))

	x_test = NTrain[NTD:,0:inp]
	y_test = NTrain[NTD:,inp:]
	# print y_test

	NTD = m - NTD

	# for sa in range(NTD):
	# 	x_test[sa,:] = NTest[sa,extract1]
	# 	y_test[sa,:] = NTest[sa,extract2]

	pseudo = np.zeros(shape=(NTD,numRBFNeurons+1))
	pseudo[:,numRBFNeurons] = 1
	# print "pseudo.shape",pseudo.shape	
	for i in range(NTD):
		for j in range(numRBFNeurons):
			dist = x_test[i,:] - centers[j,:]
			sqrdist = dist**2
			divident = np.sum(sqrdist)
			gauss = divident / (2*(sigma**2))
			pseudo[i][j] = exp(-gauss)

	sumerr = 0
	# print "pseudo",pseudo
	y_predicited = pseudo.dot(weight)
	# y_predicited += 0.5
	# print "y_predicited.shape",y_predicited.shape
	# print "y_predicited\n",y_predicited
	# print y_test.shape
	ca = []
	for z in range(NTD):
		ca.append(np.where(y_predicited[z,:]==max(y_predicited[z,:]))[0][0]+1)

	y_predicited = np.array([ca]).T
	correctly_classified = 0
	for i in range(NTD):
		# print "-->",int(y_predicited[i][0])
		# print "qwerty",int(y_test[i][0])
		if int(y_predicited[i][0])==int(y_test[i][0]):
			correctly_classified += 1
	print "Accuracy:",(correctly_classified/float(NTD))*100
	err = y_test - y_predicited
	sumerr = sumerr + np.sum(err**2)
	# print "rmse:",sqrt(sumerr/NTD)
	# plt.plot(y_test, y_predicited, color="blue")
	# plt.show()
