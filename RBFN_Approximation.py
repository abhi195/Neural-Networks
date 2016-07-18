#!/usr/bin/python

# Radial Basis Function - Approximation
# Gaussian function as activation function.
# Centroids are found using K-Means Clustring.
# Normalized spread(sigma) is calculated form these centroids and is used as common spread for all centroids.
# Lloyd's(pseudo inverse) method is used to obtain optimal output weights.
# These obtained output weights are used to approximate the testing data.
# Cross Validation is used for testing our model.
# 75% of train dataset is used for training and remaining 25% is used for testing purpose.

import numpy as np
from random import randint
from math import sqrt,exp
import matplotlib.pyplot as plt

# To get no. of inputs lying within perticular centroid(center).
def get_dims(memberships, centroid_num):
	m = memberships.shape[0]
	row_dim = 0

	for i in range(m):
		if memberships[i][0]==centroid_num:
			row_dim += 1

	return row_dim

# To obtain the centroids.
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
					prices[j,:]=0
			centroids[i,:] = (np.sum(prices,axis=0))/divisor
	return centroids

# To obtain membership matrix which is a matrix mentioning
# which centroid is closest to the given input.
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

	for i in range(m):
		memberships[i][0] = np.where(distances==min(distances[i,:]))[1][0]

	return memberships

# At first initializing the centroids randomly.
def KMeansInitCentroids(x,k):
	centroids = np.zeros(shape=(k,x.shape[1]))

	randidx = np.random.permutation(100)

	centroids = x[randidx[0:k],:]
	return centroids

# K-Means Clustring
def KMeans(x, initial_centroids, max_iters):
	k = initial_centroids.shape[0]

	centroids = initial_centroids
	prevCentroids = centroids

	for i in range(max_iters):
		memberships = findClosestCentroids(x,centroids)
		centroids = computeCentroids(x, centroids, memberships, k)
		if (prevCentroids==centroids).all():
			break
		prevCentroids = centroids

	return centroids,memberships

if __name__ == '__main__':
	
	# Load file containing training data set.
	NTrain = np.loadtxt('xyz.tra',dtype = float)
	print "loadfile:shape",NTrain.shape
	m,n = NTrain.shape
	NTD = m

	NTD=(NTD*3)/4

	# inp = no. of input neurons i.e. input features/dimensions.
	inp = n-1
	numRBFNeurons = 10

	x_train = NTrain[0:NTD,0:inp]
	y_train = NTrain[0:NTD,inp:]

	init_centroids = KMeansInitCentroids(x_train,numRBFNeurons)

	centers,memberships = KMeans(x_train,init_centroids,100)

	# Obtaining the normalized spread.
	maxi = 0
	for i in range(numRBFNeurons-1):
		for j in range(i+1,numRBFNeurons):
			dist = centers[i,:] - centers[j,:]
			sqrdist = dist**2
			temp = np.sum(sqrdist)
			if temp>maxi:
				maxi = temp
	sigma = maxi/sqrt(numRBFNeurons)

	# Obtaining output weights using Lloyd's(pseudo inverse) method.
	pseudo = np.zeros(shape=(NTD,numRBFNeurons+1))
	pseudo[:,numRBFNeurons] = 1
	for i in range(NTD):
		for j in range(numRBFNeurons):
			dist = x_train[i,:] - centers[j,:]
			sqrdist = dist**2
			divident = np.sum(sqrdist)
			gauss = divident / (2*(sigma**2))
			pseudo[i][j] = exp(-gauss)
	weight = np.linalg.pinv(pseudo).dot(y_train)

	# Testing the network.
	x_test = NTrain[NTD:,0:inp]
	y_test = NTrain[NTD:,inp:]

	NTD = m - NTD

	pseudo = np.zeros(shape=(NTD,numRBFNeurons+1))
	pseudo[:,numRBFNeurons] = 1
	for i in range(NTD):
		for j in range(numRBFNeurons):
			dist = x_test[i,:] - centers[j,:]
			sqrdist = dist**2
			divident = np.sum(sqrdist)
			gauss = divident / (2*(sigma**2))
			pseudo[i][j] = exp(-gauss)

	sumerr = 0
	y_predicited = pseudo.dot(weight)
	err = y_test - y_predicited
	sumerr = sumerr + np.sum(err**2)
	
	print sqrt(sumerr/NTD)
