#!/usr/bin/python

# Radial Basis Function - Classification
# Gaussian function as activation function.
# Class wise centroids are found using K-Means Clustring.
# Normalized spread(sigma) is calculated form these centroids and is used as common spread for all centroids.
# Lloyd's(pseudo inverse) method is used to obtain optimal output weights.
# These obtained output weights are used to predict the class labels on testing data.
# Cross Validation is used for testing our model.
# 75% of train dataset is used for training and remaining 25% is used for testing purpose.

import numpy as np
from random import randint
from math import sqrt,exp
import matplotlib.pyplot as plt

# To convert class labels into coded class labels.
def coded_conversion(metrix,inp_size):
	coded_metrix = np.zeros(shape=(inp_size,len(np.unique(metrix))))
	coded_metrix += -1
	for i in range(inp_size):
		coded_metrix[i][metrix[i][0]-1]=1

	return coded_metrix

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
	# Outputs are in class label format.
	NTrain = np.loadtxt('xyz.tra',dtype = float)
	print "loadfile-shape:",NTrain.shape
	m,n = NTrain.shape
	NTD = m

	NTD=(NTD*3)/4

	# inp = no. of input neurons i.e. input features/dimensions.
	inp = n-1
	print "Inp-features:",inp
	numRBFNeuronsPerClass = 5

	x_train = np.zeros(shape=(NTD,inp))
	y_train = np.zeros(shape=(NTD,1))

	x_train = NTrain[0:NTD,0:inp]
	y_train = NTrain[0:NTD,inp:]

	numClass = len(np.unique(y_train))

	y_train_coded = coded_conversion(y_train,NTD)

	final_centroids = np.zeros(shape=(numRBFNeuronsPerClass*numClass,inp))
	counter = 0

	# Obtaining classwise centroids.
	for c in range(numClass):
		extract = []
		for i in range(NTD):
			if int(y_train[i][0])==c+1:
				extract.append(i)

		Xc = x_train[extract,:]

		init_centroids = Xc[0:numRBFNeuronsPerClass,:]
		centers, memberships = KMeans(Xc, init_centroids, 100)

		for y in range(numRBFNeuronsPerClass):
			final_centroids[counter,:]=centers[y,:]
			counter += 1

	numRBFNeurons = final_centroids.shape[0]
	centers = final_centroids

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
	weight = np.linalg.pinv(pseudo).dot(y_train_coded)

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

	ca = []
	for z in range(NTD):
		ca.append(np.where(y_predicited[z,:]==max(y_predicited[z,:]))[0][0]+1)

	y_predicited = np.array([ca]).T
	correctly_classified = 0
	for i in range(NTD):
		if int(y_predicited[i][0])==int(y_test[i][0]):
			correctly_classified += 1
	
	print "Accuracy:",(correctly_classified/float(NTD))*100
	err = y_test - y_predicited
	sumerr = sumerr + np.sum(err**2)
