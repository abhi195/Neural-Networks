#!/usr/bin/python

import numpy as np
from random import randint
from math import sqrt,exp
import matplotlib.pyplot as plt

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
	print "loadfile:shape",NTrain.shape
	m,n = NTrain.shape
	NTD = m

	NTD=(NTD*3)/4

	inp = 2
	numRBFNeurons = 10

	# x_train = np.zeros(shape=(NTD,inp))
	# y_train = np.zeros(shape=(NTD,1))
	# extract1 = [col for col in range(inp)]
	# extract2 = [col for col in range(inp,inp+1)]
	# print extract1
	# print extract2

	x_train = NTrain[0:NTD,0:inp]
	y_train = NTrain[0:NTD,inp:]
	# print y_train

	# for sa in range(NTD):
	# 	x_train[sa,:] = NTrain[sa,extract1]
	# 	y_train[sa,:] = NTrain[sa,extract2]

	# plt.plot(x_train, y_train, color="blue")
	# plt.show()

	init_centroids = KMeansInitCentroids(x_train,numRBFNeurons)

	centers,memberships = KMeans(x_train,init_centroids,100)
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
	weight = np.linalg.pinv(pseudo).dot(y_train)


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
	for i in range(NTD):
		for j in range(numRBFNeurons):
			dist = x_test[i,:] - centers[j,:]
			sqrdist = dist**2
			divident = np.sum(sqrdist)
			gauss = divident / (2*(sigma**2))
			pseudo[i][j] = exp(-gauss)

	sumerr = 0
	y_predicited = pseudo.dot(weight)
	# print y_predicited.shape
	# print y_test.shape
	err = y_test - y_predicited
	sumerr = sumerr + np.sum(err**2)
	print sqrt(sumerr/NTD)
	# plt.plot(y_test, y_predicited, color="blue")
	# plt.show()