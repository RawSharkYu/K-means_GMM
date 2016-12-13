#!/usr/bin/python
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
#Your code here

def loadData(fileDj):
    data = []
    f = open(fileDj, 'r')
    for line in f:
        entries = line.split()
        data.append(map(float,entries[0:2]))
    return np.array(data)

## K-means functions 

def getInitialCentroids(data, k):
    initialCentroids = []
    for i in range(0,k):
        index = random.randint(0,len(data))
        initialCentroids.append(data[index])
    return initialCentroids

def getDistance(pt1,pt2):
    dist = np.sqrt(sum((pt1 - pt2) ** 2))
    return dist


def allocatePoints(data,centroids):
    centroid_num = 0
    clusters = [[] for i in range(0,len(centroids))]
    for p in data:
        index = 0
        min_dist = 99999.
        for c in centroids:
            dist = getDistance(p,c)
            if dist < min_dist:
                min_dist = dist
                centroids_num = index
            index += 1
        clusters[centroids_num].append(p)   

    return clusters

def updateCentroids(data,clusters):
    #calc new centroids  
    new_centroids = []
    index = 0
    sumX = 0
    sumY = 0
    for cluster in clusters:
        points_in_cluster = []
        for p in cluster:
            points_in_cluster.append(p)
        new_centroids.append(np.mean(points_in_cluster,axis=0))
    
    return new_centroids


def visualizeClusters(clusters):
    fig = plt.figure(figsize = (8,6))
    ax = fig.add_subplot()
    for p in clusters[0]:
        plt.scatter(p[0],p[1],c='yellow')
    for p in clusters[1]:
        plt.scatter(p[0],p[1],c='red')
    plt.title(u"K-means Result",fontsize=18)
    plt.grid(True)
    plt.show()
    return 0


def kmeans(data, k, maxIter=1000):
    data = loadData(data)
    new_centroids = getInitialCentroids(data,k)
    for i in range(0,maxIter):
        old_centroids = new_centroids
        clusters = allocatePoints(data,new_centroids)
        new_centroids = updateCentroids(data,clusters)
        for i in range(0,k):
            if(np.all(old_centroids[i]==new_centroids[i])):
                return clusters
    return clusters


def kneeFinding(data,kList):
    knee = []
    for k in kList:
        clusters = kmeans(data,k)
        distance = 0
        for c in clusters:
            center = np.mean(c,axis=0)
            for points in c:
                distance+=(getDistance(points,center)**2)
        knee.append(distance)
    plt.plot(kList,knee)
    plt.title(u"Knee Finding",fontsize=18)
    plt.show()

    return 0

def purity(fileDj, clusters):
    purities = []
    data1 = []
    data2 = []
    trueLabels = []

    f = open(fileDj, 'r')
    for line in f:
        entries = line.split()
        trueLabel = int(entries[2])
        if trueLabel == 1:
        	data1.append(map(float,entries[0:2]))
        else:
        	data2.append(map(float,entries[0:2]))
        trueLabels.append(trueLabel)

    for i in range(0,2):
    	count1=0
    	count2=0
    	for p in clusters[i]:
    		if p.tolist() in data1:
    			count1+=1
    		else:
    			count2+=1
    			
    	if max(count1,count2)==count1:
    		purities.append(float(count1)/len(clusters[i]))
    	else:
    		purities.append(float(count2)/len(clusters[i]))

    print purities
    return purities

# data = loadData("data_sets_clustering/humanData.txt")
# kneeFinding("data_sets_clustering/humanData.txt",[1,2,3,4,5,6])
#clusters = kmeans("data_sets_clustering/humanData.txt", 2, maxIter=1000)
# #visualizeClusters(clusters)
#purity("data_sets_clustering/humanData.txt",clusters)



## GMM functions 
def updateEStep(z,data,means,cov):
    for i in range(0,2):  # Dim
        for j in range(data.shape[0]):  # dado
            z[j, i] = veross(i, j,means,cov,data)
    z = (z.T / z.sum(axis=1)).T
    return z

def updateMStep(means,z,data):
    oldmeans = means
    oldmeans=oldmeans*100
    newmi = np.zeros_like(means)
    
    for i in range(0,2):
        for j in range(data.shape[0]):
            newmi[i] += z[j, i] * data[j] 
            
        means[i] = newmi[i] / z[:, i].sum()

    newmeans = means
    newmeans=newmeans*100

    diff = 0
    for i in range(data.shape[1]):
        
        diff+=(oldmeans[0][i]-newmeans[0][i])**2
    
    diff = diff*1000000000000000

    return diff


def veross(i, j,means,cov,data):
    x = data[j]
    mean = means[i]
    xm = x - mean
    a = np.exp(-.5 * np.dot(np.dot(xm, np.linalg.inv(cov)), xm))
    return 1 / (2 * np.pi * np.linalg.det(cov) ** 0.5) * a


def getInitialsGMM(X,k,covType):
    if covType == 'full':
        dataArray = np.transpose(np.array([pt[0:-1] for pt in X]))
        covMat = np.cov(dataArray)
    else:
        covMatList = []
        for i in range(len(X[0])-1):
            data = [pt[i] for pt in X]
            cov = np.asscalar(np.cov(data))
            covMatList.append(cov)
        covMat = np.diag(covMatList)

    initialClusters = {}

    X = X[:,:-1]
    dim = X.shape[1]
    numOfsamples=X.shape[0]

    means = np.ones((2, dim))

    m1 = random.randint(0,numOfsamples-1)
    m2 = random.randint(0,numOfsamples-1)

    means[0]=X[m1]
    means[1]=X[m2]

    initialClusters["cluster1"]=means[0]
    initialClusters["cluster2"]=means[1]
    initialClusters["cov"]=covMat

    return initialClusters


def visualizeClustersGMM(X,labels,clusters,covType):

    fig = plt.figure(figsize = (8,6))
    ax = fig.add_subplot(111)
    mark = ['og', 'ok', '^r']
    for i in range(np.shape(X)[0]):
        plt.plot(X[i][0],X[i][1],mark[labels[i]], markersize = 6)
   
    ax.set_xlabel(u'X',fontsize=18)
    ax.set_ylabel(u'Y',fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("GMM Result",fontsize=18)
    plt.grid(True)
    plt.show()
    return clusters

def gmmCluster(X, k, covType, maxIter=1000):
    #initial clusters
    clustersGMM = getInitialsGMM(X,k,covType)
    labels = []
    #Your code here
    X = X[:,:-1]
    numOfsamples = X.shape[0]
    dim = X.shape[1]
    
    means = np.ones((2, dim))
    means[0]=clustersGMM["cluster1"]
    means[1]=clustersGMM["cluster2"]
    cov = clustersGMM["cov"]

    z = np.zeros((numOfsamples, 2))
    oldmeans = means
    
    count = 0
    for it in range(maxIter):
        count+=1
        z=updateEStep(z,X,means,cov)
        diff= updateMStep(means,z,X)
        if diff < 0.00000001:
            break
   
    clustersGMM["cluster1"]=means[0]
    clustersGMM["cluster2"]=means[1]
    for i in range(X.shape[0]):
        if z[i][0] > z[i][1]:
            labels.append(1)
        else:
            labels.append(2)

    # print labels
    #visualizeClustersGMM(X,labels,clustersGMM,covType)
    return labels,clustersGMM


def purityGMM(X, clusters, labels):
    Y = X[:,-1:]
    X = X[:,:-1]
    truelabels = Y.T[0]

    numOfsamples = X.shape[0]
    dim = X.shape[1]
    
    purities = []
    countC1=0
    countC2=0
    c1T1=0 
    c1T2=0
    c2T1=0
    c2T2=0
    for i in range(len(labels)):
        if labels[i] == 1:
            countC1+=1
            if truelabels[i] == 1:
                c1T1+=1
            else:
                c1T2+=1
        else:
            countC2+=1
            if truelabels[i]==1:
                c2T1+=1
            else:
                c2T2+=1
    purity1=0
    purity2=0
   
    if c1T1>c1T2:
        purity1 = float(c1T1)/countC1
    else:
        purity1 = float(c1T2)/countC1

    if c2T1 > c2T2:
        purity2 = float(c2T1)/countC2
    else:
        purity2 = float(c2T2)/countC2 

    purities.append(purity1)
    purities.append(purity2)

    print purities
    return purities


def loadData2(fileDj):
    data = np.genfromtxt(fileDj, delimiter=' ')
    return data


# data = loadData("data_sets_clustering/humanData.txt")
# kneeFinding("data_sets_clustering/humanData.txt",[1,2,3,4,5,6])
#clusters = kmeans("data_sets_clustering/humanData.txt", 2, maxIter=1000)
# #visualizeClusters(clusters)
#purity("data_sets_clustering/humanData.txt",clusters)



# dataset1 = loadData2('data_sets_clustering/humanData.txt')
# dataset2 = loadData2('data_sets_clustering/audioData.txt')
# labels11,clustersGMM11 = gmmCluster(dataset1, 2, 'full')
# labels12,clustersGMM12 = gmmCluster(dataset1, 2, 'full')
# labels22,clustersGMM22 = gmmCluster(dataset2, 2, 'full')
# #purities11 = purityGMM(dataset1, clustersGMM11, labels11)
# purities22 = purityGMM(dataset2, clustersGMM22, labels22)
# print purities22


def main():
    #######dataset path
    datadir = sys.argv[1]
    pathDataset1 = datadir+'humanData.txt'
    pathDataset2 = datadir+'audioData.txt'
    dataset1 = loadData2(pathDataset1)
    dataset2 = loadData2(pathDataset2)

    #Q4
    kneeFinding(pathDataset1,range(1,7))

    #Q5
    clusters = kmeans(pathDataset1, 2, maxIter=1000)
    visualizeClusters(clusters)
    purity(pathDataset1,clusters)

    #Q7
    labels11,clustersGMM11 = gmmCluster(dataset1, 2, 'diag')
    labels12,clustersGMM12 = gmmCluster(dataset1, 2, 'full')

    #Q8
    labels21,clustersGMM21 = gmmCluster(dataset2, 2, 'diag')
    labels22,clustersGMM22 = gmmCluster(dataset2, 2, 'full')

    #Q9
    purities11 = purityGMM(dataset1, clustersGMM11, labels11)
    purities12 = purityGMM(dataset1, clustersGMM12, labels12)
    purities21 = purityGMM(dataset2, clustersGMM21, labels21)
    purities22 = purityGMM(dataset2, clustersGMM22, labels22)

if __name__ == "__main__":
    main()
