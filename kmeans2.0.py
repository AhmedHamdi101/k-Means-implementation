import struct
import numpy as np
import pandas as pd
import matplotlib as plt

#the class help to save each image and their index
class object:
    def __init__(self, array, index):
        self.array = array
        self.index = index

#function to read the data from file and put it in a 28 by 28 matrix
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def kmeans(image,k):
    #array that hold the ecludian distance
    dist = []
    #array that save the object containig the vector of the image and it's index
    cluster = [[]]
    #another array with only the matrix that help in speeding the code
    cluster2 = [[]]
    #array that will be used to match every label with cluster
    clusterLabel = []
    #array that work as a temp to calculate the true value of a cluster in each cluster
    recur = []
    #array used to calculate the accuracy
    accuracy = []
    #array used to calculate the error
    error = [[]]

    #loop to intialize all the 2d arrays
    for i in range (k):
        cluster.append([])
        cluster2.append([])
        error.append([])
    #loop to help intialize the 1d arrays
    for i in range(k):
        clusterLabel.append(0)
        recur.append(0)
        accuracy.append(0)
    #reshaping the 28 by 28 matrix into 28*28 linear vector
    data = np.reshape(image, (len(image) , 28*28) )
    #reading the labels to be used in the cluster naming
    labls = read_idx("train-labels-idx1-ubytee")
    #intializing the centriond by k numbers
    centroids = np.random.randint(0,256,(k,28*28))
    #main loop to cluster
    for i in range(20): ## outer main loop
        print(i)
        for j in range(len(cluster)):
            cluster[j].clear()
            cluster2[j].clear()

        for j in range(len(data)):##len(image)
            for m in range(k):
                #calculating the ecludian distance between every centriod
                dist.append(np.linalg.norm(data[j]-centroids[m]))
            cluster[np.argmin(dist)].append(object(data[j],j))
            #chosing the centroid with the minimum distance
            cluster2[np.argmin(dist)].append(data[j])
            dist.clear()
        #after calculating all the cluster we will need to update the centroids
        for j in range(len(cluster2) -1):
            new = np.array(cluster2[j])
            if len(new) != 0 :
                df = new.mean(axis=0)
                centroids[j] = df
    for j in range(k): #print to check centroids are really changed
        print(centroids[j])

    #loop that define each cluster to a label
    for i in range(len(cluster) -1):
        print(i)
        for h in range(k):
            recur[h] = 0
        for j in range(len(cluster[i])):
            index = cluster[i][j].index
            recur[labls[index]] = recur[labls[index]] + 1
        print("a looop")
        print(recur)
        clusterLabel[i] = np.argmax(recur)
        if np.amax(recur) != 0:
            #calculating the cluster accuracy
            accuracy[i] = np.amax(recur) / len(cluster[i])
    acc = sum(accuracy) / k
    print(clusterLabel)
    print(acc)
    # for i in range(len(centroids)):
    #     ima = centroids[i].reshape(28,28)
    #     plt.imshow(ima,cmap='gray_r',interpolation='nearst')
    #     plt.xticks(())
    #     plt.yticks(())



image = read_idx("train-images-idx3-ubyte")
kmeans(image,16)
