# CLASSIFY POINTS
# 1. find distances from each point to each centroid (distances)
# 2. which centroid is closest (smdist)
# 3. Get colors to match


import numpy as np
import matplotlib.pyplot as plt
import random

class KMeansAlgorithm():
    def __init__(self, num):
        self.N = num

    def cluster(self):
        x0 = random.random() * 10
        y0 = random.random() * 10
        return np.random.normal((x0, y0), 1, (300, 2)) # just a random array distr normally around 0, ( 1 is std dev)

    def clusterNumber(self):
        return np.concatenate([self.cluster() for i in range(self.N)]) # data.shape is the size of the np array . Randonly choosing rows to get smaple from

    def plotting(self, data, centroids, clusters):
        clrs = ('blue', 'm', 'black', 'red')
        plt.scatter(data[:, 0], data[:, 1], c = [clrs[c] for c in clusters], s = 10)
        plt.scatter(centroids[:, 0], centroids[:, 1], s = 500, c = clrs, marker = '*')  # plot centroids
        plt.show()

    def distances(self, point, centroid):
        # gives distances for for point to all three centroids.
        sumC = [np.sum(((point - centroid) ** 2), axis = 1).reshape((-1, 1)) for centroid in centroid]
        return np.concatenate(sumC, axis = 1)

    def smdist(self, distances):
        return distances.argmin(axis = 1) # gives you smallest thing in each row of an array into an array


def main():
    KMeans = KMeansAlgorithm(4)
    mData = KMeans.clusterNumber()
    mRows = random.sample(range(mData.shape[0]), KMeans.N)
    mCentroids = mData[mRows]

    minDist = np.array([])
    while True:
        prev = minDist.copy()
        minDist = KMeans.smdist(KMeans.distances(mData, mCentroids))  # calculates smallest distances

        if np.all(prev == minDist):
            break

        for i in range(KMeans.N):
            mCentroids[i] = np.average(mData[minDist == i], axis = 0) # avg x val followed by avg y value

        KMeans.plotting(mData, mCentroids, minDist)


if __name__ == '__main__':
    main()
