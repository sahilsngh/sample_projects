import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import time
style.use('ggplot')

X = np.array( [ [1.4,2],[1.8,3],[1,1],[3,3],[2.5,2],
               [1.5,8],[7,5],[8,7],[1,6.5],[2,7],
               [1,9],[1.8,9],[7,7],[3.5,1],[9,5],[8,3],[6,5], ] )

##plt.scatter(X[:,0], X[:,1], color='g', marker='*', s=150)
##plt.show()

class MeanShift:
    def __init__(self, radius=2):
        self.radius = radius
        
    def fit(self, data):
        centroids = {}
        
        for i in range(len(data)):
            centroids[i] = data[i]
            
        while True:
            new_centroids = []
            in_bandwidth = []
            
            for i in centroids:
                centroid = centroids[i]
                
                for featureset in data:
                    if np.linalg.norm(featureset-centroid) < self.radius:
                        in_bandwidth.append(featureset)
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))
                    
            uniques = sorted(list(set(new_centroids)))
            
            prev_centroids = dict(centroids)
            
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
                
            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
            if optimized:
                break
        self.centroids = centroids
        
    def predict(self, data):
        pass

clf = MeanShift()
clf.fit(X)
centroids = clf.centroids
plt.scatter(X[:,0], X[:,1], color='g', marker='*', s=150)
for i in centroids:
    plt.scatter(centroids[i][0], centroids[i][1], color='r', marker='*', s=150)
plt.show()
