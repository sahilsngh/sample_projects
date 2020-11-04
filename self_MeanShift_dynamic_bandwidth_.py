import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import time
from sklearn.datasets.samples_generator import make_blobs
style.use('ggplot')
colors = 10*["g","r","c","b","k"]

X, y = make_blobs(n_samples=50, centers=3, n_features=2)

# X = np.array( [ [1.4,2],[1.8,3],[1,1],[3,3],[2.5,2],
#                [1.5,8],[7,5],[8,7],[1,6.5],[2,7],
#                [1,9],[1.8,9],[7,7],[3.5,1],[9,5],[8,3],[6,5], ] )

class MeanShift:
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius_norm_step = radius_norm_step
        self.radius = radius
        
    def fit(self, data):
        
        if self.radius == None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step
        
        centroids = {}
        weights = [i for i in range(self.radius_norm_step)][::-1]
        
        for i in range(len(data)):
            centroids[i] = data[i]
            
        while True:
            new_centroids = []
            
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                
                for featureset in data:
                    distance = np.linalg.norm(featureset-centroid)
                    if distance == 0:
                        distance = 000000.1
                
                    weigth_index = int(distance/self.radius)
                
                    if weigth_index > self.radius_norm_step-1:
                        weigth_index = self.radius_norm_step-1

                    to_add = (weights[weigth_index]**2)*[featureset]
                    in_bandwidth += to_add

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))
                    
            uniques = sorted(list(set(new_centroids)))
            
            to_pop = []
            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
                        to_add.append(ii)
                        break
            
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass
                    
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
        
        self.classifications = {}
        for i in range(len(centroids)):
            self.classifications[i] = []
        
        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)
            
            
        
    def predict(self, data):
       
        distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

clf = MeanShift()
clf.fit(X)
centroids = clf.centroids

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], color=color, marker="*", s=150)

for i in centroids:
    plt.scatter(centroids[i][0], centroids[i][1], color='r', marker='x', s=150, linewidth=3)
plt.show()
