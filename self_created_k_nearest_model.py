import numpy as np
from collections import Counter
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

dataset = {'k': [[1, 2], [3, 3], [2, 5]], 'r': [[5, 6], [5, 8], [7, 7]]}
test_data = [6, 5]

def k_nearest_neighbors(data, test, k=3):
    if len(data) >= k:
        warnings.warn('You are an idiot!')
    distance = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(test))
            distance.append([euclidean_distance, group])

    vote = [i[1] for i in sorted(distance)[:k]]
    vote_result = Counter(vote).most_common(1)[0][0]
    print(Counter(vote).most_common(1))

    return vote_result

result = k_nearest_neighbors(dataset, test_data, k=3)
print(result)

# # this for function is for plot
# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], size=100, color=i)
# # one line for loop for plot for this upper function
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(test_data[0], test_data[1], s=100, color='g')
plt.show()
