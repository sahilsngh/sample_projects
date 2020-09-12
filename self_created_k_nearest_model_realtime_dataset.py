import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random

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

    return vote_result

df = pd.read_csv('breast-cancer-wisconsin.csv')
df.replace('?', -99999, inplace=True)
df.drop('id', 1, inplace=True)
# print(df.head())
true_dataset = df.astype(float).values.tolist()
random.shuffle(true_dataset)

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = true_dataset[:-int(test_size*len(true_dataset))]
test_data = true_dataset[-int(test_size*len(true_dataset)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

# these are the one line for loop code for upper two for loops
# [train_set[i[-1]].append(i[:-1]) for i in train_data]
# [test_set[i[-1]].append(i[:-1]) for i in test_data]

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy', correct/total)
