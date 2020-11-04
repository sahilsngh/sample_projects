'''This is an unsupervised ML algo demonstration
   Applying the Flat_KMeans clustering algorithms
   on the Titanic dataset
   Performed by Sahil Chauhan'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn import preprocessing
style.use('ggplot')

df = pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True)
#df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))
    return df

df = handle_non_numerical_data(df)
# TO see the list of all the columns 
#print(df.columns.values)


# TO tweak the columns to see if there is an affect on the accuracy.
#df.drop(['ticket','fare','boat'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])
boolean = True
counters = 0
''' to set the counters that how many time this'''
'''      algorithm will work'''
loop = 50

while boolean:
    clf = KMeans(n_clusters=2)
    clf.fit(X)

    correct = 0
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = clf.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1
    print(correct/len(X))
    counters += 1
    if counters == loop:
        boolean = False
        break


    






