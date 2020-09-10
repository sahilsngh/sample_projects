''' This model is trained on SciLearn in python by Sahil Chauhan.
    In this model python uses its pandas library to extract the data_frame.
    The data_frame consist of more than 50k entries over time of features
    Diamond and also consist of its price in Rupeees (was in USD).
    The data_frame is provided by "Shivam Agrawal" from "https://www.kaggle.com/shivam2503/diamonds"
'''

import pandas as pd
import sklearn
from sklearn import svm, preprocessing
import time


last_time = time.time()
#This section gets the csv file from github repository.
#If this section_code doesn't work you can try to download the csv file and retrieve it locally on your pc.
url = 'https://raw.githubusercontent.com/sahilsngh/ML_exersize/master/diamonds.csv'
df = pd.read_csv(url, error_bad_lines=True, index_col=0)
print(f"Time takas to download the file:{time.time()-last_time} seconds")

#creating a dictionaries to get the numerical values for non-numerical columns. 
last_time = time.time()
cut_dict = {"Fair": 5 , "Good": 4 , "Very Good": 3 , "Premium": 2 , "Ideal": 1} 
color_dict = {"D": 1 , "E": 2 , "F": 3 , "G": 4 , "H": 5 , "I": 6 , "J": 7}
clarity_dict = {"FL": 1 , "IF": 2 , "VVS1": 3, "VVS2": 4 , "VS1": 5 , "VS2": 6 , "SI1": 7, "SI2": 8, "I1": 9 , "I2": 10 , "I3": 11}

#Mapping the dict with data_frame with its respected columns.
df['cut'] = df['cut'].map(cut_dict)
df['color'] = df['color'].map(color_dict)
df['clarity'] = df['clarity'].map(clarity_dict)
print(f"Time takas to process the file:{time.time()-last_time} seconds")
df['price'] = df['price'] * 77

#Shuffling the data_frame.
last_time = time.time()
df = sklearn.utils.shuffle(df)

X = df.drop("price", axis=1).values
X = preprocessing.scale(X)
y = df["price"].values

test_size = 500

X_train = X[:-test_size] 
y_train = y[:-test_size]

X_test = X[-test_size:]
y_test = y[-test_size:]

#This is rbf regrssion model.         
clf = svm.SVR(kernel="linear")
clf.fit(X_train, y_train)
print(f"Time takas to train the Model:{time.time()-last_time} seconds")

'''
#for linear regression model you can use-
clf = svm.SVR(kernel="linear")
clf.fit(X_train, y_train)
'''

#Print success rate of your model prediction
print(f"Success rate: {clf.score(X_test, y_test) * 100}")

#To check your model prediction for first 500 batch.
a = 0
for X,y in zip(X_train, y_train):
  print(f"Model: {clf.predict([X])[0]},Actaul: {y} and the difference in prediction {y-clf.predict([X])[0]}")
  a += 1
  if a == 500:
    break







