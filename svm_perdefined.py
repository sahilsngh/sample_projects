import numpy as np
from sklearn import svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.csv')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y =np.array(df['class'])

ts = 0.2
X_train = X[:-int(len(X)*ts)]
X_test = X[-int(len(X)*ts):]
y_train = y[:-int(len(y)*ts)]
y_test = y[-int(len(y)*ts):]

# print(len(X_train), len(y_train), len(X_test), len(y_test))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy*100)


