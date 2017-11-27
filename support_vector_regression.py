import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("bitcoin_price.csv")

msk = np.random.rand(len(df)) < 0.8

X = df["Open"]
Y = df["High"]
X_train = np.array(X[msk]).reshape(len(X[msk]),1)
X_test = np.array(X[~msk]).reshape(len(X[~msk]),1)
Y_train = np.array(Y[msk]).reshape(len(X[msk]),1)
Y_test = np.array(Y[~msk]).reshape(len(X[~msk]),1)

clf = svm.SVR(kernel='poly', C=1, degree=1)
clf.fit(X_train,Y_train)
predicted_vals = clf.predict(X_test)

plt.scatter(X_test, Y_test, color='darkorange', label='data')
plt.plot(X_test, Y_test, color='blue', linewidth=3)
plt.show()