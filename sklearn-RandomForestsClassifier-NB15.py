from sklearn import preprocessing
from scipy.io.arff import loadarff
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


NB15_train = pd.read_csv('UNSW_NB15_training-set.csv', delimiter=',')
NB15_test = pd.read_csv('UNSW_NB15_testing-set.csv', delimiter=',')

training_nparray = NB15_train.to_numpy()  # This is necessary to correctly shape the array
testing_nparray = NB15_test.to_numpy()


# Preprocess
enc = preprocessing.OrdinalEncoder()

encoded_dataset = enc.fit_transform(training_nparray)  # All categorical features are now numerical
X_train = encoded_dataset[:, :-1]  # All rows, omit last column
y_train = np.ravel(encoded_dataset[:, -1:])  # All rows, only the last column

# Repeat preprocessing for test data
encoded_dataset = enc.fit_transform(testing_nparray)
X_test = encoded_dataset[:, :-1]
y_test = np.ravel(encoded_dataset[:, -1:])

# Fit to model and predict
rfc = RandomForestClassifier()
y_pred = rfc.fit(X_train, y_train).predict(X_test)

total_datapoints = X_test.shape[0]
mislabeled_datapoints = (y_test != y_pred).sum()
correct_datapoints = total_datapoints-mislabeled_datapoints
percent_correct = (correct_datapoints / total_datapoints) * 100

print("RandomForestClassifier results for UNSW_NB15:\n")
print("Total datapoints: %d\nCorrect datapoints: %d\nMislabeled datapoints: %d\nPercent correct: %.2f%%"
      % (total_datapoints, correct_datapoints, mislabeled_datapoints, percent_correct))
