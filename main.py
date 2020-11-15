#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:49:42 2020

@author: alfredocu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Read data.
digits = datasets.load_digits()
target, images = digits["target"], digits["images"]
# print(images.shape)

# Calculate data.
n_samples = digits["target"].shape[0]

# Show a random image.
sample = np.random.randint(n_samples)
plt.imshow(images[sample])
plt.title("Target: %i" % target[sample])

# Flatten images.
x = images.reshape((n_samples, -1))
# print(x.shape)

# Data for train and test.
xtrain, xtest, ytrain, ytest = train_test_split(x, target)

# Instance model.
model = svm.SVC(gamma = 0.0001)

# Training model.
model.fit(xtrain, ytrain)

# Apply metrics to the model.
print("Train: ", model.score(xtrain, ytrain))
print("Test: ", model.score(xtest, ytest))

# Make preddict of the test.
ypred = model.predict(xtest)

# Classification report.
print("\nClassification report: \n\n", metrics.classification_report(ytest, ypred))

# Confusion Matrix.
print("Confusion Matrix: \n\n", metrics.confusion_matrix(ytest, ypred))

# Draw a random prediction.
sample = np.random.randint(xtest.shape[0])
plt.imshow(xtest[sample].reshape((8,8)))
plt.title("Prediction: %i" % ypred[sample])
# plt.savefig("Predict_1.eps", format="eps")