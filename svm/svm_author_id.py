#!/usr/bin/python

"""
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors

    Sara has label 0
    Chris has label 1

"""

import sys
from collections import Counter
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

print "importing"
from sklearn.svm import SVC
print "done importing"

clf = SVC(kernel='rbf', C=10000.0)
print "done svc init"

t0 = time()
clf.fit(features_train, labels_train)
print "done fitting"

t0 = time()
pred =  clf.predict(features_test)
print pred
print Counter(pred)
print "predicting time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
t0 = time()
print accuracy_score(labels_test, pred)
print "testing time:", round(time()-t0, 3), "s"

#########################################################


