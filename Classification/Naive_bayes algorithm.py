import numpy as ny
from sklearn.naive_bayes import GaussianNB



#Traning Phase

features_train = [[x1,x2,...],.....]#Features Dataset 
labels_train = [y1, y2, y3,...] #Labels Dataset


x = ny.array(features_train)   #we will use numpy array as input for features and labels.
y = ny.array(labels_train)
clf = GaussianNB()              #Instantiate the classifier.


clf.fit(x,y)                 #Feeding the classifier with input traning datasets.

#Testing Phase


clf.predict(features_test)      #features_test is your testing dataset.


#Testing Accuracy of model

clf.score(features_test,labels_test)  #lables test is your actual labels dataset for features_test dataset to test the accuracy of model.

