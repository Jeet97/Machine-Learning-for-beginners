from sklearn.cluster import KMeans
import numpy as np


#Traning Phase
features = [[x1,x2,...],...] #Features Dataset
labels = [y1,y2,...] #labels Dataset

x = ny.array(features)
y = ny.array(labels)

clf = KMeans(n_clusters = 2) #n_clusters refers to the no of clusters that should be formed to classify data.

clf.fit(x,y)    #Feeding the Classifier with input traning datasets.

#Testing Phase

pred = clf.predict(features_test,labels_test)     #features_test is your testing dataset.


#Testing Accuracy
from sklearn.metrics import accuracy_score

acc = accuracy_score(pred,labels_test)  #lables test is your actual labels dataset for features_test dataset to test the accuracy of model.

print acc
