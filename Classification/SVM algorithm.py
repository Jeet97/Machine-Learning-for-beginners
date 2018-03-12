from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



#Traning Phase

features_train = [[x1,x2,...],.....]#Features Dataset 
labels_train = [y1, y2, y3,...] #Labels Dataset





clf = SVC(C = 10000.)  # Higher C parameter refers to high accuracy i.e more complex decision surface.



clf.fit(features_train, labels_train)    #Feeding the Classifier with input traning datasets.

#Testing Phase


pred = clf.predict(features_test)      #features_test is your testing dataset.


#Testing Accuracy


acc = accuracy_score(pred, labels_test)  #lables test is your actual labels dataset for features_test dataset to test the accuracy of model. 

print acc
