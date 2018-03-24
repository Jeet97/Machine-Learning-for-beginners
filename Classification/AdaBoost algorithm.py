from sklearn.ensemble import GradientBoostingClassifier


#Traning Phase
features_train = [[x1,x2,...],...] #Features Dataset
labels_train = [y1,y2,...] #labels Dataset


clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0) 

clf.fit(features_train,labels_train)    #Feeding the Classifier with input traning datasets.

#Testing Phase

clf.predict(features_test,labels_test)     #features_test is your testing dataset.


#Testing Accuracy

print clf.score(features_test, labels_test)  #lables test is your actual labels dataset for features_test dataset to test the accuracy of model.
