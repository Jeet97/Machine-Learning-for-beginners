from sklearn.ensemble import RandomForestClassifier



#Training Phase
fearures_train = [[x1,x2,x3,....].....]
labels_train = [y1, y2, y3.....]



clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(features_train, labels_train)


#Testing Phase


pred = clf.predict(features_test)      #features_test is your testing dataset.


#Testing Accuracy


acc = accuracy_score(pred, labels_test)  #lables test is your actual labels dataset for features_test dataset to test the accuracy of model. 

print acc
