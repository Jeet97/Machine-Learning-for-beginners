from sklearn.linear_model import LinearRegression


# Training Phase

input_train = [[x1,x2,x3],.....]  #Input variables that determines the output.

output_train = [y1,y2,y3,....]    #Output for each input.



reg = LinearRegression()        #Instantiate LinearRegression object with default parameters.

reg.fit(input_train, output_train)       #Feeding the classifier with training data

# Testing Phase

print reg.predict(input_test)           #input test will be a list or numpy array of testing data.

print reg.coef_,reg.intercept_                   # Lists the slope and intercept of each function btw input and output.

print reg.score(input_train, output_train)       #reg.score() returns r squared score.Higher the score, greater will be the accuracy.   


