# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

#Encoding categorical data

dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1,2,3))

#spliting the dataset into training and test test

library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset,split == FALSE)

#Fitting Multiple Linear Regression to the Training Set
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = training_set) #lm is fitting linear models
#profit is linear combination of all the independept variable


#predicting the test result
y_pred = predict(regressor,new_data = test_set)


#Building the optimal model using Backward Elimination

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = training_set)

summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend ,
               data = training_set)

summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = training_set)

summary(regressor)