# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

#splitting dataset into training and test set
# install.packages('caTools')

library(caTools)

set.seed(123)

split = sample.split(dataset$Salary,SplitRatio = 2/3)#name of dependent variable is salary
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset,split == FALSE)