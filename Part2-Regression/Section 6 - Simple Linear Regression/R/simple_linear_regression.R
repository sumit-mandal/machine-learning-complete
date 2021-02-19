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

#fitting simple linear regression into the training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

#predicting the test set result

y_pred = predict(regressor, newdata = test_set)

#visualising the training set results
# install.packages('ggplot2')
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') + #it is used to scatter plot
                              #it shows real value.That is original one
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor,newdata = training_set)),
            color = 'blue') + #it'll show predicted salaries of training set
  
  ggtitle('salary vs Experience(Training set)') + 
  xlab('years of Experience')+
  ylab('salary')


#visualising test set results

ggplot() + 
  geom_point(aes(x = test_set$YearsExperience,y = test_set$Salary),
             color = 'green')+
  geom_line(aes(x = training_set$YearsExperience,y = predict(regressor,newdata = training_set)),
            color = 'blue')+ 
  ggtitle('salary vs Experience(Test set)')+
  xlab('years of Experience')+
  ylab('salary')
  


  