#Decision tree regression

#importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#Fitting decision tree regression to the dataset

# install.packages("rpart")
library(rpart)
regressor = rpart(formula=Salary~.,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

#predicting a new result
y_pred = predict(regressor,data.frame(Level=6.5))

#visualising the plot
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +
  ylab('Salary')