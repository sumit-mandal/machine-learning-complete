#Random forest resgression

#importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#Fitting Random forest Regression to the dataset
# install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1],
                         y = dataset$Salary,
                         ntree = 300)

#predicting a new result
y_pred = predict(regressor,data.frame(Level =6.5))

#Visualising the plot 
library(ggplot2)
x_grid = seq(min(dataset$Level),max(dataset$Level),0.1)
ggplot() + 
  geom_point(aes(x = dataset$Level, y=dataset$Salary),
             color='red') +
  geom_line(aes(x = x_grid, y=predict(regressor,newdata = data.frame(Level = x_grid)))
            ,color = 'blue') + 
  ggtitle('Truth or bluff(Random Forest Regression)')+
  xlab('Level')+
  ylab('Salary')

             
  