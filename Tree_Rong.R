## ST563 Final project
## Author: Rong Huang
## 4/15/2021

library(tidyverse)
library(caret)
require(tree)
require(randomForest)
require(gbm)

# Read in data set
wine <- read.csv2(file = "winequality-red.csv", header = TRUE, dec = ".",
                  colClasses = c(rep("double", 7), rep("double", 4), "integer"))
# Set seed for reproducibility
set.seed(702)
## Data Processing
# Prepare data for classification
wine$class <- ifelse(wine$quality <= 5, "low", "high") %>% as.factor()

# Create Data Partition
ind <- createDataPartition(wine$quality, p = .75, list = FALSE)
# Create train/test data for H1
train <- wine[ind,]
test <- wine[-ind,]

#--------------------
#    Desion Tree
#--------------------

#----- classification -----
# 1. full size decision tree
tree.class=tree(class~.-quality,train)
plot(tree.class)
text(tree.class, pretty=0)
tree.class.pred=predict(tree.class,test,type='class')
tree.class.train_err=mean(train$class!=predict(tree.class,type='class'))
tree.class.test_err=mean(test$class!=tree.class.pred)

# 2. pruned decision tree 
cv.tree.class=cv.tree(tree.class,FUN=prune.misclass)
plot(cv.tree.class)
prune.tree.class=prune.misclass(tree.class,best=8)
plot(prune.tree.class);text(prune.tree.class,pretty=0)
prune.tree.class.pred=predict(prune.tree.class,test,type='class')
prune.tree.class.train_err=mean(train$class!=predict(prune.tree.class,type='class'))
prune.tree.class.test_err=mean(test$class!=prune.tree.class.pred)

#----- regression -----
# 1. full size decision tree
tree.reg=tree(quality~.-class,train)
plot(tree.reg)
text(tree.reg, pretty=0)
tree.reg.pred=predict(tree.reg,test)
tree.reg.train_rmse=sqrt(mean((train$quality-predict(tree.reg))^2))
tree.reg.test_rmse=sqrt(mean((test$quality-tree.reg.pred)^2))

# 2. pruned decision tree 
cv.tree.reg=cv.tree(tree.reg)
plot(cv.tree.reg)
prune.tree.reg=prune.tree(tree.reg,best=8)
plot(prune.tree.reg);text(prune.tree.reg,pretty=0)
prune.tree.reg.pred=predict(prune.tree.reg,test)
prune.tree.reg.train_rmse=sqrt(mean((train$quality-predict(prune.tree.reg))^2))
prune.tree.reg.test_rmse=sqrt(mean((test$quality-prune.tree.reg.pred)^2))


#--------------------------------------
#              Bagging
#--------------------------------------

#----- classification -----
bag.class=randomForest(class~.-quality,train,mtry=11)
bag.class.pred=predict(bag.class,test)
bag.class.train_err=mean(train$class!=predict(bag.class))
bag.class.test_err=mean(test$class!=bag.class.pred)

#----- regression -----
bag.reg=randomForest(quality~.-class,train,mtry=11)
bag.reg.pred=predict(bag.reg,test)
bag.reg.train_err=sqrt(mean((train$quality-predict(bag.reg))^2))
bag.reg.test_err=sqrt(mean((test$quality-bag.reg.pred)^2))


#--------------------------------------
#              Random Forest
#--------------------------------------

#----- classification -----
test.err=double(11)
for (mtry in 1:11){
  fit=randomForest(class~.-quality,train,mtry=mtry)
  pred=predict(fit,test)
  test.err[mtry]=mean(test$class != pred)
}
matplot(1:mtry,test.err,pch=19,col='red',type='b',
        ylab='Mean Square Error')

rf.class=randomForest(class~.-quality,train,mtry=6)
rf.class.pred=predict(rf.class,test)
rf.class.train_err=mean(train$class!=predict(rf.class))
rf.class.test_err=mean(test$class!=rf.class.pred)

#----- regression -----
test.err=double(11)
for (mtry in 1:11){
  fit=randomForest(quality~.-class,train,mtry=mtry)
  pred=predict(fit,test)
  test.err[mtry]=sqrt(mean((test$quality - pred)^2))
}
matplot(1:mtry,test.err,pch=19,col='red',type='b',
        ylab='Mean Square Error')

rf.reg=randomForest(quality~.-class,train,mtry=2)
rf.reg.pred=predict(rf.reg,test)
rf.reg.train_err=sqrt(mean((train$quality-predict(rf.reg))^2))
rf.reg.test_err=sqrt(mean((test$quality-rf.reg.pred)^2))


#--------------------------------------
#              Boosting
#--------------------------------------

#----- classification -----
boost.class=gbm(class~.-quality,train,distribution="multinomial",n.trees=5000,
                shrinkage=0.01,interaction.depth=4)
summary(boost.class) 

boost.class=predict(boost.class,test,type="class",n.trees=5000)
boost.class.train_err=mean(train$class!=predict(boost.class,type="response",n.trees=5000))
boost.class.test_err=mean(test$class!=boost.class.pred)

#----- regression -----
boost.reg=gbm(quality~.-class,train,distribution="gaussian",n.trees=5000,
              shrinkage=0.01,interaction.depth=4)
summary(boost.reg) 

boost.reg=predict(boost.reg,test,n.trees=5000)
boost.reg.train_err=sqrt(mean((train$class-predict(boost.reg,n.trees=5000))^2))
boost.reg.test_err=sqrt(mean((test$class-boost.reg.pred)^2))





