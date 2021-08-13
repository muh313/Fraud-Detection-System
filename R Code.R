cc_fraud_trainingset <- read.csv('C:\\Users\\Mhasa\\Desktop\\Applied Data Analytics\\cc_fraud_trainingset.csv')
cc_fraud_testset <- read.csv('C:\\Users\\Mhasa\\Desktop\\Applied Data Analytics\\cc_fraud_testset.csv')

library(tidyverse)
library(ggplot2)

#tabular methods
str(cc_fraud_trainingset)
cc_fraud_trainingset$Class <- factor(cc_fraud_trainingset$Class, levels = c(0, 1))
cc_fraud_testset$Class <- factor(cc_fraud_testset$Class, levels = c(0, 1))
library(skimr)
skim(cc_fraud_trainingset)
skim(cc_fraud_testset)
sum(is.na(cc_fraud_trainingset))
aTable <- table(cc_fraud_trainingset$Class)
table(cc_fraud_testset$Class)
NumOfEntries <- summary(cc_fraud_trainingset$Class)

#graphical methods
labels <- c("legitimate", "fraud")
labels <- paste(labels, round(100*prop.table(table(cc_fraud_trainingset$Class)), 2))
pie(table(cc_fraud_trainingset$Class), labels, col = c("red", "black"),
    main = "A Pie Chart for the credit card transactions!")
myFrame <- as.data.frame(table(aTable))
ggplot(myFrame, aes(x=aTable, y=NumOfEntries))

#Plotting scatter graph predictions
predictions <- rep.int(0, nrow(cc_fraud_testset))
predictions <- factor(predictions, levels = c(0,1))
library(caret)
confusionMatrix(data = predictions, reference = cc_fraud_testset$Class)
#Compares legit and fraud cases (V1 and V2)
library(ggplot2)
ggplot(data = cc_fraud_testset, aes(x = V1, y = V2, col = Class)) +
  geom_point()

#finding missing values
summary_df <- do.call(cbind, lapply(cc_fraud_trainingset[,1:2], summary))
#round values to integer
summary_df_t <- as.data.frame(round(t(summary_df),0))
#replace a name for last column to missing value
names(summary_df_t)[7] <- paste("Missing_values")
# using tideverse modify the data frame to add new columns a number of observation obs and missing proportion
library(tidyverse)
summary_df_t_2 <- summary_df_t %>% 
  mutate(obs = nrow(cc_fraud_trainingset),
         Missing_prop = Missing_values / obs)
print(summary_df_t_2)
#calculate the averages for all counts. 
summary_df_t_3<-summary_df_t_2 %>% summarise(Min = mean(Min.),
                                             first_Q = mean(`1st Qu.`),
                                             Median = median(Median),
                                             Mean = mean(Mean),
                                             third_Q = mean(`3rd Qu.`),
                                             Max = max(Max.),
                                             mean_MV = mean(Missing_values),
                                             obs = mean(obs),
                                             mean_MV_perc = mean_MV / obs*100)  #this generates the percentage

summary_df_t_3
#the mean _MV_perc  represents the percentage of missing value in the cc_fraud_trainingset dataset
#if the percentage is higher than 1%, meaning we must impute it (using mice())
library(mice)
mice(summary_df)
#to show you if there's any missing data
library(Amelia)
missmap(summary_df_t)

#detecting outliers
#set the generator for random number to get the same results everytime we run the code
set.seed(0)
#generate a vector with outliers
cc_fraud_trainingset<-c( rnorm(90), rep(1000,3))
#calculate z-scores
z_scores<-(cc_fraud_trainingset-mean(cc_fraud_trainingset))/sd(cc_fraud_trainingset)
#elements for each z-scores are < - 3 or >3 are consider to be outliers
outliers<-which(z_scores<(-3) | z_scores>3) # which return a number of indexes of elements meeting the condition
#see which elements are outliers
cc_fraud_trainingset[outliers]
#checking multicollinearity - deleting identical variables
cc_fraud_trainingset <- read.csv('C:\\Users\\Mhasa\\Desktop\\Applied Data Analytics\\cc_fraud_trainingset.csv')
head(cc_fraud_trainingset)
mydata <- data.frame(cc_fraud_trainingset[,-1])
head(mydata)
cor(mydata)
#now to remove all those NA's...
mydata = na.omit(mydata)
cor(mydata)
#converting to a model
mymodel <- lm(Class~., mydata)
mymodel
summary(mymodel)

#now lets test it for multicollinearity
library(car)
vif(mymodel)

#scaling in r - centers all columns
X <- cc_fraud_trainingset$Time
Y <- cc_fraud_trainingset$Amount
#calculates difference between time and amount
scale(cbind(X,Y))

#Modeling

library(pROC)
library(caret)
library(nnet)
library(randomForest)
library(WeightSVM)

#na.omit() ensure/handles and removes missing values
trainset = na.omit(cc_fraud_trainingset)
testset = na.omit(cc_fraud_testset)

y = trainset$Class
w = ifelse(y==0,c(30),c(700))

trainset$Class = as.factor(trainset$Class)
testset$Class = as.factor(testset$Class)

# # Naive-Bayes model fitting with 3 fold cross validation
model = train(subset(trainset,select = -c(Class)),trainset$Class,'nb',weights = w,trControl=trainControl(method='cv',number=3))
ylabels = predict(model,newdata = testset)
probs = predict(model,newdata = testset,type = 'prob')
cm = confusionMatrix(ylabels,factor(testset$Class),positive = c('1'))
cm
curve = roc(ylabels,probs$`1`)
dev.new()
plot(curve,print.auc = TRUE)
total_cost = 30*cm$table[2,1] + 700*cm$table[1,2]
cat('Total cost for Naive Bayes model:',total_cost)

# Random forest model fitting
rfmodel = randomForest(Class ~ .,data = trainset,ntree = 150,classwt=c(30,700))
rfmodel
ylabels = predict(rfmodel,newdata = testset)
probs = predict(rfmodel,newdata = testset,type = 'prob')
cm = confusionMatrix(ylabels,factor(testset$Class),positive = c('1'))
cm
curve = roc(ylabels,probs[,2])
dev.new()
plot(curve,print.auc = TRUE)
total_cost = 30*cm$table[2,1] + 700*cm$table[1,2]
cat('Total cost for RF:',total_cost)

# logistic regression model
logitmodel = glm(Class ~., family = binomial(link = "logit"),data = trainset,weights = w)
logitmodel
probs = predict(logitmodel,newdata = testset,type = 'response')
ylabels = factor(ifelse(probs > 0.5,'1','0'))
cm = confusionMatrix(ylabels,factor(testset$Class),positive = c('1'))
cm
curve = roc(ylabels,probs)
dev.new()
plot(curve,print.auc = TRUE)
total_cost = 30*cm$table[2,1] + 700*cm$table[1,2]
cat('Total cost for Logistic model:',total_cost)

# SVM model fitting (WeightedSVM package downloaded)
svmmodel = wsvm(Class ~ .,data = trainset,weight = w)
ylabels = predict(svmmodel,newdata = testset)
probs = predict(svmmodel,newdata = testset,type = 'raw')
extract <- c(probs)
curve = roc(ylabels,extract)
dev.new()
plot(curve,print.auc = TRUE)
cm = confusionMatrix(ylabels,factor(testset$Class),positive = c('1'))
cm
total_cost = 30*cm$table[2,1] + 700*cm$table[1,2]
cat('Total cost for SVM model:',total_cost)

# Neural network model fitting
set.seed(504)
netmodel = nnet(Class ~ ., data = trainset,weights = w,size = 10)
ylabels = as.factor(predict(netmodel,newdata = testset,type = 'class'))
probs = predict(netmodel,newdata = testset,type = 'raw')
curve = roc(ylabels,probs)
dev.new()
plot(curve,print.auc = TRUE)
cm = confusionMatrix(ylabels,factor(testset$Class),positive = c('1'))
cm
total_cost = 30*cm$table[2,1] + 700*cm$table[1,2]
cat('Total cost for Neural network model:',total_cost)
