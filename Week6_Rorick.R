#Assignment Week 6
#Cody Rorick

library(caret)
library(AppliedPredictiveModeling)
data(hepatic)

#Problem 12.1c Pre-process the data, split the data into a training and a testing set, and build models described
#in this chapter for the biological predictors. Using each model to predict on the testing set, which
#model has the best predictive ability for the biological predictors and what is the optimal
#performance?
low.variability.columns <- nearZeroVar(bio)
bio <- bio[,-low.variability.columns]
correlations <- cor(bio)
library(corrplot)
corrplot(correlations, order = "hclust")
highly.correlated.columns <- findCorrelation(correlations, cutoff = .85)
bio <- bio[,-highly.correlated.columns]
set.seed(100)
trainRows <- createDataPartition(injury, p = .80, list = FALSE)
trainResp <- injury[trainRows]
trainPred <- bio[trainRows,]
testResp <- injury[-trainRows]
testPred <- bio[-trainRows,]

### Linear Discriminant Analysis
ctrl <- trainControl(method = "LGOCV", number = 10, classProbs = TRUE, savePredictions = TRUE)
LDAFull <- train(trainPred, y = trainResp, method = "lda", preProc = c("center","scale"), metric = "Kappa", trControl = ctrl)
LDAFull
confusionMatrix(data = predict(LDAFull, testPred), reference = testResp) #average over 10*.25*1000 observations

#### Partial Least Squares Discriminant Analysis
set.seed(100)
ctrl <- trainControl(method = "LGOCV", number = 10, classProbs = TRUE, savePredictions = TRUE)
plsFit <- train(x = trainPred, y = trainResp, method = "pls",tuneGrid = expand.grid(.ncomp = 1:10), preProc = c("center","scale"), metric = "Kappa", trControl = ctrl)
plsFit
plot(plsFit)
confusionMatrix(data = predict(plsFit, testPred), reference = testResp)

###Penalized glmnet model
library(glmnet)
ctrl <- trainControl(method = "LGOCV", number = 10, classProbs = TRUE,savePredictions = TRUE)
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6),.lambda = seq(.01, .2, length = 5))
set.seed(100)
glmnTuned <- train(trainPred, y = trainResp, method = "glmnet", tuneGrid = glmnGrid, preProc = c("center", "scale"), metric = "Kappa", trControl = ctrl)
glmnTuned
plot(glmnTuned)
confusionMatrix(data = predict(glmnTuned, testPred), reference = testResp)

#Problem 12.1d For the optimal model for the biological predictors, what are the top five important predictors?
varImp(glmnTuned)


#Problem 12.3a Explore the data by visualizing the relationship between the predictors and the outcome. Are
#there important features of the predictor data themselves, such as between-predictor correlations
#or degenerate distributions?
library(modeldata)
data("mlc_churn")
low.variability.columns <- nearZeroVar(mlc_churn)
hist(unlist(mlc_churn[,6]), main = 'Vmail Messages', xlab = '# Vmail Messages')
mlc_churn <- mlc_churn[,-low.variability.columns]
correlations <- cor(mlc_churn[,-c(1,3,4,5,19)])
corrplot(correlations, order = "hclust")
highly.correlated.columns <- findCorrelation(correlations, cutoff = .85)
mlc_churn <- mlc_churn[,-highly.correlated.columns]

#Problem 12.3c Split the data into training set and test set using random splitting (80% and 20%). Fit models
#covered in this chapter to the training set and tune them via resampling. Which model has the best performance?
set.seed(100)
trainRows <- createDataPartition(mlc_churn$churn, p = .80, list = FALSE)
trainResp <- mlc_churn$churn[trainRows]
trainPred <- mlc_churn[trainRows,-c(15)]
testResp <- mlc_churn$churn[-trainRows]
testPred <- mlc_churn[-trainRows,-c(15)]

### Logistic Regression
set.seed(100)
ctrl <- trainControl(method = "LGOCV", summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE)
lrFull <- train(trainPred, y = trainResp, method = "glm",preProc = c("center", "scale"), metric = "ROC", trControl = ctrl)
lrFull
confusionMatrix(data = predict(lrFull, testPred), reference = testResp)
library(pROC)
rocCurve <- roc(response = testResp, predictor = predict(lrFull, testPred, type = 'prob')[,2])
auc(rocCurve)

### Penalized glmnet model
library(glmnet)
ctrl <- trainControl(method = "LGOCV",summaryFunction = twoClassSummary,classProbs = TRUE,savePredictions = TRUE)
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6),.lambda = seq(.01, .2, length = 5))
set.seed(100)
glmnTuned <- train(trainPred, y = trainResp, method = "glmnet",tuneGrid = glmnGrid,preProc = c("center", "scale"),metric = "ROC",trControl = ctrl)
glmnTuned
plot(glmnTuned)
confusionMatrix(data = predict(glmnTuned, testPred), reference = testResp)
rocCurve <- roc(response = testResp, predictor = predict(glmnTuned, testPred, type = 'prob')[,1])
auc(rocCurve)

### Neural Net Model
nnetGrid <- expand.grid(.size = 1:3, .decay = c(0, .1, .3, .5, 1))
maxSize <- max(nnetGrid$.size)
numWts <- (15 * (14 + 1) + (15+1)*2) ## 14 is the number of predictors; 2 is the number of classes; 15 is size*decay
ctrl <- trainControl(method = 'LGOCV', summaryFunction = twoClassSummary,classProbs = TRUE)
nnetFit <- train(x = trainPred, y = trainResp,method = "nnet",metric = "ROC",preProc = c("center", "scale", "spatialSign"),tuneGrid = nnetGrid,trace = FALSE,maxit = 2000,MaxNWts = numWts,trControl = ctrl)
nnetFit
plot(nnetFit)
confusionMatrix(data = predict(nnetFit, testPred), reference = testResp)
rocCurve <- roc(response = testResp, predictor = predict(nnetFit, testPred, type = 'prob')[,1])
auc(rocCurve)

### Mixed Discriminant Analysis
library(mda)
ctrl <- trainControl(method = "LGOCV", summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE) 
set.seed(100)
mdaFit <- train(x = trainPred, y = trainResp,method = "mda",metric = "ROC", tuneGrid = expand.grid(.subclasses = 1:4),trControl = ctrl)
mdaFit
plot(mdaFit)
confusionMatrix(data = predict(mdaFit, testPred), reference = testResp)
rocCurve <- roc(response = testResp, predictor = predict(mdaFit, testPred, type = 'prob')[,1])
auc(rocCurve)
