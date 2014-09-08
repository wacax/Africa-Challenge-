#Liberty Mutual Group - Fire Peril Loss Cost
#ver 0.1

#########################
#Init
rm(list=ls(all=TRUE))

#Load/install required libraries
require('data.table')
require('ggplot2')
require('leaps')
require('caret')
require('rpart')
require('gbm')
require('survival')
require('splines')
require('survival')
require('parallel')
require('plyr')

#Set Working Directory
workingDirectory <- '/home/wacax/Wacax/Kaggle/Africa Tuesday/Africa Soil Property Prediction Challenge/'
setwd(workingDirectory)

dataDirectory <- '/home/wacax/Wacax/Kaggle/Africa Tuesday/Data/'

#Load external functions

#############################
#Load Data
#Input Data
train <- read.csv(paste0(dataDirectory, 'training.csv'), header = TRUE, stringsAsFactors = FALSE)
test <- read.csv(paste0(dataDirectory, 'sorted_test.csv'), header = TRUE, stringsAsFactors = FALSE)

submissionTemplate <- read.csv(paste0(dataDirectory, 'sample_submission.csv'), header = TRUE, stringsAsFactors = FALSE)

################################
#Data Processing
#Data Transformation
train <- transform(train, Depth = as.factor(Depth))

################################
#EDA
#Rows containing NAs
noNAIndices <- which(apply(is.na(train), 1, sum) == 0)

#Ca histogram
ggplot(data = train, aes(x = log(Ca))) +  geom_histogram() 
#P histogram
ggplot(data = train, aes(x = log(P))) +  geom_histogram() 
#pH histogram
ggplot(data = train, aes(x = pH)) +  geom_histogram()
#SOC histogram
ggplot(data = train, aes(x = SOC)) +  geom_histogram()
#Sand histogram
ggplot(data = train, aes(x = Sand)) +  geom_histogram()

###################################################
#Selection using trees / caret
randomSubset <- sample.int(nrow(train), nrow(train))
linearBestModels <- regsubsets(Ca ~ ., data = train[randomSubset , seq(2, 3596)], 
                               method = 'forward', nvmax = 100)
bestMods <- summary(linearBestModels)
plot(bestMods$cp, xlab="Number of Variables", ylab="CP Error")
points(which.min(bestMods$cp), bestMods$cp[which.min(bestMods$cp)],pch=20,col="red")

ensembleFeatures <- as.data.frame(bestMods$which)
ensembleFeatures <- sort(apply(ensembleFeatures, 2, sum), decreasing = TRUE, index.return = TRUE)
ensembleFeatures <- names(ensembleFeatures$x[2:which.min(bestMods$cp)])

#Define Control for all predictor selections
ctrl <- rfeControl(functions = lmFuncs,
                   method = "cv",
                   repeats = 5,
                   verbose = TRUE)
#Predictors Selection Ca
set.seed(1000)
treeBagProfile <- rfe(train[ , seq(2, 3596)],
                      train$Ca, rfeControl = ctrl)

treeBagPredictorsC <- predictors(treeBagProfile)

#-----------------------------------------------
#Predictors Selection P
set.seed(1001)
treeBagProfile <- rfe(train[ , seq(2, 3596)],
                      train$P, rfeControl = ctrl)

treeBagPredictorsP <- predictors(treeBagProfile)

#-----------------------------------------------
#Predictors Selection pH
set.seed(1002)
treeBagProfile <- rfe(train[ , seq(2, 3596)],
                      train$pH, rfeControl = ctrl)

treeBagPredictorspH <- predictors(treeBagProfile)

#-----------------------------------------------
set.seed(1003)
treeBagProfile <- rfe(train[ , seq(2, 3596)],
                      train$SOC, rfeControl = ctrl)

treeBagPredictorsSOC <- predictors(treeBagProfile)

#-----------------------------------------------
#Predictors Selection Sand
set.seed(1004)
treeBagProfile <- rfe(train[ , seq(2, 3596)],
                      train$Sand, rfeControl = ctrl)

treeBagPredictorsSand <- predictors(treeBagProfile)

##########################################################
#MODELLING
#GBM
#Hiper parameter 5-fold Cross-validation "Ca"
GBMControl <- trainControl(method = "cv",
                           number = 5,
                           verboseIter = TRUE)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 5, 2),
                       .shrinkage = c(0.001, 0.003), 
                       .n.trees = 5000)

set.seed(1005)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODClass <- train(form = Ca ~ ., 
                     data = train[randomSubset , seq(2, 3596)],
                     method = "gbm",
                     tuneGrid = gbmGrid,
                     trControl = GBMControl,
                     distribution = 'gaussian',
                     nTrain = floor(nrow(train) * 0.7),
                     verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODClass)  + theme(legend.position = "top")

#---------------------------------------------------------------
#Hiper parameter 5-fold Cross-validation "P"
GBMControl <- trainControl(method = "cv",
                           number = 5,
                           verboseIter = TRUE)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 5, 2),
                       .shrinkage = c(0.001, 0.003), 
                       .n.trees = 5000)

set.seed(1005)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODClass <- train(form = P ~ ., 
                     data = train[randomSubset , c(seq(2, 3595), 3597)],
                     method = "gbm",
                     tuneGrid = gbmGrid,
                     trControl = GBMControl,
                     distribution = 'gaussian',
                     nTrain = floor(nrow(train) * 0.7),
                     verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODClass)  + theme(legend.position = "top")

#---------------------------------------------------------------
#Hiper parameter 5-fold Cross-validation "ph"
GBMControl <- trainControl(method = "cv",
                           number = 5,
                           verboseIter = TRUE)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 5, 2),
                       .shrinkage = c(0.001, 0.003), 
                       .n.trees = 5000)

set.seed(1005)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODClass <- train(form = pH ~ ., 
                     data = train[randomSubset , c(seq(2, 3595), 3598)],
                     method = "gbm",
                     tuneGrid = gbmGrid,
                     trControl = GBMControl,
                     distribution = 'gaussian',
                     nTrain = floor(nrow(train) * 0.7),
                     verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODClass)  + theme(legend.position = "top")

#---------------------------------------------------------------
#Hiper parameter 5-fold Cross-validation "SOC"
GBMControl <- trainControl(method = "cv",
                           number = 5,
                           verboseIter = TRUE)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 5, 2),
                       .shrinkage = c(0.001, 0.003), 
                       .n.trees = 5000)

set.seed(1005)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODClass <- train(form = SOC ~ ., 
                     data = train[randomSubset , c(seq(2, 3595), 3599)],
                     method = "gbm",
                     tuneGrid = gbmGrid,
                     trControl = GBMControl,
                     distribution = 'gaussian',
                     nTrain = floor(nrow(train) * 0.7),
                     verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODClass)  + theme(legend.position = "top")

#---------------------------------------------------------------
#Hiper parameter 5-fold Cross-validation "Sand"
GBMControl <- trainControl(method = "cv",
                           number = 5,
                           verboseIter = TRUE)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 5, 2),
                       .shrinkage = c(0.001, 0.003), 
                       .n.trees = 5000)

set.seed(1005)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODClass <- train(form = Sand ~ ., 
                     data = train[randomSubset , c(seq(2, 3595), 3600)],
                     method = "gbm",
                     tuneGrid = gbmGrid,
                     trControl = GBMControl,
                     distribution = 'gaussian',
                     nTrain = floor(nrow(train) * 0.7),
                     verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODClass)  + theme(legend.position = "top")

