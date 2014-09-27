#Liberty Mutual Group - Fire Peril Loss Cost
#ver 0.1

#########################
#Init
rm(list=ls(all=TRUE))

#Load/install required libraries
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
source(paste0(workingDirectory, 'treeFinder.R'))

#############################
#Load Data
#Input Data
train <- read.csv(paste0(dataDirectory, 'training.csv'), header = TRUE, stringsAsFactors = FALSE)
test <- read.csv(paste0(dataDirectory, 'sorted_test.csv'), header = TRUE, stringsAsFactors = FALSE)

submissionTemplate <- read.csv(paste0(dataDirectory, 'sample_submission.csv'), header = TRUE, stringsAsFactors = FALSE)

################################
#Data Processing
#Data Transformation
train <- transform(train, Depth = as.numeric(as.factor(train$Depth)) - 1)
test <- transform(test, Depth = as.numeric(as.factor(test$Depth)) - 1)

#Transform it to H2O compatible file
#Make a Shuffled CSV to train h2o models
set.seed(1010)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
trainShuffled <- train[randomSubset, ]
write.csv(trainShuffled, file = paste0(dataDirectory, 'trainingShuffled.csv'), row.names = FALSE)

#Test Data
write.csv(test, file = paste0(dataDirectory, 'testNumeric.csv'), row.names = FALSE)

################################
#EDA
#Rows containing NAs
noNAIndices <- which(apply(is.na(train), 1, sum) == 0)

#Principal Components Analysis using h20
#Merge Both train and test to perfom PCA on
write.csv(rbind(train[,2:3595], test[,2:3595]), file = paste0(dataDirectory, 'fullDataset.csv'))

#Run PCA
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)
full.hex = h2o.importFile(localH2O, path = paste0(dataDirectory, 'fullDataset.csv'))

PCAFull <- h2o.prcomp(full.hex)

#h2o shutdown WARNING, All data on the server will be lost!
h2o.shutdown(localH2O, prompt = TRUE)
detach('package:h2o', unload = TRUE)   #h20 will mask functions in the caret / gbm xvalidation 

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

gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, 2),
                       .shrinkage = c(0.001, 0.003, 0.01), 
                       .n.trees = 2000)

set.seed(1005)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODCa <- train(form = Ca~., 
                  data = train[randomSubset , seq(2, 3596)],
                  method = "gbm",
                  tuneGrid = gbmGrid,
                  trControl = GBMControl,
                  distribution = 'gaussian',
                  nTrain = floor(nrow(train) * 0.7),
                  verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODCa)  + theme(legend.position = "top")

#Find optimal number of trees
treesCa <- treeFinder(gbmMODCa$finalModel, dataNew = train[randomSubset , seq(2, 3596)])

#---------------------------------------------------------------
#Hiper parameter 5-fold Cross-validation "P"
set.seed(1006)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODP <- train(form = P ~ ., 
                 data = train[randomSubset , c(seq(2, 3595), 3597)],
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 trControl = GBMControl,
                 distribution = 'gaussian',
                 nTrain = floor(nrow(train) * 0.7),
                 verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODP)  + theme(legend.position = "top")

#Find optimal number of trees
treesP <- treeFinder(gbmMODP$finalModel, dataNew = train[randomSubset , c(seq(2, 3595), 3597)])

#---------------------------------------------------------------
#Hiper parameter 5-fold Cross-validation "ph"
set.seed(1007)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODph <- train(form = pH ~ ., 
                  data = train[randomSubset , c(seq(2, 3595), 3598)],
                  method = "gbm",
                  tuneGrid = gbmGrid,
                  trControl = GBMControl,
                  distribution = 'gaussian',
                  nTrain = floor(nrow(train) * 0.7),
                  verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODph)  + theme(legend.position = "top")

#Find optimal number of trees
treesph <- treeFinder(gbmMODph$finalModel, dataNew = train[randomSubset , c(seq(2, 3595), 3598)])

#---------------------------------------------------------------
#Hiper parameter 5-fold Cross-validation "SOC"
set.seed(1008)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODSOC <- train(form = SOC ~ ., 
                   data = train[randomSubset , c(seq(2, 3595), 3599)],
                   method = "gbm",
                   tuneGrid = gbmGrid,
                   trControl = GBMControl,
                   distribution = 'gaussian',
                   nTrain = floor(nrow(train) * 0.7),
                   verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODSOC)  + theme(legend.position = "top")

#Find optimal number of trees
treesSOC <- treeFinder(gbmMODSOC$finalModel, dataNew = train[randomSubset , c(seq(2, 3595), 3599)])

#---------------------------------------------------------------
#Hiper parameter 5-fold Cross-validation "Sand"
set.seed(1009)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODSand <- train(form = Sand ~ ., 
                    data = train[randomSubset , c(seq(2, 3595), 3600)],
                    method = "gbm",
                    tuneGrid = gbmGrid,
                    trControl = GBMControl,
                    distribution = 'gaussian',
                    nTrain = floor(nrow(train) * 0.7),
                    verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODSand)  + theme(legend.position = "top")

#Find optimal number of trees
treesSand <- treeFinder(gbmMODSand$finalModel, dataNew = train[randomSubset , c(seq(2, 3595), 3600)])

#########################################################
#Final Models using h2o GBMs 
#Create an h2o parsed data
require('h2o')
localH2O = h2o.init(ip = "localhost", port = 54321, Xmx = '4g', startH2O = TRUE)
africa.hex = h2o.importFile(localH2O, path = paste0(dataDirectory, 'trainingShuffled.csv'))

#data frames as h2o / not necesary h2o is reading directly from .csv files
#trainh2o <- as.h2o(localH2O, train, key = 'PIDN')
#testh2o <- as.h2o(localH2O, test, key = 'PIDN')

GBMModelCa <- h2o.gbm(x = seq(2, 3595),
                      y = 'Ca',
                      data = africa.hex,
                      distribution = 'gaussian', 
                      interaction.depth = as.numeric(gbmMODCa$bestTune[2]),
                      shrinkage = as.numeric(gbmMODCa$bestTune[3]), 
                      n.trees = treesCa)
plot(GBMModelCa)
#----------------------------------------------------------------
GBMModelP <- h2o.gbm(x = seq(2, 3595),
                     y = 'P',
                     data = africa.hex,
                     distribution = 'gaussian', 
                     interaction.depth =as.numeric(gbmMODP$bestTune[2]),
                     shrinkage = as.numeric(gbmMODP$bestTune[3]), 
                     n.trees = treesP)
plot(GBMModelP)
#----------------------------------------------------------------
GBMModepH <- h2o.gbm(x = seq(2, 3595),
                     y = 'pH',
                     data = africa.hex,
                     distribution = 'gaussian', 
                     interaction.depth = as.numeric(gbmMODph$bestTune[2]),
                     shrinkage = as.numeric(gbmMODph$bestTune[3]), 
                     n.trees = treesph)
plot(GBMModepH)
#----------------------------------------------------------------
GBMModeSOC <- h2o.gbm(x = seq(2, 3595),
                      y = 'SOC',
                      data = africa.hex,
                      distribution = 'gaussian', 
                      interaction.depth = as.numeric(gbmMODSOC$bestTune[2]),
                      shrinkage = as.numeric(gbmMODSOC$bestTune[3]), 
                      n.trees = treesSOC)
plot(GBMModeSOC)
#----------------------------------------------------------------
GBMModeSand <- h2o.gbm(x = seq(2, 3595),
                       y = 'Sand',
                       data = africa.hex,
                       distribution = 'gaussian', 
                       interaction.depth = as.numeric(gbmMODSand$bestTune[2]),
                       shrinkage = as.numeric(gbmMODSand$bestTune[3]), 
                       n.trees = treesSand)
plot(GBMModeSand)
##########################################################
#PREDICTIONS
#GBM
africaTest.hex = h2o.importFile(localH2O, path = paste0(dataDirectory, 'testNumeric.csv'))
#Ca
GBMPredictionCa <- as.data.frame(h2o.predict(GBMModelCa, newdata = africaTest.hex))
#P
GBMPredictionP <- as.data.frame(h2o.predict(GBMModelP, newdata = africaTest.hex))
#pH
GBMPredictionpH <- as.data.frame(h2o.predict(GBMModepH, newdata = africaTest.hex))
#SOC
GBMPredictionSOC <- as.data.frame(h2o.predict(GBMModeSOC, newdata = africaTest.hex))
#Sand
GBMPredictionSand <- as.data.frame(h2o.predict(GBMModeSand, newdata = africaTest.hex))

#########################################################
#Deep Learning with H2O
#Simple Validation (80/20 data split)
africaHexTrain <- africa.hex[1:floor(dim(africa.hex)[1] * 0.6), ]
africaHexValid <- africa.hex[floor(dim(africa.hex)[1] * 0.6) + 1:dim(africa.hex)[1], ]

#run 10 epochs 
#Run the grid search
gridSearch1st <- h2o.deeplearning(x = seq(2, 3595),
                                  y = 'Ca',
                                  data = africaHexTrain,
                                  validation = africaHexValid,
                                  classification = FALSE, balance_classes = FALSE, 
                                  activation = 'RectifierWithDropout',
                                  hidden = c(20, 20),
                                  hidden_dropout_ratios = c(0.1, 0.1),
                                  epochs = 5)
gridSearch1st <- h2o.deeplearning(x = seq(2, 3595),
                                  y = 'Ca',
                                  data = africaHexTrain,
                                  validation = africaHexValid,
                                  classification = FALSE, balance_classes = FALSE, 
                                  activation = 'TanhWithDropout',
                                  hidden = c(20, 20),
                                  hidden_dropout_ratios = c(0.1, 0.1),
                                  epochs = 5)
#Create a set of network topologies
hidden_layers = list(c(200,200), c(100,300,100),c(500,500,500))

#--------------------------------------------------
DeepNNModelCa <- h2o.deeplearning(x = seq(2, 3595),
                                  y = 'Ca',
                                  data = africa.hex,
                                  classification = FALSE, balance_classes = FALSE, 
                                  activation = 'MaxoutWithDropout',
                                  hidden = c(100, 100),
                                  hidden_dropout_ratios = c(0.5,0.5),
                                  epochs = 100)
#----------------------------------------------------------------
DeepNNGBMModelP <- h2o.deeplearning(x = seq(2, 3595),
                                    y = 'P',
                                    data = africa.hex,
                                    classification = FALSE, balance_classes = FALSE, 
                                    activation = 'MaxoutWithDropout',
                                    hidden = c(100, 100),
                                    hidden_dropout_ratios = c(0.5,0.5,0.5),
                                    epochs = 100)
#----------------------------------------------------------------
DeepNNGBMModepH <- h2o.deeplearning(x = seq(2, 3595),
                                    y = 'pH',
                                    data = africa.hex,
                                    classification = FALSE, balance_classes = FALSE, 
                                    activation = 'MaxoutWithDropout',
                                    hidden = c(100, 100),
                                    hidden_dropout_ratios = c(0.5,0.5,0.5),
                                    epochs = 100)
#----------------------------------------------------------------
DeepNNGBMModeSOC <- h2o.deeplearning(x = seq(2, 3595),
                                     y = 'SOC',
                                     data = africa.hex,
                                     classification = FALSE, balance_classes = FALSE, 
                                     activation = 'MaxoutWithDropout',
                                     hidden = c(100, 100),
                                     hidden_dropout_ratios = c(0.5,0.5,0.5),
                                     epochs = 100)
#----------------------------------------------------------------
DeepNNGBMModeSand <- h2o.deeplearning(x = seq(2, 3595),
                                      y = 'Sand',
                                      data = africa.hex,
                                      classification = FALSE, balance_classes = FALSE, 
                                      activation = 'MaxoutWithDropout',
                                      hidden = c(100, 100),
                                      hidden_dropout_ratios = c(0.5,0.5,0.5),
                                      epochs = 100)

#h2o shutdown WARNING, All data on the server will be lost!
h2o.shutdown(localH2O, prompt = TRUE)

#########################################################
#Write .csv 
submissionTemplate$Ca <- unlist(GBMPredictionCa)
submissionTemplate$P <- unlist(GBMPredictionP)
submissionTemplate$pH <- unlist(GBMPredictionpH)
submissionTemplate$SOC <- unlist(GBMPredictionSOC)
submissionTemplate$Sand <- unlist(GBMPredictionSand)
write.csv(submissionTemplate, file = "PredictionI.csv", row.names = FALSE)


