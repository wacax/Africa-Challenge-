#Liberty Mutual Group - Fire Peril Loss Cost
#ver 1.1

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
require('fastICA')

#Set Working Directory
workingDirectory <- '/home/wacax/Wacax/Kaggle/Africa Tuesday/Africa Soil Property Prediction Challenge/'
setwd(workingDirectory)

dataDirectory <- '/home/wacax/Wacax/Kaggle/Africa Tuesday/Data/'

#Load external functions
source(paste0(workingDirectory, 'treeFinder.R'))
source(paste0(workingDirectory, 'gridCrossValidationh2oDeepnets.R'))

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
trainShuffled$P <- log(trainShuffled$P + 2)
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
ggplot(data = train, aes(x = Ca)) +  geom_histogram() 
#P histogram
ggplot(data = train, aes(x = P)) +  geom_histogram() 
#pH histogram
ggplot(data = train, aes(x = pH)) +  geom_histogram()
#SOC histogram
ggplot(data = train, aes(x = SOC)) +  geom_histogram()
#Sand histogram
ggplot(data = train, aes(x = Sand)) +  geom_histogram()

###################################################
#Forward linear model selection
#derpaderpa <- regsubsets(Ca ~ ., data = train[ , c(allSpectralDataNoCO2, 3596)], nvmax = 100, method = 'forward')
#bestMods <- summary(derpaderpa)
#names(bestMods)
#bestNumberOfPredictors <- which.min(bestMods$cp)
#plot(bestMods$cp, xlab="Number of Variables", ylab="CP Error")
#points(bestNumberOfPredictors, bestMods$cp[bestNumberOfPredictors],pch=20,col="red")

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
#Cross Validation Control Params
GBMControl <- trainControl(method = "cv",
                           number = 5,
                           verboseIter = TRUE)

gbmGrid <- expand.grid(.interaction.depth = c(3, 5),
                       .shrinkage = c(0.001, 0.003, 0.01), 
                       .n.trees = 2000)

#Hiper parameter 5-fold Cross-validation "Ca"
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
gbmMODP <- train(form = log(P + 2) ~ ., 
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
localH2O = h2o.init(ip = "localhost", port = 54421, max_mem_size = '4g', startH2O = TRUE)
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
africaTest.hex = h2o.importFile(localH2O, path = paste0(dataDirectory, 'testNumeric.csv'))
#Ca
GBMPredictionCa <- as.data.frame(h2o.predict(GBMModelCa, newdata = africaTest.hex))
#P
GBMPredictionP <- exp(as.data.frame(h2o.predict(GBMModelP, newdata = africaTest.hex))) - 2
#pH
GBMPredictionpH <- as.data.frame(h2o.predict(GBMModepH, newdata = africaTest.hex))
#SOC
GBMPredictionSOC <- as.data.frame(h2o.predict(GBMModeSOC, newdata = africaTest.hex))
#Sand
GBMPredictionSand <- as.data.frame(h2o.predict(GBMModeSand, newdata = africaTest.hex))
#########################################################
#h2o shutdown WARNING, All data on the server will be lost!
h2o.shutdown(localH2O, prompt = TRUE)

#########################################################
#Deep Learning with H2O
#Create an h2o parsed data
require('h2o')
localH2O = h2o.init(ip = "localhost", port = 54321, max_mem_size = '1g', startH2O = TRUE)
africa.hex = h2o.importFile(localH2O, path = paste0(dataDirectory, 'trainingShuffled.csv'))

##################################################################################
CO2Signal <- seq(which(names(africa.hex) == 'm2379.76'), which(names(africa.hex) == 'm2352.76'))

allSpectralData <- seq(2, 3579)
allSpectralDataNoCO2 <- c(seq(2, which(names(africa.hex) == 'm2379.76')), 
                          seq(which(names(africa.hex) == 'm2352.76'), 3579))

spatialPredictors <- seq(which(names(africa.hex) == 'BSAN'), which(names(africa.hex) == 'TMFI'))
depthIx <- which(names(africa.hex) == 'Depth')

#########################################################
#h2o shutdown WARNING, All data on the server will be lost!
h2o.shutdown(localH2O, prompt = FALSE)

#########################################################

#ICA
trainMatrix <- model.matrix(~ . , data = train[ , allSpectralDataNoCO2]) 
derp <- fastICA(trainMatrix, method = 'C', n.comp = 200, verbose = TRUE)
derpa <- icafast(trainMatrix, 200)

#hyperparameter search
#hyperParametersAllSpectra <- gridCrossValidationh2oDeepnets(africa.hex, predictorsCols = allSpectralData,
#                                                            noOfEpochs = 7)
#hyperParametersSpectraNoCO2 <- gridCrossValidationh2oDeepnets(africa.hex, predictorsCols = allSpectralDataNoCO2,
#                                                              noOfEpochs = 7)
#hyperParametersAllSpectraDepth <- gridCrossValidationh2oDeepnets(africa.hex, predictorsCols = c(allSpectralData, depthIx), 
#                                                                 noOfEpochs = 7)
#hyperParametersSpectraNoCO2Depth <- gridCrossValidationh2oDeepnets(africa.hex, predictorsCols = c(allSpectralDataNoCO2, depthIx),
#                                                                   noOfEpochs = 7)
#hyperParametersAllData <- gridCrossValidationh2oDeepnets(africa.hex, predictorsCols = c(allSpectralData, spatialPredictors, depthIx),
#                                                         noOfEpochs = 7)
#hyperParametersAllDataNoCO2 <- gridCrossValidationh2oDeepnets(africa.hex, predictorsCols = c(allSpectralDataNoCO2, spatialPredictors, depthIx),
#                                                              noOfEpochs = 7)

#new hyperparameter search
hyperParametersAllSpectra <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingShuffled.csv'),
                                                            predictorsCols = allSpectralData,
                                                            noOfEpochs = 6, maxMem = '13g')
hyperParametersSpectraNoCO2 <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingShuffled.csv'),
                                                              predictorsCols = allSpectralDataNoCO2,
                                                              noOfEpochs = 6, maxMem = '13g')
hyperParametersAllSpectraDepth <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingShuffled.csv'),
                                                                 predictorsCols = c(allSpectralData, depthIx), 
                                                                 noOfEpochs = 6, maxMem = '13g')
hyperParametersSpectraNoCO2Depth <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingShuffled.csv'),
                                                                   predictorsCols = c(allSpectralDataNoCO2, depthIx),
                                                                   noOfEpochs = 6, maxMem = '13g')
hyperParametersAllData <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingShuffled.csv'),
                                                         predictorsCols = c(allSpectralData, spatialPredictors, depthIx),
                                                         noOfEpochs = 6, maxMem = '13g')
hyperParametersAllDataNoCO2 <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingShuffled.csv'),
                                                              predictorsCols = c(allSpectralDataNoCO2, spatialPredictors, depthIx),
                                                              noOfEpochs = 6, maxMem = '13g')

noDropout <- c('Rectifier', 'Tanh', 'Maxout')
hidden_layers = list(c(50, 50), c(100, 100), c(50, 50, 50), c(100, 100, 100))
gridAda <- expand.grid(c(0.9, 0.95, 0.99), c(1e-12, 1e-10, 1e-8, 1e-6), stringsAsFactors = TRUE) #this creates all possible combinations
gridLs <- expand.grid(c(0, 1e-5, 1e-3), c(0, 1e-5, 1e-3), stringsAsFactors = TRUE) #this creates all possible combinations

#--------------------------------------------------
#Create an h2o parsed data
require('h2o')
localH2O = h2o.init(ip = "localhost", port = 54421, max_mem_size = '4g', startH2O = TRUE)
africa.hex = h2o.importFile(localH2O, path = paste0(dataDirectory, 'trainingShuffled.csv'))

DeepNNModelCa <- h2o.deeplearning(x = seq(2, 3579),
                                  y = hyperParameters[1],
                                  data = africa.hex,
                                  classification = FALSE, balance_classes = FALSE, 
                                  activation = hyperParameters[1, 2],
                                  hidden = hidden_layers[[as.numeric(hyperParameters[1, 3])]],
                                  adaptive_rate = TRUE,
                                  rho = gridAda[as.numeric(hyperParameters[1, 4]), 1],
                                  epsilon = gridAda[as.numeric(hyperParameters[1, 4]), 2],
                                  input_dropout_ratio = ifelse(hyperParameters[1, 2] %in% noDropout, 0, 0.1),
                                  l2 = ifelse(hyperParameters[1, 2] == 'Rectifier' | hyperParameters[1, 2] == 'Tanh', 1e-5, 0),
                                  epochs = 100, force_load_balance = TRUE)

#----------------------------------------------------------------
DeepNNGBMModelP <- h2o.deeplearning(x = seq(2, 3579),
                                    y = hyperParameters[2],
                                    data = africa.hex,
                                    classification = FALSE, balance_classes = FALSE, 
                                    activation = hyperParameters[2, 2],
                                    hidden = hidden_layers[[as.numeric(hyperParameters[2, 3])]],
                                    adaptive_rate = TRUE,
                                    rho = gridAda[as.numeric(hyperParameters[2, 4]), 1],
                                    epsilon = gridAda[as.numeric(hyperParameters[2, 4]), 2],
                                    input_dropout_ratio = ifelse(hyperParameters[2, 2] %in% noDropout, 0, 0.1),
                                    l2 = ifelse(hyperParameters[2, 2] == 'Rectifier' | hyperParameters[2, 2] == 'Tanh', 1e-5, 0),
                                    epochs = 100, force_load_balance = TRUE)
#----------------------------------------------------------------
DeepNNGBMModepH <- h2o.deeplearning(x = seq(2, 3579),
                                    y = hyperParameters[3],
                                    data = africa.hex,
                                    classification = FALSE, balance_classes = FALSE, 
                                    activation = hyperParameters[3, 2],
                                    hidden = hidden_layers[[as.numeric(hyperParameters[3, 3])]],
                                    adaptive_rate = TRUE,
                                    rho = gridAda[as.numeric(hyperParameters[3, 4]), 1],
                                    epsilon = gridAda[as.numeric(hyperParameters[3, 4]), 2],
                                    input_dropout_ratio = ifelse(hyperParameters[3, 2] %in% noDropout, 0, 0.1),
                                    l2 = ifelse(hyperParameters[3, 2] == 'Rectifier' | hyperParameters[3, 2] == 'Tanh', 1e-5, 0),
                                    epochs = 100, force_load_balance = TRUE)
#----------------------------------------------------------------
DeepNNGBMModeSOC <- h2o.deeplearning(x = seq(2, 3579),
                                     y = hyperParameters[4],
                                     data = africa.hex,
                                     classification = FALSE, balance_classes = FALSE, 
                                     activation = hyperParameters[4, 2],
                                     hidden = hidden_layers[[as.numeric(hyperParameters[4, 3])]],
                                     adaptive_rate = TRUE,
                                     rho = gridAda[as.numeric(hyperParameters[4, 4]), 1],
                                     epsilon = gridAda[as.numeric(hyperParameters[4, 4]), 2],
                                     input_dropout_ratio = ifelse(hyperParameters[4, 2] %in% noDropout, 0, 0.1),
                                     l2 = ifelse(hyperParameters[4, 2] == 'Rectifier' | hyperParameters[4, 2] == 'Tanh', 1e-5, 0),
                                     epochs = 100, force_load_balance = TRUE)
#----------------------------------------------------------------
DeepNNGBMModeSand <- h2o.deeplearning(x = seq(2, 3579),
                                      y = hyperParameters[5],
                                      data = africa.hex,
                                      classification = FALSE, balance_classes = FALSE, 
                                      activation = hyperParameters[5, 2],
                                      hidden = hidden_layers[[as.numeric(hyperParameters[5, 3])]],
                                      adaptive_rate = TRUE,
                                      rho = gridAda[as.numeric(hyperParameters[5, 4]), 1],
                                      epsilon = gridAda[as.numeric(hyperParameters[5, 4]), 2],
                                      input_dropout_ratio = ifelse(hyperParameters[5, 2] %in% noDropout, 0, 0.1),
                                      l2 = ifelse(hyperParameters[5, 2] == 'Rectifier' | hyperParameters[5, 2] == 'Tanh', 1e-5, 0),
                                      epochs = 100, force_load_balance = TRUE)

##########################################################
#PREDICTIONS
africaTest.hex = h2o.importFile(localH2O, path = paste0(dataDirectory, 'testNumeric.csv'))
#Ca
NNPredictionCa <- as.data.frame(h2o.predict(DeepNNModelCa, newdata = africaTest.hex))
#P
NNPredictionP <- exp(as.data.frame(h2o.predict(DeepNNGBMModelP, newdata = africaTest.hex))) - 2
#pH
NNPredictionpH <- as.data.frame(h2o.predict(DeepNNGBMModepH, newdata = africaTest.hex))
#SOC
NNPredictionSOC <- as.data.frame(h2o.predict(DeepNNGBMModeSOC, newdata = africaTest.hex))
#Sand
NNPredictionSand <- as.data.frame(h2o.predict(DeepNNGBMModeSand, newdata = africaTest.hex))

#########################################################
#h2o shutdown WARNING, All data on the server will be lost!
h2o.shutdown(localH2O, prompt = TRUE)

#########################################################
#Write .csv 
#GBM
submissionTemplate$Ca <- unlist(GBMPredictionCa)
submissionTemplate$P <- exp(unlist(GBMPredictionP)) - 2
submissionTemplate$pH <- unlist(GBMPredictionpH)
submissionTemplate$SOC <- unlist(GBMPredictionSOC)
submissionTemplate$Sand <- unlist(GBMPredictionSand)
write.csv(submissionTemplate, file = "PredictionII.csv", row.names = FALSE)
#Deep NN
submissionTemplate$Ca <- unlist(NNPredictionCa)
submissionTemplate$P <- exp(unlist(NNPredictionP)) - 2
submissionTemplate$pH <- unlist(NNPredictionpH)
submissionTemplate$SOC <- unlist(NNPredictionSOC)
submissionTemplate$Sand <- unlist(NNPredictionSand)
write.csv(submissionTemplate, file = "Prediction2_3579.csv", row.names = FALSE)
