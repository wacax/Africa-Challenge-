#Liberty Mutual Group - Fire Peril Loss Cost
#ver 1.1

#########################
#Init
rm(list=ls(all=TRUE))

#Load/install required libraries
require('ggplot2')
require('reshape2')
require('stringr')
require('grid')
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
source(paste0(workingDirectory, 'multiplot.R'))
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

#Signal Processing
#Spectra / CO2 and others
CO2Signal <- seq(which(names(train) == 'm2379.76'), which(names(train) == 'm2352.76'))

allSpectralData <- seq(2, 3579)
allSpectralDataNoCO2 <- c(seq(2, which(names(train) == 'm2379.76')), 
                          seq(which(names(train) == 'm2352.76'), 3579))

spatialPredictors <- seq(which(names(train) == 'BSAN'), which(names(train) == 'TMFI'))
depthIx <- which(names(train) == 'Depth')

#Plotting Raw signals
plotSpectra <- function(numberOfSamples, spectralData, subsample){
  #based on http://www.kaggle.com/c/afsis-soil-properties/forums/t/10184/first-derivative
  ixs <- sample(which(1:nrow(train) %in% subsample), numberOfSamples)
  trainRawSub <- melt(train[ixs, ], id.vars = "PIDN", measure.vars = spectralData)
  trainRawSub$variable <- as.numeric(str_replace_all(trainRawSub$variable,"m",""))
  ggplot(trainRawSub, aes(x = variable, y = value, colour = PIDN)) + geom_line()
}

#Subsoil all spectra
allSpectraSubsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(train$Depth == 0))
#Topsoil all spectra
allSpectraTopsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(train$Depth == 1))
#Subsoil no CO2
SpectralDataNoCO2Subsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2, subsample = which(train$Depth == 0))
#Topsoil no CO2
SpectralDataNoCO2Topsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2, subsample = which(train$Depth == 1))

multiplot(allSpectraSubsoil, allSpectraTopsoil, SpectralDataNoCO2Subsoil, SpectralDataNoCO2Topsoil, cols = 2)

################################
#EDA

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
localH2O = h2o.init(ip = "localhost", port = 54321, max_mem_size = '4g', startH2O = TRUE)
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
h2o.shutdown(localH2O, prompt = FALSE)

#.csv Creation
#GBM
submissionTemplate$Ca <- unlist(GBMPredictionCa)
submissionTemplate$P <- exp(unlist(GBMPredictionP)) - 2
submissionTemplate$pH <- unlist(GBMPredictionpH)
submissionTemplate$SOC <- unlist(GBMPredictionSOC)
submissionTemplate$Sand <- unlist(GBMPredictionSand)
write.csv(submissionTemplate, file = "PredictionGBMI.csv", row.names = FALSE)

#########################################################
#Deep Learning with H2O
#Create an h2o parsed data
require('h2o')
localH2O <- h2o.init(ip = "localhost", port = 54321, max_mem_size = '1g', startH2O = TRUE)
africa.hex <- h2o.importFile(localH2O, path = paste0(dataDirectory, 'trainingShuffled.csv'))

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

#new hyperparameter search
hyperParametersAllSpectra <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingShuffled.csv'),
                                                            predictorsCols = allSpectralData,
                                                            noOfEpochs = 6, maxMem = '5g')
hyperParametersSpectraNoCO2 <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingShuffled.csv'),
                                                              predictorsCols = allSpectralDataNoCO2,
                                                              noOfEpochs = 6, maxMem = '5g')
hyperParametersAllSpectraDepth <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingShuffled.csv'),
                                                                 predictorsCols = c(allSpectralData, depthIx), 
                                                                 noOfEpochs = 6, maxMem = '5g')
hyperParametersSpectraNoCO2Depth <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingShuffled.csv'),
                                                                   predictorsCols = c(allSpectralDataNoCO2, depthIx),
                                                                   noOfEpochs = 6, maxMem = '5g')
hyperParametersAllData <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingShuffled.csv'),
                                                         predictorsCols = c(allSpectralData, spatialPredictors, depthIx),
                                                         noOfEpochs = 6, maxMem = '5g')
hyperParametersAllDataNoCO2 <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingShuffled.csv'),
                                                              predictorsCols = c(allSpectralDataNoCO2, spatialPredictors, depthIx),
                                                              noOfEpochs = 6, maxMem = '5g')

noDropout <- c('Rectifier', 'Tanh', 'Maxout')
hidden_layers = list(c(50, 50), c(100, 100), c(50, 50, 50), c(100, 100, 100))
gridAda <- expand.grid(c(0.9, 0.95, 0.99), c(1e-12, 1e-10, 1e-8, 1e-6), stringsAsFactors = TRUE) #this creates all possible combinations
gridLs <- expand.grid(c(0, 1e-5, 1e-3), c(0, 1e-5, 1e-3), stringsAsFactors = TRUE) #this creates all possible combinations

#--------------------------------------------------
#Create an h2o parsed data
require('h2o')
localH2O <- h2o.init(ip = "localhost", port = 54321, max_mem_size = '5g', startH2O = TRUE)
africa.hex <- h2o.importFile(localH2O, path = paste0(dataDirectory, 'trainingShuffled.csv'))
#Test Data
africaTest.hex = h2o.importFile(localH2O, path = paste0(dataDirectory, 'testNumeric.csv'))

DeepNNModelCa <- h2o.deeplearning(x = c(allSpectralDataNoCO2, spatialPredictors, depthIx),
                                  y = hyperParametersAllDataNoCO2[[1]][1, 'predCol'],
                                  data = africa.hex,
                                  classification = FALSE, balance_classes = FALSE, 
                                  activation = hyperParametersAllDataNoCO2[[1]][1, 2],
                                  hidden = hidden_layers[[as.numeric(hyperParametersAllDataNoCO2[[1]][1, 3])]],
                                  adaptive_rate = TRUE,
                                  rho = gridAda[as.numeric(hyperParametersAllDataNoCO2[[1]][1, 4]), 1],
                                  epsilon = gridAda[as.numeric(hyperParametersAllDataNoCO2[[1]][1, 4]), 2],
                                  input_dropout_ratio = 0,
                                  l1 = gridLs[as.numeric(hyperParametersAllDataNoCO2[[1]][1, 5]), 1],
                                  l2 = gridLs[as.numeric(hyperParametersAllDataNoCO2[[1]][1, 5]), 2],
                                  epochs = 200)
#Prediction
NNPredictionCa <- as.data.frame(h2o.predict(DeepNNModelCa, newdata = africaTest.hex))
#Clean data in server
h2o.rm(object = localH2O, keys = h2o.ls(localH2O)$Key[1])

#----------------------------------------------------------------
DeepNNGBMModelP <- h2o.deeplearning(x = c(allSpectralData, spatialPredictors, depthIx),
                                    y = hyperParametersAllData[[1]][2, 'predCol'],
                                    data = africa.hex,
                                    classification = FALSE, balance_classes = FALSE, 
                                    activation = hyperParametersAllData[[1]][2, 2],
                                    hidden = hidden_layers[[as.numeric(hyperParametersAllData[[1]][2, 3])]],
                                    adaptive_rate = TRUE,
                                    rho = gridAda[as.numeric(hyperParametersAllData[[1]][2, 4]), 1],
                                    epsilon = gridAda[as.numeric(hyperParametersAllData[[1]][2, 4]), 2],
                                    input_dropout_ratio = 0,
                                    l1 = gridLs[as.numeric(hyperParametersAllData[[1]][1, 5]), 1],
                                    l2 = gridLs[as.numeric(hyperParametersAllData[[1]][1, 5]), 2],
                                    epochs = 200)
#Prediction
NNPredictionP <- exp(as.data.frame(h2o.predict(DeepNNGBMModelP, newdata = africaTest.hex))) - 2
#Clean data in server
h2o.rm(object = localH2O, keys = h2o.ls(localH2O)$Key[1])

#----------------------------------------------------------------
DeepNNGBMModepH <- h2o.deeplearning(x = c(allSpectralDataNoCO2, depthIx),
                                    y = hyperParametersSpectraNoCO2Depth[[1]][3, 'predCol'],
                                    data = africa.hex,
                                    classification = FALSE, balance_classes = FALSE, 
                                    activation = hyperParametersSpectraNoCO2Depth[[1]][3, 2],
                                    hidden = hidden_layers[[as.numeric(hyperParametersSpectraNoCO2Depth[[1]][3, 3])]],
                                    adaptive_rate = TRUE,
                                    rho = gridAda[as.numeric(hyperParametersSpectraNoCO2Depth[[1]][3, 4]), 1],
                                    epsilon = gridAda[as.numeric(hyperParametersSpectraNoCO2Depth[[1]][3, 4]), 2],
                                    input_dropout_ratio = 0,
                                    l1 = gridLs[as.numeric(hyperParametersSpectraNoCO2Depth[[1]][1, 5]), 1],
                                    l2 = gridLs[as.numeric(hyperParametersSpectraNoCO2Depth[[1]][1, 5]), 2],
                                    epochs = 200)
#Prediction
NNPredictionpH <- as.data.frame(h2o.predict(DeepNNGBMModepH, newdata = africaTest.hex))
#Clean data in server
h2o.rm(object = localH2O, keys = h2o.ls(localH2O)$Key[1])

#----------------------------------------------------------------
DeepNNGBMModeSOC <- h2o.deeplearning(x = c(allSpectralData, spatialPredictors, depthIx),
                                     y = hyperParametersAllData[[1]][4, 'predCol'],
                                     data = africa.hex,
                                     classification = FALSE, balance_classes = FALSE, 
                                     activation = hyperParametersAllData[[1]][4, 2],
                                     hidden = hidden_layers[[as.numeric(hyperParametersAllData[[1]][4, 3])]],
                                     adaptive_rate = TRUE,
                                     rho = gridAda[as.numeric(hyperParametersAllData[[1]][4, 4]), 1],
                                     epsilon = gridAda[as.numeric(hyperParametersAllData[[1]][4, 4]), 2],
                                     input_dropout_ratio = 0,
                                     l1 = gridLs[as.numeric(hyperParametersAllData[[1]][1, 5]), 1],
                                     l2 = gridLs[as.numeric(hyperParametersAllData[[1]][1, 5]), 2],
                                     epochs = 200)
#Prediction
NNPredictionSOC <- as.data.frame(h2o.predict(DeepNNGBMModeSOC, newdata = africaTest.hex))
#Clean data in server
h2o.rm(object = localH2O, keys = h2o.ls(localH2O)$Key[1])

#----------------------------------------------------------------
DeepNNGBMModeSand <- h2o.deeplearning(x = c(allSpectralDataNoCO2, spatialPredictors, depthIx),
                                      y = hyperParametersAllDataNoCO2[[1]][5, 'predCol'],
                                      data = africa.hex,
                                      classification = FALSE, balance_classes = FALSE, 
                                      activation = hyperParametersAllDataNoCO2[[1]][5, 2],
                                      hidden = hidden_layers[[as.numeric(hyperParametersAllDataNoCO2[[1]][5, 3])]],
                                      adaptive_rate = TRUE,
                                      rho = gridAda[as.numeric(hyperParametersAllDataNoCO2[[1]][5, 4]), 1],
                                      epsilon = gridAda[as.numeric(hyperParametersAllDataNoCO2[[1]][5, 4]), 2],
                                      input_dropout_ratio = 0,
                                      l1 = gridLs[as.numeric(hyperParametersAllDataNoCO2[[1]][1, 5]), 1],
                                      l2 = gridLs[as.numeric(hyperParametersAllDataNoCO2[[1]][1, 5]), 2],
                                      epochs = 200)
#Prediction
NNPredictionSand <- as.data.frame(h2o.predict(DeepNNGBMModeSand, newdata = africaTest.hex))
#Clean data in server
h2o.rm(object = localH2O, keys = h2o.ls(localH2O)$Key[1])

#########################################################
#h2o shutdown WARNING, All data on the server will be lost!
h2o.shutdown(localH2O, prompt = FALSE)

#########################################################
#Write .csv 
#Deep NN
submissionTemplate$Ca <- unlist(NNPredictionCa)
submissionTemplate$P <- exp(unlist(NNPredictionP)) - 2
submissionTemplate$pH <- unlist(NNPredictionpH)
submissionTemplate$SOC <- unlist(NNPredictionSOC)
submissionTemplate$Sand <- unlist(NNPredictionSand)
write.csv(submissionTemplate, file = "PredictionDeepNNI.csv", row.names = FALSE)
