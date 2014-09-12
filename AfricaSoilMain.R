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
require('h2o')

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
test <- transform(test, Depth = as.factor(Depth))

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

gbmGrid <- expand.grid(.interaction.depth = seq(1, 5, 2),
                       .shrinkage = c(0.001, 0.003), 
                       .n.trees = 5000)

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
threshold <- 0.001
treesCa <- gbm.perf(gbmMODCa$finalModel, method = 'test')
treesIterated <- max(gbmGrid$.n.trees)
gbmMODExpanded <- gbmMODCa$finalModel

while(treesCa >= treesIterated - 20 & (gbmMODExpanded$test.error[treesIterated] - gbmMODExpanded$test.error[treesIterated - 100] > threshold){
  # do another 5000 iterations  
  gbmMODExpanded <- gbm.more(gbmMODExpanded, max(gbmGrid$.n.trees),
                             data = train[randomSubset , seq(2, 3596)],
                             verbose = TRUE)
  treesCa <- gbm.perf(gbmMODExpanded, method = 'test')
  treesIterated <- treesIterated + max(gbmGrid$.n.trees)
  
  if(treesIterated >= 15000){break}  
}

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
threshold <- 0.0005
treesP <- gbm.perf(gbmMODP$finalModel, method = 'test')
treesIterated <- max(gbmGrid$.n.trees)
gbmMODExpanded <- gbmMODP$finalModel

while(treesP >= treesIterated - 20 & (gbmMODExpanded$test.error[treesIterated] - gbmMODExpanded$test.error[treesIterated - 100] > threshold){
  # do another 5000 iterations  
  gbmMODExpanded <- gbm.more(gbmMODExpanded, max(gbmGrid$.n.trees),
                             data = train[randomSubset ,  c(seq(2, 3595), 3597)],
                             verbose = TRUE)
  treesP <- gbm.perf(gbmMODExpanded, method = 'test')
  treesIterated <- treesIterated + max(gbmGrid$.n.trees)
          
  if(treesIterated >= 15000){break}  
}
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
threshold <- 0.001
treespH <- gbm.perf(gbmMODph$finalModel, method = 'test')
treesIterated <- max(gbmGrid$.n.trees)
gbmMODExpanded <- gbmMODph$finalModel

while(treespH >= treesIterated - 20 & (gbmMODExpanded$test.error[treesIterated] - gbmMODExpanded$test.error[treesIterated - 100] > threshold){
  # do another 5000 iterations  
  gbmMODExpanded <- gbm.more(gbmMODExpanded, max(gbmGrid$.n.trees),
                             data = train[randomSubset , c(seq(2, 3595), 3598)],
                             verbose = TRUE)
  treespH <- gbm.perf(gbmMODExpanded, method = 'test')
  treesIterated <- treesIterated + max(gbmGrid$.n.trees)
  
  if(treesIterated >= 15000){break}  
}
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
threshold <- 0.001
treesSOC <- gbm.perf(gbmMODSOC$finalModel, method = 'test')
treesIterated <- max(gbmGrid$.n.trees)
gbmMODExpanded <- gbmMODSOC$finalModel

while(treesSOC >= treesIterated - 20 & (gbmMODExpanded$test.error[treesIterated] - gbmMODExpanded$test.error[treesIterated - 100] > threshold){
  # do another 5000 iterations  
  gbmMODExpanded <- gbm.more(gbmMODExpanded, max(gbmGrid$.n.trees),
                             data = train[randomSubset , c(seq(2, 3595), 3599)],
                             verbose = TRUE)
  treesSOC <- gbm.perf(gbmMODExpanded, method = 'test')
  treesIterated <- treesIterated + max(gbmGrid$.n.trees)
  
  if(treesIterated >= 15000){break}  
}

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
threshold <- 0.001
treesSand <- gbm.perf(gbmMODSand$finalModel, method = 'test')
treesIterated <- max(gbmGrid$.n.trees)
gbmMODExpanded <- gbmMODSand$finalModel

while(treesSand >= treesIterated - 20 & (gbmMODExpanded$test.error[treesIterated] - gbmMODExpanded$test.error[treesIterated - 100] > threshold){
  # do another 5000 iterations  
  gbmMODExpanded <- gbm.more(gbmMODExpanded, max(gbmGrid$.n.trees),
                             data = train[randomSubset , c(seq(2, 3595), 3600)],
                             verbose = TRUE)
  treesSand <- gbm.perf(gbmMODExpanded, method = 'test')
  treesIterated <- treesIterated + max(gbmGrid$.n.trees)
  
  if(treesIterated >= 15000){break}  
}

#########################################################
#Final Models using h2o GBMs 
#Create an h2o parsed data
require('h2o')
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)
africa.hex = h2o.importFile(localH2O, path = paste0(dataDirectory, 'training.csv'))

GBMModelCa <- h2o.gbm(x = seq(2, 3595),
                      y = 'Ca',
                      data = africa.hex,
                      distribution = 'gaussian', 
                      interaction.depth = gbmMODCa$bestTune[2],
                      shrinkage = gbmMODCa$bestTune[3], 
                      n.trees = treesCa)
plot(GBMModelCa)
#----------------------------------------------------------------
GBMModelP <- h2o.gbm(x = seq(2, 3595),
                     y = 'P',
                     data = africa.hex,
                     distribution = 'gaussian', 
                     interaction.depth = gbmMODP$bestTune[2],
                     shrinkage = gbmMODP$bestTune[3], 
                     n.trees = treesP)
plot(GBMModelP)
#----------------------------------------------------------------
GBMModepH <- h2o.gbm(x = seq(2, 3595),
                     y = 'pH',
                     data = africa.hex,
                     distribution = 'gaussian', 
                     interaction.depth = gbmMODpH$bestTune[2],
                     shrinkage = gbmMODpH$bestTune[3], 
                     n.trees = treespH)
plot(GBMModepH)
#----------------------------------------------------------------
GBMModeSOC <- h2o.gbm(x = seq(2, 3595),
                      y = 'SOC',
                      data = africa.hex,
                      distribution = 'gaussian', 
                      interaction.depth = gbmMODSOC$bestTune[2],
                      shrinkage = gbmMODSOC$bestTune[3], 
                      n.trees = treesSOC)
plot(GBMModeSOC)
#----------------------------------------------------------------
GBMModeSand <- h2o.gbm(x = seq(2, 3595),
                       y = 'Sand',
                       data = africa.hex,
                       distribution = 'gaussian', 
                       interaction.depth = gbmMODSand$bestTune[2],
                       shrinkage = gbmMODSandC$bestTune[3], 
                       n.trees = treesSand)
plot(GBMModeSand)
##########################################################
#PREDICTIONS
#GBM
africaTest.hex = h2o.importFile(localH2O, path = paste0(dataDirectory, 'sorted_test.csv'))
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

#h2o shutdown WARNING, All data on the server will be lost!
h2o.shutdown(localH2O, prompt = TRUE)

#########################################################
#Write .csv 
submissionTemplate$Ca <- GBMPredictionCa
submissionTemplate$Ca <- GBMPredictionP
submissionTemplate$Ca <- GBMPredictionpH
submissionTemplate$Ca <- GBMPredictionSOC
submissionTemplate$Ca <- GBMPredictionSand
write.csv(submissionTemplate, file = "PredictionI.csv", row.names = FALSE)


