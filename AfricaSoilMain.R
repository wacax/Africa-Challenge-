#Liberty Mutual Group - Fire Peril Loss Cost
#ver 1.2

#########################
#Init
rm(list=ls(all=TRUE))

#Load/install required libraries
require('ggplot2')
require('reshape2')
require('stringr')
require('grid')
require('prospectr')
require('pls')
require('ptw')
require('leaps')
require('caret')
require('glmnet')
require('parallel')
require('doParallel')

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
CO2SignalOR <- seq(which(names(train) == 'm2379.76'), which(names(train) == 'm2352.76'))
allSpectralDataOR <- seq(which(names(train) == 'm7497.96'), which(names(train) == 'm599.76'))
allSpectralDataNoCO2OR <- c(seq(which(names(train) == 'm7497.96'), which(names(train) == 'm2379.76')), 
                            seq(which(names(train) == 'm2352.76'), which(names(train) == 'm599.76')))
spatialPredictorsOR <- seq(which(names(train) == 'BSAN'), which(names(train) == 'TMFI'))
depthIxOR <- which(names(train) == 'Depth')

#Plotting signals
plotSpectra <- function(numberOfSamples, spectralData, subsample, dataDF){
  #based on http://www.kaggle.com/c/afsis-soil-properties/forums/t/10184/first-derivative
  ixs <- sample(which(1:nrow(dataDF) %in% subsample), numberOfSamples)
  trainRawSub <- melt(dataDF[ixs, ], id.vars = "PIDN", measure.vars = spectralData)
  trainRawSub$variable <- as.numeric(str_replace_all(trainRawSub$variable,"m",""))
  ggplot(trainRawSub, aes(x = variable, y = value, colour = PIDN)) + geom_line()
}

#Subsoil all spectra
allSpectraSubsoil <- plotSpectra(10, spectralData = allSpectralDataOR, subsample = which(train$Depth == 0), train)
#Topsoil all spectra
allSpectraTopsoil <- plotSpectra(10, spectralData = allSpectralDataOR, subsample = which(train$Depth == 1), train)
#Subsoil no CO2
SpectralDataNoCO2Subsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2OR, subsample = which(train$Depth == 0), train)
#Topsoil no CO2
SpectralDataNoCO2Topsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2OR, subsample = which(train$Depth == 1), train)

multiplot(allSpectraSubsoil, allSpectraTopsoil, SpectralDataNoCO2Subsoil, SpectralDataNoCO2Topsoil, cols = 2)

#--------------------------------
#Transformations
#Smoothing
#Binning
#w/ 11 bins
trainBin11 <- binning(train[ , allSpectralDataOR], bin.size=10)
trainBin11 <- cbind(train[ , 'PIDN'], trainBin11, train[ , spatialPredictorsOR], train$Depth)
names(trainBin11)[1] <- 'PIDN'
names(trainBin11)[length(trainBin11)] <- 'Depth'
#Spectra / CO2 and others
allSpectralData <- seq(2, length(trainBin11) - 16)
spatialPredictors <- seq(which(names(trainBin11) == 'BSAN'), which(names(trainBin11) == 'TMFI'))
depthIx <- which(names(trainBin11) == 'Depth')#
#Subsoil all spectra
bin11SmootherallSpectraSubsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainBin11$Depth == 0), trainBin11)
#Topsoil all spectra
bin11SmootherallSpectraTopsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainBin11$Depth == 1), trainBin11)

multiplot(bin11SmootherallSpectraSubsoil, bin11SmootherallSpectraTopsoil, cols = 2)

#w/ 50 bins
trainBin50 <- binning(train[ , allSpectralDataOR], bin.size=50)
trainBin50 <- cbind(train[ , 'PIDN'], trainBin50, train[ , spatialPredictorsOR], train$Depth)
names(trainBin50)[1] <- 'PIDN'
names(trainBin50)[length(trainBin50)] <- 'Depth'
#Spectra / CO2 and others
allSpectralData <- seq(2, length(trainBin50) - 16)
spatialPredictors <- seq(which(names(trainBin50) == 'BSAN'), which(names(trainBin50) == 'TMFI'))
depthIx <- which(names(trainBin50) == 'Depth')
#Subsoil all spectra
bin50SmootherallSpectraSubsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainBin50$Depth == 0), trainBin50)
#Topsoil all spectra
bin50SmootherallSpectraTopsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainBin50$Depth == 1), trainBin50)

multiplot(bin50SmootherallSpectraSubsoil, bin50SmootherallSpectraSubsoil, cols = 2)

#Running Mean Smoother
trainMeanSmoother <- as.data.frame(movav(train[ , allSpectralDataOR], w = 11))
trainMeanSmoother <- cbind(train[ , 'PIDN'], trainMeanSmoother, train[ , spatialPredictorsOR], train$Depth)
names(trainMeanSmoother)[1] <- 'PIDN'
names(trainMeanSmoother)[length(trainMeanSmoother)] <- 'Depth'

#Spectra / CO2 and others
CO2Signal <- seq(which(names(trainMeanSmoother) == 'm2379.76'), which(names(trainMeanSmoother) == 'm2352.76'))
allSpectralData <- seq(2, length(trainMeanSmoother) - 16)
allSpectralDataNoCO2 <- c(seq(2, which(names(trainMeanSmoother) == 'm2379.76')), 
                          seq(which(names(trainMeanSmoother) == 'm2352.76'), length(trainMeanSmoother) - 16))
spatialPredictors <- seq(which(names(trainMeanSmoother) == 'BSAN'), which(names(trainMeanSmoother) == 'TMFI'))
depthIx <- which(names(trainMeanSmoother) == 'Depth')

#Subsoil all spectra
meanSmootherallSpectraSubsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainMeanSmoother$Depth == 0), trainMeanSmoother)
#Topsoil all spectra
meanSmootherallSpectraTopsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainMeanSmoother$Depth == 1), trainMeanSmoother)
#Subsoil no CO2
meanSmootherSpectralDataNoCO2Subsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2, subsample = which(trainMeanSmoother$Depth == 0), trainMeanSmoother)
#Topsoil no CO2
meanSmootherSpectralDataNoCO2Topsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2, subsample = which(trainMeanSmoother$Depth == 1), trainMeanSmoother)

multiplot(meanSmootherallSpectraSubsoil, meanSmootherallSpectraTopsoil,
          meanSmootherSpectralDataNoCO2Subsoil, meanSmootherSpectralDataNoCO2Topsoil, cols = 2)

#Running Polinomial Smoother (Savitzky-Golay filtering)
trainSGSmoother <- as.data.frame(savitzkyGolay(train[ , allSpectralDataOR], p = 3, w = 11, m = 0))
trainSGSmoother <- cbind(train[ , 'PIDN'], trainSGSmoother, train[ , spatialPredictorsOR], train$Depth)
names(trainSGSmoother)[1] <- 'PIDN'
names(trainSGSmoother)[length(trainSGSmoother)] <- 'Depth'

#Spectra / CO2 and others
CO2Signal <- seq(which(names(trainSGSmoother) == 'm2379.76'), which(names(trainSGSmoother) == 'm2352.76'))
allSpectralData <- seq(2, length(trainSGSmoother) - 16)
allSpectralDataNoCO2 <- c(seq(2, which(names(trainSGSmoother) == 'm2379.76')), 
                          seq(which(names(trainSGSmoother) == 'm2352.76'), length(trainSGSmoother) - 16))
spatialPredictors <- seq(which(names(trainSGSmoother) == 'BSAN'), which(names(trainSGSmoother) == 'TMFI'))
depthIx <- which(names(trainSGSmoother) == 'Depth')

#Subsoil all spectra
SGSmootherallSpectraSubsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainSGSmoother$Depth == 0), trainSGSmoother)
#Topsoil all spectra
SGSmootherallSpectraTopsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainSGSmoother$Depth == 1), trainSGSmoother)
#Subsoil no CO2
SGSmootherSpectralDataNoCO2Subsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2, subsample = which(trainSGSmoother$Depth == 0), trainSGSmoother)
#Topsoil no CO2
SGSmootherSpectralDataNoCO2Topsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2, subsample = which(trainSGSmoother$Depth == 1), trainSGSmoother)

multiplot(SGSmootherallSpectraSubsoil, SGSmootherallSpectraTopsoil,
          SGSmootherSpectralDataNoCO2Subsoil, SGSmootherSpectralDataNoCO2Topsoil, cols = 2)

#-----------------------------------------------
#Baseline Correction
#Savitzky - Golay and Gorry with second derivative
trainSGBasslineCor <- as.data.frame(savitzkyGolay(train[ , allSpectralDataOR], p = 3, w = 11, m = 2))
trainSGBasslineCor <- cbind(train[ , 'PIDN'], trainSGBasslineCor, train[ , spatialPredictorsOR], train$Depth)
names(trainSGBasslineCor)[1] <- 'PIDN'
names(trainSGBasslineCor)[length(trainSGBasslineCor)] <- 'Depth'

#Spectra / CO2 and others
CO2Signal <- seq(which(names(trainSGBasslineCor) == 'm2379.76'), which(names(trainSGBasslineCor) == 'm2352.76'))
allSpectralData <- seq(2, length(trainSGBasslineCor) - 16)
allSpectralDataNoCO2 <- c(seq(2, which(names(trainSGBasslineCor) == 'm2379.76')), 
                          seq(which(names(trainSGBasslineCor) == 'm2352.76'), length(trainSGBasslineCor) - 16))
spatialPredictors <- seq(which(names(trainSGBasslineCor) == 'BSAN'), which(names(trainSGBasslineCor) == 'TMFI'))
depthIx <- which(names(trainSGBasslineCor) == 'Depth')

#Subsoil all spectra
SGBassallSpectraSubsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainSGBasslineCor$Depth == 0), trainSGBasslineCor)
#Topsoil all spectra
SGBassallSpectraTopsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainSGBasslineCor$Depth == 1), trainSGBasslineCor)
#Subsoil no CO2
SGBassSpectralDataNoCO2Subsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2, subsample = which(trainSGBasslineCor$Depth == 0), trainSGBasslineCor)
#Topsoil no CO2
SGBassSpectralDataNoCO2Topsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2, subsample = which(trainSGBasslineCor$Depth == 1), trainSGBasslineCor)

multiplot(SGBassallSpectraSubsoil, SGBassallSpectraTopsoil,
          SGBassSpectralDataNoCO2Subsoil, SGBassSpectralDataNoCO2Topsoil, cols = 2)

#MSC (Multiplicative Scatter Correction)
trainMSC <- as.data.frame(msc(as.matrix(train[ , allSpectralDataOR])))
trainMSC <- cbind(train[ , 'PIDN'], trainMSC, train[ , spatialPredictorsOR], train$Depth)
names(trainMSC)[1] <- 'PIDN'
names(trainMSC)[length(trainMSC)] <- 'Depth'

#Spectra / CO2 and others
CO2Signal <- seq(which(names(trainMSC) == 'm2379.76'), which(names(trainMSC) == 'm2352.76'))
allSpectralData <- seq(2, length(trainMSC) - 16)
allSpectralDataNoCO2 <- c(seq(2, which(names(trainMSC) == 'm2379.76')), 
                          seq(which(names(trainMSC) == 'm2352.76'), length(trainMSC) - 16))
spatialPredictors <- seq(which(names(trainMSC) == 'BSAN'), which(names(trainMSC) == 'TMFI'))
depthIx <- which(names(trainMSC) == 'Depth')

#Subsoil all spectra
MSCSpectraSubsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainMSC$Depth == 0), trainMSC)
#Topsoil all spectra
MSCSpectraTopsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainMSC$Depth == 1), trainMSC)
#Subsoil no CO2
MSCSpectraNoCO2Subsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2, subsample = which(trainMSC$Depth == 0), trainMSC)
#Topsoil no CO2
MSCSpectraNoCO2Topsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2, subsample = which(trainMSC$Depth == 1), trainMSC)

multiplot(MSCSpectraSubsoil, MSCSpectraTopsoil, MSCSpectraNoCO2Subsoil, MSCSpectraNoCO2Topsoil, cols = 2)

#Asymmetric Least Squares
trainALS <- as.data.frame(baseline.corr(train[ , allSpectralDataOR]))
trainALS <- cbind(train[ , 'PIDN'], trainALS, train[ , spatialPredictorsOR], train$Depth)
names(trainALS)[1] <- 'PIDN'
names(trainALS)[length(trainALS)] <- 'Depth'

#Spectra / CO2 and others
CO2Signal <- seq(which(names(trainALS) == 'm2379.76'), which(names(trainALS) == 'm2352.76'))
allSpectralData <- seq(2, length(trainALS) - 16)
allSpectralDataNoCO2 <- c(seq(2, which(names(trainALS) == 'm2379.76')), 
                          seq(which(names(trainALS) == 'm2352.76'), length(trainALS) - 16))
spatialPredictors <- seq(which(names(trainALS) == 'BSAN'), which(names(trainALS) == 'TMFI'))
depthIx <- which(names(trainALS) == 'Depth')

#Subsoil all spectra
ALSSpectraSubsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainALS$Depth == 0), trainALS)
#Topsoil all spectra
ALSSpectraTopsoil <- plotSpectra(10, spectralData = allSpectralData, subsample = which(trainALS$Depth == 1), trainALS)
#Subsoil no CO2
ALSDataNoCO2Subsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2, subsample = which(trainALS$Depth == 0), trainALS)
#Topsoil no CO2
ALSDataNoCO2Topsoil <- plotSpectra(10, spectralData = allSpectralDataNoCO2, subsample = which(trainALS$Depth == 1), trainALS)

multiplot(ALSSpectraSubsoil, ALSSpectraTopsoil, ALSDataNoCO2Subsoil, ALSDataNoCO2Topsoil, cols = 2)

################################
#EDA
#Ca histogram
Cagg <- ggplot(data = train, aes(x = Ca)) +  geom_histogram() 
#P histogram
Pgg <- ggplot(data = train, aes(x = P)) +  geom_histogram() 
#pH histogram
pHgg <- ggplot(data = train, aes(x = pH)) +  geom_histogram()
#SOC histogram
SOCgg <- ggplot(data = train, aes(x = SOC)) +  geom_histogram()
#Sand histogram
Sandgg <- ggplot(data = train, aes(x = Sand)) +  geom_histogram()

multiplot(Cagg, Pgg, pHgg, SOCgg, Sandgg, cols = 3)

###################################################
#Data Smoothing and Baseine Correction
#Train Data
#Running Polinomial Smoother (Savitzky-Golay filtering)
dataALSSGS <- as.data.frame(savitzkyGolay(rbind(train[ , allSpectralDataOR], test[ , allSpectralDataOR]),
                                               p = 2, w = 19, m = 1))
#dataALSSGS <- as.data.frame(baseline.corr(dataSGSmoother))
#training data
trainALSSGS <- cbind(train[ , 'PIDN'], dataALSSGS[1:dim(train)[1], ], train[ , spatialPredictorsOR],
                     train$Depth, train[ , c('Ca', 'P', 'pH', 'SOC', 'Sand')])
names(trainALSSGS)[1] <- 'PIDN'
names(trainALSSGS)[length(trainALSSGS) - 5] <- 'Depth'
#test data
testALSSGS <- cbind(test[ , 'PIDN'], dataALSSGS[(dim(train)[1] + 1):dim(dataALSSGS)[1], ],
                    test[ , spatialPredictorsOR],
                    test$Depth)
names(testALSSGS)[1] <- 'PIDN'
names(testALSSGS)[length(testALSSGS)] <- 'Depth'

#Make a Shuffled CSV to train h2o models
set.seed(1011)
randomSubset <- sample.int(nrow(trainALSSGS), nrow(trainALSSGS)) #full data
trainALSSGS <- trainALSSGS[randomSubset, ]
trainALSSGS$PLog <- log(trainALSSGS$P + 2)
write.csv(trainALSSGS, file = paste0(dataDirectory, 'trainingALSSGSShuffled.csv'), row.names = FALSE)

#Make CSV to test h2o models
write.csv(testALSSGS, file = paste0(dataDirectory, 'testALSSGS.csv'), row.names = FALSE)

###################################################
#MODELLING
#GBM
train <- read.csv(paste0(dataDirectory, 'trainingALSSGSShuffled.csv'), header = TRUE, stringsAsFactors = FALSE)

#Spectra / CO2 and others
CO2Signal <- seq(which(names(train) == 'm2379.76'), which(names(train) == 'm2352.76'))
allSpectralData <- seq(2, length(train) - 21)
allSpectralDataNoCO2 <- c(seq(2, which(names(train) == 'm2379.76')), 
                          seq(which(names(train) == 'm2352.76'), length(train) - 21))
spatialPredictors <- seq(which(names(train) == 'BSAN'), which(names(train) == 'TMFI'))
depthIx <- which(names(train) == 'Depth')

#Cross Validation Control Params
GBMControl <- trainControl(method = "cv",
                           number = 5,
                           verboseIter = TRUE)

gbmGrid <- expand.grid(.interaction.depth = c(3, 5),
                       .shrinkage = c(0.003, 0.01), 
                       .n.trees = 1000)

#Hiper parameter 5-fold Cross-validation "Ca"
set.seed(1005)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODCa <- train(form = Ca~., 
                  data = train[randomSubset , c(allSpectralDataNoCO2, spatialPredictors, depthIx, 3586)],
                  method = "gbm",
                  tuneGrid = gbmGrid,
                  trControl = GBMControl,
                  distribution = 'gaussian',
                  nTrain = floor(nrow(train) * 0.7),
                  verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODCa)  + theme(legend.position = "top")

#Find optimal number of trees
treesCa <- treeFinder(gbmMODCa$finalModel, dataNew = train[randomSubset , c(allSpectralDataNoCO2, spatialPredictors, depthIx, 3586)])

#---------------------------------------------------------------
#Hiper parameter 5-fold Cross-validation "P"
set.seed(1006)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODP <- train(form = P ~ ., 
                 data = train[randomSubset , c(allSpectralDataNoCO2, spatialPredictors, depthIx, 3587)],
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 trControl = GBMControl,
                 distribution = 'gaussian',
                 nTrain = floor(nrow(train) * 0.7),
                 verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODP)  + theme(legend.position = "top")

#Find optimal number of trees
treesP <- treeFinder(gbmMODP$finalModel, dataNew = train[randomSubset , c(allSpectralDataNoCO2, spatialPredictors, depthIx, 3587)])

#---------------------------------------------------------------
#Hiper parameter 5-fold Cross-validation "ph"
set.seed(1007)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODph <- train(form = pH ~ ., 
                  data = train[randomSubset , c(allSpectralDataNoCO2, spatialPredictors, depthIx, 3588)],
                  method = "gbm",
                  tuneGrid = gbmGrid,
                  trControl = GBMControl,
                  distribution = 'gaussian',
                  nTrain = floor(nrow(train) * 0.7),
                  verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODph)  + theme(legend.position = "top")

#Find optimal number of trees
treesph <- treeFinder(gbmMODph$finalModel, dataNew = train[randomSubset , c(allSpectralDataNoCO2, spatialPredictors, depthIx, 3588)])

#---------------------------------------------------------------
#Hiper parameter 5-fold Cross-validation "SOC"
set.seed(1008)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODSOC <- train(form = SOC ~ ., 
                   data = train[randomSubset , c(allSpectralDataNoCO2, spatialPredictors, depthIx, 3589)],
                   method = "gbm",
                   tuneGrid = gbmGrid,
                   trControl = GBMControl,
                   distribution = 'gaussian',
                   nTrain = floor(nrow(train) * 0.7),
                   verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODSOC)  + theme(legend.position = "top")

#Find optimal number of trees
treesSOC <- treeFinder(gbmMODSOC$finalModel, dataNew = train[randomSubset , c(allSpectralDataNoCO2, spatialPredictors, depthIx, 3589)])

#---------------------------------------------------------------
#Hiper parameter 5-fold Cross-validation "Sand"
set.seed(1009)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data
gbmMODSand <- train(form = Sand ~ ., 
                    data = train[randomSubset , c(allSpectralDataNoCO2, spatialPredictors, depthIx, 3590)],
                    method = "gbm",
                    tuneGrid = gbmGrid,
                    trControl = GBMControl,
                    distribution = 'gaussian',
                    nTrain = floor(nrow(train) * 0.7),
                    verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODSand)  + theme(legend.position = "top")

#Find optimal number of trees
treesSand <- treeFinder(gbmMODSand$finalModel, dataNew = train[randomSubset , c(allSpectralDataNoCO2, spatialPredictors, depthIx, 3590)])

#########################################################
#Final Models using h2o GBMs 
#Create an h2o parsed data
require('h2o')
localH2O = h2o.init(ip = "localhost", port = 54321, max_mem_size = '4g', startH2O = TRUE)
africa.hex = h2o.importFile(localH2O, path = paste0(dataDirectory, 'trainingALSSGSShuffled.csv'))

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
africaTest.hex = h2o.importFile(localH2O, path = paste0(dataDirectory, 'testALSSGS.csv'))
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
submissionTemplate$P <- unlist(GBMPredictionP)
submissionTemplate$pH <- unlist(GBMPredictionpH)
submissionTemplate$SOC <- unlist(GBMPredictionSOC)
submissionTemplate$Sand <- unlist(GBMPredictionSand)
write.csv(submissionTemplate, file = "PredictionGBMI.csv", row.names = FALSE)

#########################################################
#Read predictors
africaTrain <- read.csv(paste0(dataDirectory, 'trainingALSSGSShuffled.csv'), header = TRUE, stringsAsFactors = FALSE)

CO2Signal <- seq(which(names(africaTrain) == 'm2379.76'), which(names(africaTrain) == 'm2352.76'))

allSpectralData <- seq(2, length(africaTrain) - 22)
allSpectralDataNoCO2 <- c(seq(2, which(names(africaTrain) == 'm2379.76')), 
                          seq(which(names(africaTrain) == 'm2352.76'), length(africaTrain) - 22))

spatialPredictors <- seq(which(names(africaTrain) == 'BSAN'), which(names(africaTrain) == 'TMFI'))
depthIx <- which(names(africaTrain) == 'Depth')

rm(africaTrain)
#########################################################
#Deep NNs
#hyperparameter search
hyperParametersAllSpectra <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingALSSGSShuffled.csv'),
                                                            predictorsCols = allSpectralData,
                                                            maxMem = '13g')
hyperParametersSpectraNoCO2 <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingALSSGSShuffled.csv'),
                                                              predictorsCols = allSpectralDataNoCO2,
                                                              maxMem = '13g')
hyperParametersAllSpectraDepth <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingALSSGSShuffled.csv'),
                                                                 predictorsCols = c(allSpectralData, depthIx), 
                                                                 maxMem = '13g')
hyperParametersSpectraNoCO2Depth <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingALSSGSShuffled.csv'),
                                                                   predictorsCols = c(allSpectralDataNoCO2, depthIx),
                                                                   maxMem = '13g')
hyperParametersAllData <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingALSSGSShuffled.csv'),
                                                         predictorsCols = c(allSpectralData, spatialPredictors, depthIx),
                                                         maxMem = '13g')
hyperParametersAllDataNoCO2 <- gridCrossValidationh2oDeepnets(DataDir = paste0(dataDirectory, 'trainingALSSGSShuffled.csv'),
                                                              predictorsCols = c(allSpectralDataNoCO2, spatialPredictors, depthIx),
                                                              maxMem = '13g')

noDropout <- c('Rectifier', 'Tanh', 'Maxout')
hidden_layers = list(c(200, 200), c(100, 100, 100), c(200, 200, 200))
gridAda <- expand.grid(c(0.95, 0.99), c(1e-12, 1e-10), stringsAsFactors = TRUE) #this creates all possible combinations
gridLs <- expand.grid(c(0, 1e-5), c(0, 1e-5), stringsAsFactors = TRUE) #this creates all possible combinations

#--------------------------------------------------
#Create an h2o parsed data
require('h2o')
localH2O <- h2o.init(ip = "localhost", port = 54321, max_mem_size = '13g', startH2O = TRUE, nthreads = -1)
africa.hex <- h2o.importFile(localH2O, path = paste0(dataDirectory, 'trainingALSSGSShuffled.csv'))

#Test Data
africaTest.hex <- h2o.importFile(localH2O, path = paste0(dataDirectory, 'testALSSGS.csv'))

predictionsNN <- apply(hyperParametersAllDataNoCO2[[1]], 1, function(hyperparameters, trainHex, testHex, nEnsembles){
  Ensemble <- sapply(1:nEnsembles, function(dummy){
    print(h2o.ls(localH2O))
    DeepNNModel <- h2o.deeplearning(x = c(allSpectralData, spatialPredictors, depthIx),
                                    y = hyperparameters[1],
                                    data = trainHex,
                                    classification = FALSE, balance_classes = FALSE, 
                                    activation = hyperparameters[2],
                                    hidden = hidden_layers[[as.numeric(hyperparameters[3])]],
                                    adaptive_rate = TRUE,                                    
                                    rho = gridAda[as.numeric(hyperparameters[4]), 1],
                                    epsilon = gridAda[as.numeric(hyperparameters[4]), 2],
                                    input_dropout_ratio = 0,
                                    l1 = gridLs[as.numeric(hyperparameters[5]), 1],
                                    l2 = gridLs[as.numeric(hyperparameters[5]), 2],
                                    loss = 'MeanSquare',
                                    reproducible = FALSE,
                                    epochs = 200)  
    #Prediction
    if(hyperparameters[1] == 'PLog'){
      NNPrediction <- exp(as.data.frame(h2o.predict(DeepNNModel, newdata = testHex))) - 2
    }else{
      NNPrediction <- as.data.frame(h2o.predict(DeepNNModel, newdata = testHex))
    }
    #Clean data in server
    print(h2o.ls(localH2O))
    h2o.rm(object = localH2O, keys = h2o.ls(localH2O)$Key[1:(length(h2o.ls(localH2O)$Key) - 2)])
    return(NNPrediction)      
  })
  meansEnsemble <- rowMeans(as.data.frame(Ensemble))
  return(meansEnsemble)
  
}, africa.hex, africaTest.hex, 2)

#h2o shutdown WARNING, All data on the server will be lost!
h2o.shutdown(localH2O, prompt = FALSE)
#Predictions Matrix
predictionsNN <- as.data.frame(predictionsNN)

#########################################################
#Write .csv 
#Deep NN
submissionTemplate[ , 2:6] <- predictionsNN[ , 1:5] 
write.csv(submissionTemplate, file = "PredictionDeepNNV.csv", row.names = FALSE)

#Deep NN P log
submissionTemplate[ , 3] <- predictionsNN[ , 6] 
submissionTemplate[ , c(2, 4, 5, 6)] <- predictionsNN[ , c(1, 3, 4, 5)] 
write.csv(submissionTemplate, file = "PredictionDeepNNVLog.csv", row.names = FALSE)

#Deep NN P combined
submissionTemplate[ , 3] <- rowMeans(cbind(predictionsNN[ , 2], predictionsNN[ , 6]))
submissionTemplate[ , c(2, 4, 5, 6)] <- predictionsNN[ , c(1, 3, 4, 5)] 
write.csv(submissionTemplate, file = "PredictionDeepNNVCombined.csv", row.names = FALSE)

####################################################
#GLMNET
#Cross-validation
africaTrain <- read.csv(paste0(dataDirectory, 'trainingALSSGSShuffled.csv'), header = TRUE, stringsAsFactors = FALSE)
africaTest <- read.csv(paste0(dataDirectory, 'testALSSGS.csv'), header = TRUE, stringsAsFactors = FALSE)

africaTrainMatrix <- model.matrix(~ . , data = africaTrain[, c(allSpectralDataNoCO2, spatialPredictors, depthIx)]) 
africaTestMatrix <- model.matrix(~ . , data = africaTest[ , c(allSpectralDataNoCO2, spatialPredictors, depthIx)])

#CA
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
registerDoParallel(detectCores() - 1)
trainCVCa <- cv.glmnet(x = africaTrainMatrix,
                       y = africaTrain[ , 'Ca'], 
                       nfolds = 5,
                       parallel = TRUE, family = 'gaussian')
plot(trainCVCa)
coef(trainCVCa)
#Recommended use
plot(trainCVCa$glmnet.fit, xvar="lambda", label=TRUE)

#Prediction
GLMNETPredictionCa <- predict(trainCVCa, newx = africaTestMatrix, s= "lambda.min")   #it needs to be fixed, since model matrix deletes data, the test matrix ends up being incomplete

#P
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
registerDoParallel(detectCores() - 1)
trainCVP <- cv.glmnet(x = africaTrainMatrix,
                      y = africaTrain[ , 'P'], 
                      nfolds = 5,
                      parallel = TRUE, family = 'gaussian')
plot(trainCVP)
coef(trainCVP)
#Recommended use
plot(trainCVP$glmnet.fit, xvar="lambda", label=TRUE)

#Prediction
GLMNETPredictionP <- predict(trainCVP, newx = africaTestMatrix, s= "lambda.min")   #it needs to be fixed, since model matrix deletes data, the test matrix ends up being incomplete

#pH
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
registerDoParallel(detectCores() - 1)
trainCVpH <- cv.glmnet(x = africaTrainMatrix,
                       y = africaTrain[ , 'pH'], 
                       nfolds = 5,
                       parallel = TRUE, family = 'gaussian')
plot(trainCVpH)
coef(trainCVpH)
#Recommended use
plot(trainCVpH$glmnet.fit, xvar="lambda", label=TRUE)

#Prediction
GLMNETPredictionpH <- predict(trainCVpH, newx = africaTestMatrix, s= "lambda.min")   #it needs to be fixed, since model matrix deletes data, the test matrix ends up being incomplete

#SOC
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
registerDoParallel(detectCores() - 1)
trainCVSOC <- cv.glmnet(x = africaTrainMatrix,
                        y = africaTrain[ , 'SOC'], 
                        nfolds = 5,
                        parallel = TRUE, family = 'gaussian')
plot(trainCVSOC)
coef(trainCVSOC)
#Recommended use
plot(trainCVSOC$glmnet.fit, xvar="lambda", label=TRUE)

#Prediction
GLMNETPredictionSOC <- predict(trainCVSOC, newx = africaTestMatrix, s= "lambda.min")   #it needs to be fixed, since model matrix deletes data, the test matrix ends up being incomplete

#Sand
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
registerDoParallel(detectCores() - 1)
trainCVSand <- cv.glmnet(x = africaTrainMatrix,
                         y = africaTrain[ , 'Sand'], 
                         nfolds = 5,
                         parallel = TRUE, family = 'gaussian')
plot(trainCVSand)
coef(trainCVSand)
#Recommended use
plot(trainCVSand$glmnet.fit, xvar="lambda", label=TRUE)

#Prediction
GLMNETPredictionSand <- predict(trainCVSand, newx = africaTestMatrix, s= "lambda.min")   #it needs to be fixed, since model matrix deletes data, the test matrix ends up being incomplete

#Write .csv
submissionTemplate$Ca <- GLMNETPredictionCa
submissionTemplate$P <- GLMNETPredictionP
submissionTemplate$pH <- GLMNETPredictionpH
submissionTemplate$SOC <- GLMNETPredictionSOC
submissionTemplate$Sand <- GLMNETPredictionSand
write.csv(submissionTemplate, file = "PredictionGLMNETI.csv", row.names = FALSE)

