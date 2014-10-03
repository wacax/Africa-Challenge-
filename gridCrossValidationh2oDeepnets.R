gridCrossValidationh2oDeepnets <- function(H2OParsedDataObject, 
                                           predictorsCols = 1:dim(as.data.frame(africa.hex))[2],
                                           noOfEpochs = 5, nFolds = 5, printScore = TRUE){
  
  require('Metrics')
  
  #Use 80% of data and 20% of test scores
  Hex80 <- H2OParsedDataObject[1:floor(dim(H2OParsedDataObject)[1] * 0.8), ]
  HexTest <- H2OParsedDataObject[(floor(dim(H2OParsedDataObject)[1] * 0.8) + 1):floor(dim(H2OParsedDataObject)[1]), ]  
    
  #Cols to predict
  predCol <- c('Ca', 'P', 'pH', 'SOC', 'Sand')
  activations <- c('RectifierWithDropout', 'TanhWithDropout', 'MaxoutWithDropout', 
                   'Rectifier', 'Tanh', 'Maxout')
  targets80 <- cbind(as.data.frame(Hex80$Ca), as.data.frame(Hex80$P), as.data.frame(Hex80$pH), 
                     as.data.frame(Hex80$SOC), as.data.frame(Hex80$Sand))
  targetsTest <- cbind(as.data.frame(HexTest$Ca), as.data.frame(HexTest$P), as.data.frame(HexTest$pH), 
                       as.data.frame(HexTest$SOC), as.data.frame(HexTest$Sand))  
  noDropout <- c('Rectifier', 'Tanh', 'Maxout')
  
  optimalActivations <- sapply(predCol, function(target){    
    activationsErrors <- sapply(activations, function(actvs){ 
      #n-fold x-validation      
      set.seed(10101)
      folds <- sample(rep(1:nFolds, length = nrow(Hex80)))
      CVErrors <- sapply(1:nFolds, function(k){
        model <- h2o.deeplearning(x = predictorsCols,
                                  y = target,
                                  data = Hex80[folds != k, ],
                                  classification = FALSE, balance_classes = FALSE, 
                                  validation = Hex80[folds == k, ], 
                                  activation = actvs,
                                  hidden = c(30, 30),
                                  input_dropout_ratio = ifelse(actvs %in% noDropout, 0, 0.1),
                                  l2 = ifelse(actvs == 'Rectifier' | actvs == 'Tanh', 1e-5, 0),
                                  epochs = noOfEpochs, force_load_balance = TRUE)  
        
        Prediction <- unlist(as.data.frame(h2o.predict(model, newdata = Hex80[folds == k, ])))
        RMSEError <- rmse(targets80[folds == k, target], Prediction)
        return(RMSEError) 
      })
      print(paste0(target, ' RMSE Error of ', mean(CVErrors), ' with activation: ', actvs))
      return(mean(CVErrors))      
    })    
    return(names(which.min(activationsErrors)))    
  })
  
  #Create a set of network topologies
  hidden_layers = list(c(50, 50), c(100, 100), c(50, 50, 50), c(100, 100, 100))
  
  optimalParameters <- cbind(predCol, optimalActivations)
  
  optimalArchitecture <- apply(optimalParameters, 1, function(parameters){
    activationsErrors <- lapply(hidden_layers, function(architecture){   
      #n-fold x-validation      
      set.seed(10101)
      folds <- sample(rep(1:nFolds, length = nrow(Hex80)))
      CVErrors <- sapply(1:nFolds, function(k){
        model <- h2o.deeplearning(x = predictorsCols,
                                  y = parameters[1],
                                  data = Hex80[folds != k, ],
                                  classification = FALSE, balance_classes = FALSE, 
                                  validation = Hex80[folds == k, ], 
                                  activation = parameters[2],
                                  hidden = architecture,
                                  input_dropout_ratio = ifelse(parameters[2] %in% noDropout, 0, 0.1),
                                  l2 = ifelse(parameters[2] == 'Rectifier' | parameters[2] == 'Tanh', 1e-5, 0),
                                  epochs = noOfEpochs * 2, force_load_balance = TRUE)  
        
        Prediction <- unlist(as.data.frame(h2o.predict(model, newdata = Hex80[folds == k, ])))
        RMSEError <- rmse(targets80[folds == k, target], Prediction)
        return(RMSEError)         
      })
      print(paste0(parameters[1], ' RMSE Error of ', mean(CVErrors), ' with activation: ', parameters[2], 
                   ' and ', length(architecture), ' hidden layers of ', architecture, ' units each'))
      return(mean(CVErrors))            
    })    
    return(which.min(unlist(activationsErrors)))    
  })
  
  optimalParameters <- cbind(optimalParameters, optimalArchitecture)  
  #ADADELTA
  #grid search
  gridAda <- expand.grid(c(0.9, 0.95, 0.99), c(1e-12, 1e-10, 1e-8, 1e-6), stringsAsFactors = TRUE) #this creates all possible combinations
  
  optimalAdaParams <- apply(optimalParameters, 1, function(parameters){
    activationsErrors <- apply(gridAda, 1, function(adaDelta){ 
      CVErrors <- sapply(1:nFolds, function(k){        
        model <- h2o.deeplearning(x = predictorsCols,
                                  y = parameters[1],
                                  data = Hex80[folds != k, ],
                                  classification = FALSE, balance_classes = FALSE,
                                  validation = Hex80[folds == k, ],                                 
                                  activation = parameters[2],
                                  hidden = hidden_layers[[as.numeric(parameters[3])]],
                                  adaptive_rate = TRUE,
                                  rho = as.numeric(adaDelta[1]),
                                  epsilon = as.numeric(adaDelta[2]),
                                  input_dropout_ratio = ifelse(parameters[2] %in% noDropout, 0, 0.1),
                                  l2 = ifelse(parameters[2] == 'Rectifier' | parameters[2] == 'Tanh', 1e-5, 0),
                                  epochs = noOfEpochs * 2, force_load_balance = TRUE)  
        
        Prediction <- unlist(as.data.frame(h2o.predict(model, newdata = Hex80[folds == k, ])))
        RMSEError <- rmse(targets80[folds == k, target], Prediction)         
        return(RMSEError)        
      })
      print(paste0(parameters[1], ' RMSE Error of ', mean(CVErrors), ' with activation: ', parameters[2], 
                   ' and ', length(hidden_layers[[as.numeric(parameters[3])]]), ' hidden layers of ',
                   hidden_layers[[as.numeric(parameters[3])]], ' units each. Adadelta rho of ', as.numeric(adaDelta[1]), 
                   ' and Epsilon of: ', as.numeric(adaDelta[2])))   
      return(mean(CVErrors))      
    })    
    return(which.min(activationsErrors))    
  })
  
  optimalParameters <- cbind(optimalParameters, optimalAdaParams)
  
  #Compute Combined Score  
  if(printScore == TRUE){
    RMSEs <- apply(optimalParameters, 1, function(parameters){
      model <- h2o.deeplearning(x = predictorsCols,
                                y = parameters[1],
                                data = Hex80,
                                classification = FALSE, balance_classes = FALSE, 
                                activation = parameters[2],
                                hidden = hidden_layers[[as.numeric(parameters[3])]],
                                adaptive_rate = TRUE,
                                rho = gridAda[as.numeric(parameters[4]), 1],
                                epsilon = gridAda[as.numeric(parameters[4]), 2],
                                input_dropout_ratio = ifelse(parameters[2] %in% noDropout, 0, 0.1),
                                l2 = ifelse(parameters[2] == 'Rectifier' | parameters[2] == 'Tanh', 1e-5, 0),
                                epochs = noOfEpochs * 4, force_load_balance = TRUE)  
      
      Prediction <- unlist(as.data.frame(h2o.predict(model, newdata = HexTest)))
      if(parameters[1] != 'P'){
        RMSEError <- rmse(targetsTest[parameters[1]], Prediction)        
      }else{
        RMSEError <- rmse(exp(targetsTest[parameters[1]] - 2), exp(Prediction) - 2)
      }
      print(paste0(parameters[1], ' RMSE Error of ', RMSEError, ' with activation: ', parameters[2], 
                   ' and ', length(hidden_layers[[as.numeric(parameters[3])]]), ' hidden layers of ',
                   hidden_layers[[as.numeric(parameters[3])]], ' units each. Adadelta rho of ', gridAda[as.numeric(parameters[4]), 1], 
                   ' and Epsilon of: ', gridAda[as.numeric(parameters[4]), 2]))
      return(RMSEError)
    })  
    print(paste0('MCRMSE Score of: ', mean(RMSEs)))
  }
  #Return all results plus errors
  return(list(optimalParameters, RMSEs))
}
