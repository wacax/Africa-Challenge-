gridCrossValidationh2oDeepnets <- function(DataDir, 
                                           predictorsCols = 1:dim(as.data.frame(DataDir))[2],
                                           nFolds = 5, printScore = FALSE, maxMem = '1g'){
  
  require('h2o')
  require('Metrics')
  
  #Cols to predict
  predCol <- c('Ca', 'P', 'pH', 'SOC', 'Sand', 'PLog')
  activations <- c('RectifierWithDropout', 'TanhWithDropout', 'MaxoutWithDropout', 
                   'Rectifier', 'Tanh', 'Maxout')
  noDropout <- c('Rectifier', 'Tanh', 'Maxout')
  
  localH2O <- h2o.init(ip = "localhost", port = 54321, max_mem_size = maxMem, startH2O = TRUE, nthreads = -1)
  hex <- h2o.importFile(localH2O, path = DataDir)
  splitObject <- h2o.splitFrame(hex, ratios = 0.8, shuffle = TRUE)
  targets80 <- cbind(as.data.frame(splitObject[[1]]$Ca), as.data.frame(splitObject[[1]]$P), as.data.frame(splitObject[[1]]$pH), 
                     as.data.frame(splitObject[[1]]$SOC), as.data.frame(splitObject[[1]]$Sand), as.data.frame(splitObject[[1]]$Plog))
  targetsTest <- cbind(as.data.frame(splitObject[[2]]$Ca), as.data.frame(splitObject[[2]]$P), as.data.frame(splitObject[[2]]$pH), 
                       as.data.frame(splitObject[[2]]$SOC), as.data.frame(splitObject[[2]]$Sand), as.data.frame(splitObject[[2]]$Plog)) 
  
  optimalActivations <- sapply(predCol, function(target){    
    activationsErrors <- sapply(activations, function(actvs){       
      #n-fold x-validation      
      set.seed(10101)
      folds <- sample(rep(1:nFolds, length = nrow(splitObject[[1]])))
      CVErrors <- sapply(1:nFolds, function(k){
        model <- h2o.deeplearning(x = predictorsCols,
                                  y = target,
                                  data = splitObject[[1]][folds != k, ],
                                  classification = FALSE, balance_classes = FALSE, 
                                  validation = splitObject[[1]][folds == k, ], 
                                  activation = actvs,
                                  hidden = c(50, 50),
                                  input_dropout_ratio = 0,
                                  l2 = ifelse(actvs == 'Rectifier' | actvs == 'Tanh', 1e-5, 0),
                                  loss = 'MeanSquare',  
                                  reproducible = FALSE,
                                  epochs = 10)  
        
        Prediction <- unlist(as.data.frame(h2o.predict(model, newdata = splitObject[[1]][folds == k, ])))
        RMSEError <- rmse(targets80[folds == k, target], Prediction)
        h2o.rm(object = localH2O, keys = h2o.ls(localH2O)$Key[1:(length(h2o.ls(localH2O)$Key) - 3)])
        return(RMSEError) 
      })
      print(paste0(target, ' RMSE Error of ', mean(CVErrors), ' with activation: ', actvs))
      return(mean(CVErrors))      
    })    
    return(names(which.min(activationsErrors)))    
  })  
  
  #Update optimal activations
  optimalParameters <- cbind(predCol, optimalActivations)
  
  #Create a set of network topologies
  hidden_layers = list(c(200, 200), c(100, 100, 100), c(200, 200, 200))
  
  optimalArchitecture <- apply(optimalParameters, 1, function(parameters){
    activationsErrors <- lapply(hidden_layers, function(architecture){
      
      #n-fold x-validation      
      set.seed(10102)
      folds <- sample(rep(1:nFolds, length = nrow(splitObject[[1]])))
      CVErrors <- sapply(1:nFolds, function(k){
        model <- h2o.deeplearning(x = predictorsCols,
                                  y = parameters[1],
                                  data = splitObject[[1]][folds != k, ],
                                  classification = FALSE, balance_classes = FALSE, 
                                  validation = splitObject[[1]][folds == k, ], 
                                  activation = parameters[2],
                                  hidden = architecture,
                                  input_dropout_ratio = 0,
                                  l2 = ifelse(parameters[2] == 'Rectifier' | parameters[2] == 'Tanh', 1e-5, 0),
                                  loss = 'MeanSquare',  
                                  reproducible = FALSE,                                  
                                  epochs = floor((architecture[1]/5 * length(architecture))*0.6))  
        
        Prediction <- unlist(as.data.frame(h2o.predict(model, newdata = splitObject[[1]][folds == k, ])))
        RMSEError <- rmse(targets80[folds == k, parameters[1]], Prediction)
        h2o.rm(object = localH2O, keys = h2o.ls(localH2O)$Key[1:(length(h2o.ls(localH2O)$Key) - 3)])
        return(RMSEError)         
      })
      print(paste0(parameters[1], ' RMSE Error of ', mean(CVErrors), ' with activation: ', parameters[2], 
                   ' and ', length(architecture), ' hidden layers of ', architecture, ' units each'))
      return(mean(CVErrors))            
    })    
    return(which.min(unlist(activationsErrors)))    
  })
  
  #Update optimal activations
  optimalParameters <- cbind(optimalParameters, optimalArchitecture)  
  
  #ADADELTA
  #grid search
  gridAda <- expand.grid(c(0.95, 0.99), c(1e-12, 1e-10), stringsAsFactors = TRUE) #this creates all possible combinations
  
  optimalAdaParams <- apply(optimalParameters, 1, function(parameters){
    activationsErrors <- apply(gridAda, 1, function(adaDelta){ 
      CVErrors <- sapply(1:nFolds, function(k){  
        
        #n-fold x-validation      
        set.seed(10103)
        folds <- sample(rep(1:nFolds, length = nrow(splitObject[[1]])))
        model <- h2o.deeplearning(x = predictorsCols,
                                  y = parameters[1],
                                  data = splitObject[[1]][folds != k, ],
                                  classification = FALSE, balance_classes = FALSE,
                                  validation = splitObject[[1]][folds == k, ],                                 
                                  activation = parameters[2],
                                  hidden = hidden_layers[[as.numeric(parameters[3])]],
                                  adaptive_rate = TRUE,
                                  rho = as.numeric(adaDelta[1]),
                                  epsilon = as.numeric(adaDelta[2]),
                                  input_dropout_ratio = 0,
                                  l2 = ifelse(parameters[2] == 'Rectifier' | parameters[2] == 'Tanh', 1e-5, 0),
                                  loss = 'MeanSquare',   
                                  reproducible = FALSE,                                  
                                  epochs = floor((hidden_layers[[as.numeric(parameters[3])]][1]/5 * length(hidden_layers[[as.numeric(parameters[3])]]))*0.6))  
        
        Prediction <- unlist(as.data.frame(h2o.predict(model, newdata = splitObject[[1]][folds == k, ])))
        RMSEError <- rmse(targets80[folds == k, parameters[1]], Prediction)      
        h2o.rm(object = localH2O, keys = h2o.ls(localH2O)$Key[1:(length(h2o.ls(localH2O)$Key) - 3)])
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
  
  #Update optimal activations  
  optimalParameters <- cbind(optimalParameters, optimalAdaParams)  
  
  #l1-l2 regularization
  #grid search
  gridLs <- expand.grid(c(0, 1e-5), c(0, 1e-5), stringsAsFactors = TRUE) #this creates all possible combinations
  
  optimalLParams <- apply(optimalParameters, 1, function(parameters){
    activationsErrors <- apply(gridLs, 1, function(L){ 
      CVErrors <- sapply(1:nFolds, function(k){ 
        
        #n-fold x-validation      
        set.seed(10104)
        folds <- sample(rep(1:nFolds, length = nrow(splitObject[[1]])))
        model <- h2o.deeplearning(x = predictorsCols,
                                  y = parameters[1],
                                  data = splitObject[[1]][folds != k, ],
                                  classification = FALSE, balance_classes = FALSE,
                                  validation = splitObject[[1]][folds == k, ],                                 
                                  activation = parameters[2],
                                  hidden = hidden_layers[[as.numeric(parameters[3])]],
                                  adaptive_rate = TRUE,                                  
                                  rho = gridAda[as.numeric(parameters[4]), 1],
                                  epsilon = gridAda[as.numeric(parameters[4]), 2],
                                  input_dropout_ratio = 0,
                                  l1 = as.numeric(L[1]),
                                  l2 = as.numeric(L[2]),
                                  loss = 'MeanSquare', 
                                  reproducible = FALSE,                                  
                                  epochs = floor((hidden_layers[[as.numeric(parameters[3])]][1]/5 * length(hidden_layers[[as.numeric(parameters[3])]]))*0.6))  
        
        Prediction <- unlist(as.data.frame(h2o.predict(model, newdata = splitObject[[1]][folds == k, ])))
        RMSEError <- rmse(targets80[folds == k, parameters[1]], Prediction)    
        h2o.rm(object = localH2O, keys = h2o.ls(localH2O)$Key[1:(length(h2o.ls(localH2O)$Key) - 3)])
        return(RMSEError)        
      })
      print(paste0(parameters[1], ' RMSE Error of ', mean(CVErrors), ' with activation: ', parameters[2], 
                   ' and ', length(hidden_layers[[as.numeric(parameters[3])]]), ' hidden layers of ',
                   hidden_layers[[as.numeric(parameters[3])]], ' units each. Adadelta rho of ',  gridAda[as.numeric(parameters[4]), 1], 
                   ' and Epsilon of: ',  gridAda[as.numeric(parameters[4]), 2], ' L1 Value of: ', as.numeric(L[1]), 
                   ' L2 Value of: ', as.numeric(L[2])))   
      return(mean(CVErrors))      
    })    
    return(which.min(activationsErrors))    
  })
  
  #Update optimal activations    
  optimalParameters <- cbind(optimalParameters, optimalLParams)
  
  #Compute Combined Score  
  if(printScore == TRUE){
    
    RMSEs <- apply(optimalParameters, 1, function(parameters){      
      model <- h2o.deeplearning(x = predictorsCols,
                                y = parameters[1],
                                data = splitObject[[1]],
                                classification = FALSE, balance_classes = FALSE, 
                                activation = parameters[2],
                                validation = splitObject[[2]],                                  
                                hidden = hidden_layers[[as.numeric(parameters[3])]],
                                adaptive_rate = TRUE,                                
                                rho = gridAda[as.numeric(parameters[4]), 1],
                                epsilon = gridAda[as.numeric(parameters[4]), 2],
                                input_dropout_ratio = 0,
                                l1 = gridLs[as.numeric(parameters[5]), 1],
                                l2 = gridLs[as.numeric(parameters[5]), 2],
                                loss = 'MeanSquare',   
                                reproducible = FALSE,                                
                                epochs = floor((hidden_layers[[as.numeric(parameters[3])]][1]/5 * length(hidden_layers[[as.numeric(parameters[3])]]))*0.6))  
      
      Prediction <- unlist(as.data.frame(h2o.predict(model, newdata = splitObject[[2]])))
      if(parameters[1] != 'PLog'){
        RMSEError <- rmse(targetsTest[parameters[1]], Prediction)        
      }else{
        RMSEError <- rmse((exp(targetsTest[parameters[1]]) - 2), exp(Prediction) - 2)
      }
      h2o.rm(object = localH2O, keys = h2o.ls(localH2O)$Key[1:(length(h2o.ls(localH2O)$Key) - 3)])
      print(paste0(parameters[1], ' RMSE Error of ', RMSEError, ' with activation: ', parameters[2], 
                   ' and ', length(hidden_layers[[as.numeric(parameters[3])]]), ' hidden layers of ',
                   hidden_layers[[as.numeric(parameters[3])]], ' units each. Adadelta rho of ',
                   gridAda[as.numeric(parameters[4]), 1], 
                   ' and Epsilon of: ', gridAda[as.numeric(parameters[4]), 2],
                   ' L1 Value of: ', gridLs[as.numeric(parameters[5]), 1], 
                   ' L2 Value of: ', gridLs[as.numeric(parameters[5]), 2]))
      return(RMSEError)
    })  
    print(paste0('MCRMSE Score of: ', mean(RMSEs[1:5])))
  }
  
  #Return all results plus errors
  return(list(optimalParameters, RMSEs))
}
