gridCrossValidationh2oDeepnets <- function(H2OParsedDataObject, noOfEpochs = 5){
  
  require('Metrics')
  
  #Simple Validation (60/20/20 data split)
  HexTrain <- H2OParsedDataObject[1:floor(dim(H2OParsedDataObject)[1] * 0.6), ]
  HexValid <- H2OParsedDataObject[(floor(dim(H2OParsedDataObject)[1] * 0.6) + 1):floor(dim(H2OParsedDataObject)[1] * 0.8), ]
  HexTest <- H2OParsedDataObject[(floor(dim(H2OParsedDataObject)[1] * 0.8) + 1):floor(dim(H2OParsedDataObject)[1]), ]  
  
  #Cols to predict
  predCol <- c('Ca', 'P', 'pH', 'SOC', 'Sand')
  activations <- c('RectifierWithDropout', 'TanhWithDropout', 'MaxoutWithDropout')
  targetsVal <- cbind(as.data.frame(HexValid$Ca), as.data.frame(HexValid$P), as.data.frame(HexValid$pH), 
                      as.data.frame(HexValid$SOC), as.data.frame(HexValid$Sand))
  
  optimalActivations <- sapply(predCol, function(target){
    
    activationsErrors <- sapply(activations, function(actvs, tar){      
      
      model <- h2o.deeplearning(x = seq(2, 3595),
                                y = target,
                                data = HexTrain,
                                classification = FALSE, balance_classes = FALSE, 
                                activation = actvs,
                                hidden = c(20, 20),
                                hidden_dropout_ratios = c(0.5, 0.5),
                                epochs = noOfEpochs)  
      
      Prediction <- unlist(as.data.frame(h2o.predict(model, newdata = HexValid)))
      RMSEError <- rmse(targetsVal[ , i], Prediction)
      return(RMSEError)    
    }, target)
    
    return(names(which.min(activationsErrors)))
    
  })
  
  
  
  for (i in 1:length(predCol)){
    #run 5 epochs 
    gridSearchRWD <- h2o.deeplearning(x = seq(2, 3595),
                                      y = predCol[i],
                                      data = africaHexTrain,
                                      validation = africaHexValid,
                                      classification = FALSE, balance_classes = FALSE, 
                                      activation = 'RectifierWithDropout',
                                      hidden = c(20, 20),
                                      hidden_dropout_ratios = c(0.1, 0.1),
                                      epochs = noOfEpochs)
    gridSearchTWD <- h2o.deeplearning(x = seq(2, 3595),
                                      y = predCol[i],
                                      data = africaHexTrain,
                                      validation = africaHexValid,
                                      classification = FALSE, balance_classes = FALSE, 
                                      activation = 'TanhWithDropout',
                                      hidden = c(20, 20),
                                      hidden_dropout_ratios = c(0.1, 0.1),
                                      epochs = noOfEpochs)  
    #Run the grid search
    
  }
  
  #Create a set of network topologies
  hidden_layers = list(c(200,200), c(100,300,100),c(500,500,500))
  
  

  
}
