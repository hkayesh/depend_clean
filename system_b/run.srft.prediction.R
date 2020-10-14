run.srft.prediction <-function(training.data, prediction.data, output.path) {
  
  all_data = rbind.fill(training.data, prediction.data)
  
  dataset_size = length(all_data$comment)
  train_data_count = length(training.data$comment)
  
  dtm = nlp.preprocess(all_data$comment)
  
  dtm.train <- dtm[1:train_data_count, ]
  dtm.test <- dtm[(train_data_count+1):dataset_size, ]
  
  labels <- c('environment', 'waiting time', 'staff attitude and professionalism', 'care quality')
  
  for (label in labels) {
    labels.train <- as.factor(new.enet.corpus(training.data, label))
    
    if (label == 'staff attitude and professionalism') {
      enet.train(dtm.train, labels.train, label)
    } 
    else if (label == 'care quality' || label == 'environment') {
      svm.train(dtm.train, labels.train, label, kernel.name = 'linear', c.value =  79.8)
    }
    else { # waiting time
      enet.train(dtm.train, labels.train, label)
      #svm.train(dtm.train, labels.train, label, kernel.name = 'rbf', c.value =  79.8)
    }
  }
  
  label <- 'environment'
  env_pred <- svm.predict(dtm.test, label)
  env_pred_raw <- svm.predict(dtm.test, label, response.type='proba')
  
  label <- 'waiting time'
  wt_pred <- enet.predict(dtm.test, label)
  wt_pred_raw <- enet.predict(dtm.test, label, response.type='proba')
  
  label <- 'staff attitude and professionalism'
  saap_pred <- enet.predict(dtm.test, label)
  saap_pred_raw <- enet.predict(dtm.test, label, response.type='proba')
  
  label <- 'care quality'
  cq_pred <- svm.predict(dtm.test, label)
  cq_pred_raw <- svm.predict(dtm.test, label, response.type='proba')
  
  #combine predictions and save in correct format
  result_data <- data.frame(stringsAsFactors=FALSE)
  
  comments = prediction.data$comment
  max.num.categories = 11
  
  #food.and.parking = apply.dictionaries(comments) 
  for (i in 1:length(saap_pred)) {
    row = data.frame(matrix(NA, 1, max.num.categories))
    
    # declare dummy column headers; required for rbind
    names(row) <-LETTERS[1:length(row)] 
    
    aspects <- list()
    if (env_pred[i] != "0") {
      aspects[length(aspects) + 1] <- paste(env_pred[i], env_pred_raw[i, "environment"])
    }
    
    if (wt_pred[i] != "0") {
      aspects[length(aspects)+1] <- paste(wt_pred[i], wt_pred_raw[i, "1"])
    }
    
    if (saap_pred[i] != "0") {
      aspects[length(aspects)+1] <- paste(saap_pred[i], saap_pred_raw[i, "1"])
    }
    
    if (cq_pred[i] != "0") {
      aspects[length(aspects)+1] <- paste(cq_pred[i], cq_pred_raw[i, "care quality"])
    }
    
    #if (length(food.and.parking[[i]]) > 0) {
    #  aspects = c(aspects, food.and.parking[[i]])
    #}
    
    # insert comemnt at the fist column 
    row[1, 1] = comments[i] 
    
    # if no aspect found, default is 'other'
    if (length(aspects) > 0) {
      for (j in 1:length(aspects)) {
        row[1, j+1] = aspects[j]    
      } 
    }
    else {
      row[1, 2] = 'other 0.01'
    }
    
    result_data <- rbind(result_data, row)
  }
  
  # save data as csv file
  write.table(result_data, file=paste(output.path), row.names=FALSE, col.names = FALSE, sep = ",", na = "", qmethod = c("escape", "double"))
  cat (paste("System B output saved to the file: ", output.path))
} 