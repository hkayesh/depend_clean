run.training <-function(training.data) {
  labels <- c('environment', 'waiting time', 'staff attitude and professionalism', 'care quality')
  
  for (label in labels) {
    dtm.train <- nlp.preprocess(training.data$comment)
    labels.train <- as.factor(new.enet.corpus(training.data, label))
    
    print (dtm.train)
    
    dictionary <- as.data.frame(Terms(dtm.train))
    saveRDS(dictionary, file=terms.path)
    
    if (label == 'waiting time' || label == 'care quality') {
      enet.train(dtm.train, labels.train, label)
    } 
    else if (label == 'environment') {
      svm.train(dtm.train, labels.train, label, kernel.name = 'linear')
    }
    else { # staff attitude and professionalism
      svm.train(dtm.train, labels.train, label, kernel.name = 'rbf')
    }
  }
}