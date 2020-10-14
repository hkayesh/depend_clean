run.evaluation <-function(training.data, test.data, LABEL) {
  
  set.seed(111)
  
  labels.train <- as.factor(new.enet.corpus(training.data, LABEL))  
  labels.test <- as.factor(new.enet.corpus(test.data, LABEL))  
  
  all_data = rbind.fill(training.data, test.data)
  
  dataset_size = length(all_data$comment)
  train_data_count = length(training.data$comment)
  
  dtm = nlp.preprocess(all_data$comment)
  
  dtm.train <- dtm[1:train_data_count, ]
  dtm.test <- dtm[(train_data_count+1):dataset_size, ]
  
  
  #send comments only to naive bayes (special case)
  cat("\nNaive Bayes\n")
  nb.evaluate(dtm.train, labels.train, dtm.test, labels.test, LABEL)
  
  cat("\nElastic net: logistic regression L1/L2\n")
  enet.evaluate(dtm.train, labels.train, dtm.test, labels.test, LABEL)
  
  cat("\nSVM - Radial Basis kernel\n")
  svm.evaluate(dtm.train, labels.train, dtm.test, labels.test, LABEL, kernel='rbf')
  
  cat("\nSVM - linear kernel\n")
  svm.evaluate(dtm.train, labels.train, dtm.test, labels.test, LABEL)
}