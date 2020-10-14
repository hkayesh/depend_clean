nb.train <- function(dtm.train, labels.train, LABEL) {
 
  # Apply the convert_count function to get final training data
  trainNB <- apply(dtm.train, 2, convert_count)
  
  # train model
  classifier <- naiveBayes(trainNB, labels.train, laplace = 1)
  
  # save model
  save(classifier, file = paste0(models.path, LABEL, ".model"))
  
  return (classifier)
}

nb.predict <- function(dtm.pred, LABEL, response.type = 'class') {
  
  testNB <- apply(dtm.test, 2, convert_count)
  
  classifier <- load.model(LABEL)
  if(response.type == 'proba')
    #spred <- predict(classifier, newx=test_mat, s = "lambda.min", type="response")
  else {
    spred <- predict(classifier, newdata=testNB) 
  }
  
  return(spred)
}

nb.evaluate <- function(dtm.train, labels.train, dtm.test, labels.test, LABEL) {
  nb.train(dtm.train, labels.train, LABEL)
  spred <- nb.predict(dtm.test, LABEL)
  
  conf.mat <- confusionMatrix(spred, labels.test, positive = LABEL)
  cat("\nNaive Bayes\n")
  header<-paste("P", "R", "F1\n", sep = '\t')
  content<-paste(round(conf.mat$byClass[5] ,5), round(conf.mat$byClass[6] ,5), round(conf.mat$byClass[7] ,5), sep = '\t')
  cat(header)
  cat(content)
}