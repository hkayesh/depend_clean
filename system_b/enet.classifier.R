enet.train <- function(dtm.train, labels.train, LABEL) {
  train_mat <- chisqTwo(dtm.train, labels.train)
  
  dictionary <- as.data.frame(colnames(train_mat))
  enet.terms.path <<- 'dictionary.enet.Rda'
  saveRDS(dictionary, file=enet.terms.path)
  
  classifier 	<- cv.glmnet(train_mat, labels.train, family = "binomial", nfolds=10, type.measure="class")
  
  # save model
  save(classifier, file = paste0(models.path, LABEL, ".model"))
  
  return (classifier)
}

enet.predict <- function(dtm.pred, LABEL, response.type = 'class') {
  training.dict <- readRDS(enet.terms.path)
  training.terms <<- as.vector(training.dict[, 1])
  
  test_mat  <- testmat(training.terms, as.matrix(dtm.pred))
  
  classifier <- load.model(LABEL)
  if(response.type == 'proba')
    spred <- predict(classifier, newx=data.matrix(test_mat), s = "lambda.min", type="response")
  else {
    spred <- predict(classifier, newx=data.matrix(test_mat), s = "lambda.min", type="class")  
  }
  
  return(spred)
}

enet.evaluate <- function(dtm.train, labels.train, dtm.test, labels.test, LABEL) {
  enet.train(dtm.train, labels.train, LABEL)
  spred <- enet.predict(dtm.test, LABEL)
  
  conf.mat <- confusionMatrix(spred, labels.test, positive = LABEL)
  header<-paste("P", "R", "F1\n", sep = '\t')
  precision <- round(conf.mat$byClass[5] ,5)
  recall <- round(conf.mat$byClass[6] ,5)
  f1.score <- round(conf.mat$byClass[7] ,5)
  content <- paste(precision, recall, f1.score , sep = '\t')
  cat(header)
  cat(content)
  #cat(paste(f1.score, ","))
}
