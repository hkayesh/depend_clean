svm.train <- function(dtm.train, labels.train, LABEL, kernel.name='linear', c.value=68.7) {
  #c.value = 68.7 # for MMHSCT dataset
  #c.value = 79.8 # for SRFT dataset
  
  if (kernel.name == 'rbf') {
    kernel.package.name <- 'rbfdot'
  }
  else {
    kernel.package.name <- 'vanilladot'
  }
  
  classifier 	<- ksvm(as.matrix(dtm.train), labels.train, type="C-svc", kernel=kernel.package.name, C=c.value, scaled=c(), prob.model=TRUE)
  
  # save model
  save(classifier, file = paste0(models.path, LABEL, ".model"))
  
  return (classifier)
}

svm.predict <- function(dtm.pred, LABEL, response.type='class') {
  classifier <- load.model(LABEL)
  if (response.type == 'proba') {
    spred  	<- predict(classifier, dtm.pred, type=("probabilities"))
  }
  else { # class
    spred  	<- predict(classifier, dtm.pred)
  }
  return(spred)
}

svm.evaluate <- function(dtm.train, labels.train, dtm.test, labels.test, LABEL, kernel='linear') {
  svm.train(dtm.train, labels.train, LABEL, kernel)
  spred <- svm.predict(dtm.test, LABEL)
  
  conf.mat <- confusionMatrix(spred, labels.test, positive = LABEL)
  header<-paste("P", "R", "F1\n", sep = '\t')
  precision <- round(conf.mat$byClass[5] ,5)
  recall <- round(conf.mat$byClass[6] ,5)
  f1.score <- round(conf.mat$byClass[7] ,5)
  content <- paste(precision, recall, f1.score , sep = '\t')
  cat (header)
  cat (content)
  #cat(paste(f1.score, ","))
}

grid.search <- function(df) {
  getParamSet("classif.ksvm")
  ksvm <- makeLearner("classif.ksvm", predict.type = "response")
  discrete_ps = makeParamSet(
    makeNumericParam("C", lower = 1, upper = 101, trafo = function(x) x+1 ),
    makeNumericParam("sigma", lower = 1, upper = 101)
  )
  ctrl = makeTuneControlGrid()
  rdesc = makeResampleDesc("CV", iters = 5)
  
  classif.task = makeClassifTask(data = df, target = 'class')
  
  res = tuneParams("classif.ksvm", task = classif.task, resampling = rdesc,
                   par.set = discrete_ps, control = ctrl)
}
