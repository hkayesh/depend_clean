#NB Pipeline

run.ml <- function(seed = NULL, data, test.data, LABEL){
  #dataset_size = length(data[[1]])
  #train_data_count = length(data[[1]])*train_split
  
  set.seed(seed)
  
  n1 <- data.frame(comment = data$comment)
  n1$class <- new.corpus(data, LABEL)
  
  n1$class <- as.factor(n1$class)
  n1 <- n1[sample(nrow(n1)), ]
  
  n2 <- data.frame(comment = test.data$comment)
  n2$class <- new.corpus(test.data, LABEL)
  
  n2$class <- as.factor(n2$class)
  n2 <- n2[sample(nrow(n2)), ]
  
  
  
  #NLP pre-processing
  training.corpus <- Corpus(VectorSource(n1$comment))
  training.corpus.processed <- training.corpus %>%  
    tm_map(content_transformer(tolower)) %>% 
    tm_map(removePunctuation) %>%
    tm_map(stemDocument) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords(kind="en")) %>%
    tm_map(stripWhitespace)
  
  test.corpus <- Corpus(VectorSource(n2$comment))
  test.corpus.processed <- test.corpus %>%  
    tm_map(content_transformer(tolower)) %>% 
    tm_map(removePunctuation) %>%
    tm_map(stemDocument) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords(kind="en")) %>%
    tm_map(stripWhitespace)
  
  #generate document term matrix
  dtm.train <- DocumentTermMatrix(training.corpus.processed)
  dtm.test <- DocumentTermMatrix(test.corpus.processed)
  
  #subset dataset (60/40%) to training and test
  df.train <- n1
  df.test <- n2
  
  corpus.clean.train <- training.corpus.processed
  corpus.clean.test <- test.corpus.processed
  
  bitrigramtokeniser <- function(x, n) {
    RWeka:::NGramTokenizer(x, RWeka:::Weka_control(min =1, max = 3))
  }
  
  #generate document term matrix for NB
  dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(wordLengths=c(2, Inf), 
                                                                      tokenize = bitrigramtokeniser, 
                                                                      weighting = function(x) weightTfIdf(x, normalize = FALSE),
                                                                      bounds=list(global=c(floor(length(corpus.clean.train)*0.01), floor(length(corpus.clean.train)*.8)))
                                                                      ))
  
  dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(wordLengths=c(2, Inf), 
                                                                    tokenize = bitrigramtokeniser, 
                                                                    weighting = function(x) weightTfIdf(x, normalize = FALSE),
                                                                    bounds=list(global=c(floor(length(corpus.clean.test)*0.01), floor(length(corpus.clean.test)*.8)))
                                                                    ))
  
  # Apply the convert_count function to get final training and testing DTMs
  trainNB <- apply(dtm.train.nb, 2, convert_count)
  testNB <- apply(dtm.test.nb, 2, convert_count)
  
  # train model
  classifier <- naiveBayes(trainNB, df.train$class, laplace = 1)
  
  # save the model
  #save(classifier, file = paste0(LABEL, ".model"))
  
  # predict 
  pred <- predict(classifier, newdata=testNB) 
  table("Predictions"= pred,  "Actual" = df.test$class )
  
  #evaluation and analysis
  conf.mat <- confusionMatrix(pred, df.test$class, positive=LABEL)
  
  header<-paste("\nP", "R", "F1\n", sep = '\t')
  content<-paste(round(conf.mat$byClass[5] ,5), round(conf.mat$byClass[6] ,5), round(conf.mat$byClass[7] ,5), sep = '\t')
  
  cat(header)
  cat(content)
}

#Elastic net, SVM Guassian and linear
  run.lasso <-function(training.data, test.data, LABEL) {

  labels.train <- as.factor(new.enet.corpus(training.data, LABEL))  
  labels.test <- as.factor(new.enet.corpus(test.data, LABEL))  
  
  dtm.train <- nlp.preprocess(training.data$comment)
  dtm.test <- nlp.preprocess(test.data$comment, Terms(dtm.train))
  
  dictionary <- as.data.frame(Terms(dtm.train))
  saveRDS(dictionary, file="dictionary.Rda")
  
  models.path <<- 'combine.models/'
  enet.evaluate(dtm.train, labels.train, dtm.test, labels.test, LABEL)
  
  cat("\nSVM - Radial Basis kernel\n")
  svm.evaluate(dtm.train, labels.train, dtm.test, labels.test, LABEL, kernel='rbf')
  
  cat("\nSVM - linear kernel\n")
  svm.evaluate(dtm.train, labels.train, dtm.test, labels.test, LABEL)
}

run.prediction <- function(data){
  
  output.path = 'combine.outputs/'
  
  #NLP pre-processing
  corpus <- VCorpus(VectorSource(data$comment))
  corpus.processed <- corpus %>%  
    tm_map(content_transformer(tolower)) %>% 
    tm_map(removePunctuation) %>%
    tm_map(stemDocument) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords(kind="en")) %>%
    tm_map(stripWhitespace)
  
  
  #generate document term matrix
  dtm <- DocumentTermMatrix(corpus.processed)
  
  bitrigramtokeniser <- function(x, n) {
    RWeka:::NGramTokenizer(x, RWeka:::Weka_control(min =1, max = 3))
  }
  
  training.dict <- readRDS("dictionary.Rda")
  training.terms <<- as.vector(training.dict[, 1])
  
  dtm.test <- DocumentTermMatrix(corpus.processed, control = list(dictionary=training.terms, 
                                                                  wordLengths=c(2, Inf), 
                                                                  #tokenize = bitrigramtokeniser,
                                                                  weighting = function(x) weightTfIdf(x, normalize = FALSE)
                                                                  #bounds=list(global=c(floor(length(corpus.processed)*0.01), floor(length(corpus.processed)*.8)))
                                                                  ))

  # Apply the convert_count function to get final training and testing DTMs
  #testNB <- apply(dtm.test.nb, 2, convert_count)
  
  #load saved models
  load.models()
  
  # predict using saved models
  env_pred <- as.character(predict(env.model, dtm.test)) 
  env_pred_raw <- predict(env.model, dtm.test, type=("probabilities")) 
  #print (env_pred)
  
  test_mat  <- testmat(training.terms, as.matrix(dtm.test))
  wt_pred <- as.character(predict(wt.model, newx=test_mat, s = "lambda.min", type="class"))
  wt_pred_raw <- predict(wt.model, newx=test_mat, s = "lambda.min", type="response")
  
  saap_pred <- as.character(predict(saap.model, dtm.test))
  saap_pred_raw <- predict(saap.model, dtm.test, type='probabilities')
  #print (saap_pred)
  
  cq_pred <- as.character(predict(cq.model, newx=test_mat, s = "lambda.min", type="class"))
  cq_pred_raw <- predict(cq.model, newx=test_mat, s = "lambda.min", type="response")
  
  #combine predictions and save in correct format
  result_data <- data.frame(stringsAsFactors=FALSE)
  
  comments = data$comment
  max.num.categories = 11
  
  #food.and.parking = apply.dictionaries(comments) 
  for (i in 1:length(saap_pred)) {
    row = data.frame(matrix(NA, 1, max.num.categories))
    
    # declare dummy column headers; required for rbind
    names(row) <-LETTERS[1:length(row)] 
    
    aspects = list()
    if (env_pred[i] != "0") {
      #aspects[length(aspects) + 1] <- env_pred[i]
      aspects[length(aspects) + 1] <- paste(c(env_pred[i], env_pred_raw[i, "environment"]), collapse = " ")
    }
    
    if (wt_pred[i] != "0") {
      #aspects[length(aspects)+1] <- wt_pred[i]
      aspects[length(aspects)+1] <- paste(c(wt_pred[i], wt_pred_raw[i, "1"]), collapse = " ")
    }
    
    if (saap_pred[i] != "0") {
      #aspects[length(aspects)+1] <- saap_pred[i]
      aspects[length(aspects)+1] <- paste(c(saap_pred[i], saap_pred_raw[i, "staff attitude and professionalism"]), collapse = " ")
    }
    
    if (cq_pred[i] != "0") {
      #aspects[length(aspects)+1] <- cq_pred[i]
      aspects[length(aspects)+1] <- paste(c(cq_pred[i], cq_pred_raw[i, "1"]), collapse = " ")
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
      row[1, 2] = 'other 0.10'
    }
    
    result_data <- rbind(result_data, row)
  }
  
  # save data as csv file
  write.table(result_data, file=paste( output.path, 'predictions_155.csv'), row.names=FALSE, col.names = FALSE, sep = ",", na = "", qmethod = c("escape", "double"))
}

run.prediction.2 <- function(training.data, prediction.data){ 
  
  dataset_size = length(data[[1]])
  train_data_count = length(training.data[[1]])
  
  set.seed(seed)
  
  #NLP pre-processing
  corpus <- Corpus(VectorSource(data$comment))
  corpus.processed <- corpus %>%  
    tm_map(content_transformer(tolower)) %>% 
    tm_map(removePunctuation) %>%
    tm_map(stemDocument) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords(kind="en")) %>%
    tm_map(stripWhitespace)
  
  
  #generate document term matrix
  dtm <- DocumentTermMatrix(corpus.processed)
  
  bitrigramtokeniser <- function(x, n) {
    RWeka:::NGramTokenizer(x, RWeka:::Weka_control(min =1, max = 3))
  }
  
  #generate document term matrix for NB
  dtm.test.nb <- DocumentTermMatrix(corpus.processed, control=list(wordLengths=c(2, Inf), 
                                                                   tokenize = bitrigramtokeniser, 
                                                                   weighting = function(x) weightTfIdf(x, normalize = FALSE),
                                                                   bounds=list(global=c(floor(length(corpus.processed)*0.01), floor(length(corpus.processed)*.8)))))
  
  # Apply the convert_count function to get final training and testing DTMs
  testNB <- apply(dtm.test.nb, 2, convert_count)
  
  #load saved models
  load.models()
  
  # predict using saved models
  env_pred <- as.character(predict(env.model, newdata=testNB)) 
  wt_pred <- as.character(predict(wt.model, newdata=testNB))
  saap_pred <- as.character(predict(saap.model, newdata=testNB))
  cq_pred <- as.character(predict(cq.model, newdata=testNB))
  
  env_pred_raw <- predict(env.model, newdata=testNB, type=("raw")) 
  wt_pred_raw <- predict(wt.model, newdata=testNB, type=("raw"))
  saap_pred_raw <- predict(saap.model, newdata=testNB, type=("raw"))
  cq_pred_raw <- predict(cq.model, newdata=testNB, type=("raw"))
  
  print (env_pred_raw)
  
  #combine predictions and save in correct format
  result_data <- data.frame(stringsAsFactors=FALSE)
  
  comments = data$comment
  max.num.categories = 11
  
  #food.and.parking = apply.dictionaries(comments) 
  for (i in 1:length(saap_pred)) {
    row = data.frame(matrix(NA, 1, max.num.categories))
    
    # declare dummy column headers; required for rbind
    names(row) <-LETTERS[1:length(row)] 
    
    aspects = list()
    if (env_pred[i] != "0") {
      #aspects[length(aspects) + 1] <- env_pred[i]
      aspects[length(aspects) + 1] <- paste(c(env_pred[i], env_pred_raw[i, "environment"]), collapse = " ")
    }
    
    if (wt_pred[i] != "0") {
      #aspects[length(aspects)+1] <- wt_pred[i]
      aspects[length(aspects)+1] <- paste(c(wt_pred[i], wt_pred_raw[i, "waiting time"]), collapse = " ")
    }
    
    if (saap_pred[i] != "0") {
      #aspects[length(aspects)+1] <- saap_pred[i]
      aspects[length(aspects)+1] <- paste(c(saap_pred[i], saap_pred_raw[i, "staff attitude and professionalism"]), collapse = " ")
    }
    
    if (cq_pred[i] != "0") {
      #aspects[length(aspects)+1] <- cq_pred[i]
      aspects[length(aspects)+1] <- paste(c(cq_pred[i], cq_pred_raw[i, "care quality"]), collapse = " ")
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
      row[1, 2] = 'other 0.4'
    }
    
    result_data <- rbind(result_data, row)
  }
  
  # save data as csv file
  write.table(result_data, file=paste(output.path, 'predictions_111.csv'), row.names=FALSE, col.names = FALSE, sep = ",", na = "", qmethod = c("escape", "double"))
}


