

nlp.preprocess <- function(comments, training_terms=NULL) {
  
  #NLP pre-processing
  corpus <- Corpus(VectorSource(comments))
  corpus.processed <- corpus %>%  
    tm_map(content_transformer(tolower)) %>% 
    tm_map(removePunctuation) %>%
    tm_map(stemDocument) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords(kind="en")) %>%
    tm_map(stripWhitespace)
  
  
  bitrigramtokeniser <- function(x, n) {
    RWeka:::NGramTokenizer(x, RWeka:::Weka_control(min = 1, max = 3))
  }
  
  if (is.null(training_terms)) {
    control.options = list(wordLengths=c(2, Inf),
                    tokenize = bitrigramtokeniser,
                    weighting = function(x) weightTfIdf(x, normalize = FALSE),
                    bounds=list(global=c(floor(length(corpus.processed)*0.01), floor(length(corpus.processed)*.8)))
                    )
  }
  else {
    control.options = list( dictionary=training_terms,
                            wordLengths=c(2, Inf),
                           tokenize = bitrigramtokeniser,
                           weighting = function(x) weightTfIdf(x, normalize = FALSE),
                           bounds=list(global=c(floor(length(corpus.processed)*0.01), floor(length(corpus.processed)*.8)))
    )
  }
  dtm <- DocumentTermMatrix(corpus.processed, control = control.options)
  
  return (dtm)
}