apply.dictionaries <- function(comments, pred) {
  food.dictionary <- list('food', 'canteen', 'canten', 'coffee', 'cofee', 'coffe', 'tea', 'drink', 'drinks')
  parking.dictionary = list('car park', 'car-park', 'carpark', 'parking', 'bicycle')
  
  all.aspects = list()
  index = 1
  
  for (comment in comments) {
    comment.aspects <- list()
    tokens <- NGramTokenizer(comment, Weka_control(min = 1, max = 2))
    food.diff <- setdiff(food.dictionary, tokens)
    parking.diff <- setdiff(parking.dictionary, tokens)
    
    
    if (length(food.dictionary) != length(food.diff)) {
      comment.aspects[[length(comment.aspects) + 1]] <- 'food'
    }
    
    if (length(parking.dictionary) != length(parking.diff)) {
      comment.aspects[[length(comment.aspects) + 1]] <- 'parking'
    }
    
    all.aspects[[index]] <- comment.aspects
    index <- index + 1
      
  }
  
  return (all.aspects)
}
  