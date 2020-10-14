#A function to convert human labelled multi-class data to one-against-all labelled data
#For example, when label="waiting time" it will only keep the latter label while assigning 0 to the remaining.
#Assumption column one: text/comment, column two: category/class

new.corpus <- function(corpus, label){
  gold <- vector(mode="numeric", length=length(corpus[[1]]))
  for(i in 1:length(corpus[[1]])){
    for(j in 2:length(corpus)){
      if(grepl(label, as.character(corpus[i,j]))){
        gold[i]<-label
      }
    }
  }
  
  return(gold)
}
