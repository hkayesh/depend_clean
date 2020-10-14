#generate gold data for elastic net and SVMs
new.enet.corpus <- function(corpus, label){
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