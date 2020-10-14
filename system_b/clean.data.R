# Make sure this clean function is equal to Humayun's system
clean.data <- function(x){
  gsub("x0085_", "", x$comment, fixed = TRUE)
} 