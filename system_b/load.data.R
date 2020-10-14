#OBS: replace data PATH

#Load data: uncomment the relevant dataset
#Assumption column one: text/comment, column two: category/class


# MMHSCT/SRFT

load.data <- function(file.path){
  data <- read.csv(file.path, sep = ",", header = FALSE)
  #names(data) <- c("comment", "care quality", "staff attitude and professionalism", "waiting time", "environment", "other")
  
  #data <- read.csv("example_data.csv", sep = ",", header = FALSE)
  names(data) <- c("comment")
  return(data)
}

load.training.data <- function(file.path) {
  data <- read.csv(file.path, sep = ",", header = FALSE)
  #data <- read.csv("mmhsct_segments.csv", sep = ",", header = FALSE)
  names(data) <- c("comment", "topic")
  
  return(data)
}

load.test.data <- function(){
  # Comment-level dataset
  #test_data <- read.csv("/home/hmayun/PycharmProjects/create-dataset-r/evaluation_datasets/r_comment_level_datasets/r_mmhsct_test_33.csv", sep = ",", header = FALSE)
  test_data <- read.csv("/home/hmayun/PycharmProjects/create-dataset-r/evaluation_datasets/r-comment-level-datasets-2/r_mmhsct_test_111.csv", sep = ",", header = FALSE)
  names(test_data) <- c("comment", "care quality", "staff attitude and professionalism", "waiting time", "environment", "other")
  
  return(test_data)
}
