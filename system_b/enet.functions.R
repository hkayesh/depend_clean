# function required for enet
chisqTwo <- function(dtm, labels, n_out=2000){
  mat 		<- as.matrix(dtm)
  cat1		<- 	colSums(mat[labels==T,])	  	# total number of times phrase used in cat1 
  cat2		<- 	colSums(mat[labels==F,])	 	# total number of times phrase used in cat2 
  n_cat1		<- 	sum(mat[labels==T,]) - cat1   	# total number of phrases in soft minus cat1
  n_cat2		<- 	sum(mat[labels==F,]) - cat2   	# total number of phrases in hard minus cat2
  
  num 		<- (cat1*n_cat2 - cat2*n_cat1)^2
  den 		<- (cat1 + cat2)*(cat1 + n_cat1)*(cat2 + n_cat2)*(n_cat1 + n_cat2)
  chisq 		<- num/den
  
  chi_order	<- chisq[order(chisq)][1:n_out]   
  mat 		<- mat[, colnames(mat) %in% names(chi_order)]
  
}

testmat <- function(train_mat_cols, test_mat){	
  #train_mat_cols <- colnames(train_mat); test_mat <- as.matrix(test_dtm)
  test_mat 	<- test_mat[, colnames(test_mat) %in% train_mat_cols]
  
  miss_names 	<- train_mat_cols[!(train_mat_cols %in% colnames(test_mat))]
  if(length(miss_names)!=0){
    colClasses  <- rep("numeric", length(miss_names))
    df 			<- read.table(text = '', colClasses = colClasses, col.names = miss_names)
    df[1:nrow(test_mat),] <- 0
    test_mat 	<- cbind(test_mat, df)
  }
  as.matrix(test_mat)
}