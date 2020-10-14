#function to load models

#model_dir = "mmhsct.models/"
#model_dir = "srft.models/"
model_dir = "combine.models/"


load.models <- function(){
  #load MMHSCT ml-models
  load(file = paste0(model_dir, "environment.model"))
  assign("env.model", classifier, envir = .GlobalEnv) 
  
  
  load(file = paste0(model_dir, "waiting time.model"))
  assign("wt.model", classifier, envir = .GlobalEnv)
  
  
  load(file = paste0(model_dir, "staff attitude and professionalism.model"))
  assign("saap.model", classifier, envir = .GlobalEnv)
  
  
  load(file = paste0(model_dir, "care quality.model"))
  assign("cq.model", classifier, envir = .GlobalEnv)
}

load.model <- function(LABEL) {
  load(file = paste0(model_dir, LABEL, ".model"))
  assign("cls.model", classifier, envir = .GlobalEnv)
  return(cls.model)
}
