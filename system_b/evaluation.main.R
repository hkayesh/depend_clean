#The main script to run experiments.

# hide warnings
#options(warn = -1)

source("load.libs.R")
source("convert.count.R")
source("load.data.R")
source("new.corpus.R")
source("clean.data.R")
source("enet.functions.R")
source("new.enet.corpus.R")
#source("apply.dictionaries.R")
source("load.models.R")
source("nlp.preprocessing.R")
source("svm.classifier.R")
source("enet.classifier.R")
source("nb.classifier.R")
source("run.evaluation.R")
source("run.training.R")
source("run.prediction.R")

load.libs()

models.path <<- 'combine.models/'
output.path <<- 'combine.outputs/mmhsct_output_confidence_155.csv'


training.data <-load.training.data()
test.data <- load.test.data()

training.data$comment <- clean.data(training.data)
test.data$comment <- clean.data(test.data)

categories = c('environment', 'waiting time', 'staff attitude and professionalism', 'care quality', 'other')
#for (target in categories) {
  target = 'waiting time'
  #cat (paste("\n\n********************* ", target))
  run.evaluation(training.data, test.data, target)
#}

