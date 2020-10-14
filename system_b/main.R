#The main script to run experiments.

# hide warnings
options(warn = -1)

source("load.libs.R")
source("convert.count.R")
source("load.data.R")
source("new.corpus.R")
source("clean.data.R")
source("enet.functions.R")
source("new.enet.corpus.R")
source("apply.dictionaries.R")
source("load.models.R")
source("nlp.preprocessing.R")
source("svm.classifier.R")
source("enet.classifier.R")
source("nb.classifier.R")
source("run.evaluation.R")
#source("run.training.R")
source("run.prediction.R")
source("run.srft.prediction.R")
load.libs()


option_list = list(
  make_option(c("--train"), type="character", help="--train file_path", metavar="character"),
  make_option(c("--data"), type="character", help="--data file_path", metavar="character"),
  make_option(c("--output"), type="character", help="--output file_path", metavar="character"),
  make_option(c("--dataset"), type="character", default="site-a", help="--dataset type", metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

training.file.path <- opt$train #"files/r_mmhsct_train_111.csv"
data.file.path <- opt$data #"files/r_mmhsct_test_111.csv"
output.path <- opt$output #"files/mmhsct_output_confidence_111.csv"
dataset.type <- opt$dataset #'mmhsct'

training.data <-load.training.data(training.file.path)
prediction.data <-load.data(data.file.path)

training.data$comment <- clean.data(training.data)
prediction.data$comment <- clean.data(prediction.data)
set.seed(111)

models.path <- 'combine.models/'
if (!dir.exists(models.path)) {
    dir.create(models.path)
}

if (dataset.type == 'site-b') {
  run.srft.prediction(training.data, prediction.data, output.path)
} else {
  run.prediction(training.data, prediction.data, output.path)
}


