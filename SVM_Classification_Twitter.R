# This is the R-script for the classification task of identifying evaluations of political projections in large text corpora, as described in the manuscript: 
# "Beyond sentiment: An algorithmic strategy for identifying evaluations within large text corpora"

# The variables Q1 and Q2 correspond to the two coding decisions as described in the manuscript (pp.17-19) as well as in our coding instruction in the supplementary material:

# Coding decision Q1: Does the snippet contain a future projection?
# Q1 = 0 --> text snippet does not contain a future projection (e.g. "like there is no tomorrow")
# Q1 = 1 --> text snippet does contain a future projection (e.g. "she is going to win these elections")

# Coding decision Q2: Does the potentially evaluative term (PET) evaluate the future projection?
# Q2 = 0 --> PET does not evaluate the projection (e.g. "there are [plenty][+] of possibilities for him to still lose the elections")
# Q2 = 1 --> PET used to evaluate the projection, consistently with marked tendency (e.g. "We will make America [great][+] again") 
# Q2 = 2 --> PET used to evaluate the projection, opposite to marked tendency (e.g. "One cannot be seriously [happy][+] about the prospective Trump victory")
# Q2 = 3 --> PET used to evaluate the projection, with unclear/ambiguous tendency (e.g. "Looks like these elections are going to be [interesting][+].")
# Q2 = 4 --> not clear whether the PET evaluates the projection (e.g."[importantly][+], there are plenty of opportunities for him to lose the upcoming elections")

#import dataset
require(readxl)
Classifications_Total <- read_excel(".../Classifications_Input.xlsx")

#recode missing values in Q1 as non-projections
Classifications_Total$Q1[is.na(Classifications_Total$Q2)] <- 0

# recode missing values in Q2 as non-evaluative statements
Classifications_Total$Q2[is.na(Classifications_Total$Q2)] <- 0

#recode unclear and ambivalent evaluations in Q2 as non-evaluative statements given their small size-number in the annotated corpus
Classifications_Total$Q2[Classifications_Total$Q2 == 3] <- 0
Classifications_Total$Q2[Classifications_Total$Q2 == 4] <- 0

#create corpus object

require(quanteda)
require(quanteda.textmodels)
require(caret)
library(e1071)
library(yardstick)
library(xlsx)
library(quanteda.sentiment)
corp_projections <- corpus(Classifications_Total, text_field = "SNIPPET")
summary(corp_projections, 5)
set.seed(300)

#split train and test datasets from individually coded snippets
id_train <- sample(1:10004, replace = FALSE)
head(id_train, 10)
corp_projections$TEXT_ID <- 1:ndoc(corp_projections)

#pre-processing
corp_2 <- gsub('\\[', '', corp_projections)
corp_3 <- gsub(']][-]',' xxneg',corp_2)
corp_4 <- gsub(']][+]',' xxpos',corp_3)

#create four different dfmts with different pre-processing pipelines
#Stopwords Included & Trigrams Exluded (SITE)
toks_SITE <- tokens(corp_4,remove_punct = TRUE, remove_number = TRUE)
toks_SITE_ngrams <- tokens_ngrams(toks_SITE, n=1:2)
dfmt_SITE_ngrams <- dfm(toks_SITE_ngrams)
dfmt_SITE_trim <- dfm_trim(dfmt_SITE_ngrams, min_termfreq = 5)
dfmt_SITE_trim2 <- dfm_trim(dfmt_SITE_trim, max_termfreq = .99, termfreq_type = "prop")
additional_features <- dfmt_SITE_trim2$Q1 %>% as.matrix()
dfmt_SITE_added <- cbind(dfmt_SITE_trim2, additional_features)

#Stopwords Included & Trigrams Included (SITI)
toks_SITI <- tokens(corp_4,remove_punct = TRUE, remove_number = TRUE)
toks_SITI_ngrams <- tokens_ngrams(toks_SITI, n=1:3)
dfmt_SITI_ngrams <- dfm(toks_SITI_ngrams)
dfmt_SITI_trim <- dfm_trim(dfmt_SITI_ngrams, min_termfreq = 5)
dfmt_SITI_trim2 <- dfm_trim(dfmt_SITI_trim, max_termfreq = .99, termfreq_type = "prop")
additional_features <- dfmt_SITI_trim2$Q1 %>% as.matrix()
dfmt_SITI_added <- cbind(dfmt_SITI_trim2, additional_features)

#Stopwords Excluded & Trigrams Excluded (SETE)
toks_SETE <- tokens(corp_4,remove_punct = TRUE, remove_number = TRUE) %>% 
  tokens_remove(pattern = stopwords("en"))
toks_SETE_ngrams <- tokens_ngrams(toks_SETE, n=1:2)
dfmt_SETE_ngrams <- dfm(toks_SETE_ngrams)
dfmt_SETE_trim <- dfm_trim(dfmt_SETE_ngrams, min_termfreq = 5)
dfmt_SETE_trim2 <- dfm_trim(dfmt_SETE_trim, max_termfreq = .99, termfreq_type = "prop")
additional_features <- dfmt_SETE_trim2$Q1 %>% as.matrix()
dfmt_SETE_added <- cbind(dfmt_SETE_trim2, additional_features)

#Stopwords Excluded & Trigrams Included (SETI)
toks_SETI <- tokens(corp_4,remove_punct = TRUE, remove_number = TRUE) %>% 
  tokens_remove(pattern = stopwords("en"))
toks_SETI_ngrams <- tokens_ngrams(toks_SETI, n=1:3)
dfmt_SETI_ngrams <- dfm(toks_SETI_ngrams)
dfmt_SETI_trim <- dfm_trim(dfmt_SETI_ngrams, min_termfreq = 5)
dfmt_SETI_trim2 <- dfm_trim(dfmt_SETI_trim, max_termfreq = .99, termfreq_type = "prop")
additional_features <- dfmt_SETI_trim2$Q1 %>% as.matrix()
dfmt_SETI_added <- cbind(dfmt_SETI_trim2, additional_features)

#Run 10-Fold Cross Validation with using only Twitter data in the Test Set
#define training set for k-fold cross validation
dfmat_training <- dfm_subset(dfmt_SITI_added, TEXT_ID %in% id_train)

#Randomly shuffle the data
yourData<-dfmat_training[sample(nrow(dfmat_training)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
evalMeasures <- NULL

#Perform 10 fold cross validation
for(i in 1:10){
  #Segment your data by fold using the which() function 
  testIndexes <- which(folds==i ,arr.ind=TRUE)
  testData <- yourData[testIndexes, ]
  trainData <- yourData[-testIndexes, ]
  # Filter the test data to only include rows with "@Twitter" in the "SOURCE" column
  testData <- testData[grepl("@Twitter", testData$SOURCE), ]
  
  if (nrow(testData) == 0) {
    warning("No test data with '@Twitter' in the 'SOURCE' column for fold", i, ". Skipping fold.")
    next
  }
  tmod_svm <- textmodel_svm(trainData, trainData$Q2)
  summary(tmod_svm)
  dfmat_matched <- dfm_match(testData, features = featnames(trainData))
  actual_class <- dfmat_matched$Q2
  predicted_class <- predict(tmod_svm, newdata = dfmat_matched)
  tab_class <- table(predicted_class, actual_class)
  print(tab_class)
  print(confusionMatrix(tab_class, mode = "everything"))
  kthEvaluation <- f_meas(tab_class)
  evalMeasures <- rbind(evalMeasures, kthEvaluation)
  
}
print(evalMeasures)
# Average over all folds
mean(evalMeasures$.estimate)
sd(evalMeasures$.estimate)

#Run 10-Fold Cross Validation with using only media data in the Test Set

#define training set for k-fold cross validation
dfmat_training <- dfm_subset(dfmt_SITI_added, TEXT_ID %in% id_train)

#Randomly shuffle the data
yourData<-dfmat_training[sample(nrow(dfmat_training)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
evalMeasures <- NULL

#Perform 10 fold cross validation
for(i in 1:10){
  #Segment your data by fold using the which() function 
  testIndexes <- which(folds==i ,arr.ind=TRUE)
  testData <- yourData[testIndexes, ]
  trainData <- yourData[-testIndexes, ]
  # Filter the test data to only include rows without "@Twitter" in the "SOURCE" column
  testData <- testData[!grepl("@Twitter", testData$SOURCE), ]
  
  if (nrow(testData) == 0) {
    warning("No test data without '@Twitter' in the 'SOURCE' column for fold", i, ". Skipping fold.")
    next
  }
  tmod_svm <- textmodel_svm(trainData, trainData$Q2)
  summary(tmod_svm)
  dfmat_matched <- dfm_match(testData, features = featnames(trainData))
  actual_class <- dfmat_matched$Q2
  predicted_class <- predict(tmod_svm, newdata = dfmat_matched)
  tab_class <- table(predicted_class, actual_class)
  print(tab_class)
  print(confusionMatrix(tab_class, mode = "everything"))
  kthEvaluation <- f_meas(tab_class)
  evalMeasures <- rbind(evalMeasures, kthEvaluation)
  
}
print(evalMeasures)
# Average over all folds
mean(evalMeasures$.estimate)
sd(evalMeasures$.estimate)

