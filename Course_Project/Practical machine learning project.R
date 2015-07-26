library(caret)
library(readr)
library(randomForest)

# Config ----
na.tresh <- 100   # max missing values in a collumn to be considered usefull
train.part <- .75

# Functions ----
cleanData <- function(df = data.frame(), na.tresh = numeric()){
        
        # Identify and remove columns with too many NA values
        cleanCol <- colSums(is.na(df))
        cleanCol <- which(cleanCol < na.tresh)
        
        df <- df[,cleanCol] 
        
        # Identify and remove columns with character value
        cleanCol <- lapply (names(df), 
                            function (x) {typeof(df[,x])})
        cleanCol <- which(cleanCol != "character")
        
        # choose only numeric values and the class (last column)
        df <- df[,c(cleanCol, length(df))] 
        
        # Remove timestamps, window number and row number
        df <- df[, -(1:4)]
        df
}

# Read and clean the data ----
rawTrainingData <- read_csv("pml-training.csv")
rawTrainingData <- cleanData(rawTrainingData, na.tresh)

ifelse (sum(is.na(rawTrainingData)),
        "There are missing vaues in the training set!",
        "There are NOT missing vaues in the training set!"
        )

# Training the model ----
tc <- trainControl(## 10-fold CV
        method = "repeatedcv",
        number = 10,
        ## repeated ten times
        repeats = 10) 

set.seed(232323)

inTraining <- createDataPartition(rawTrainingData$classe, p = train.part, list = FALSE)
training   <- rawTrainingData[inTraining, ]
validating <- rawTrainingData[-inTraining, ]

#Exploratory analysis

ggplot(validating, aes(x = roll_belt, y = pitch_belt, color = classe)) + geom_point()


set.seed(232323)

# start.train <- Sys.time() 
# modFitRF   <- train(as.factor(classe) ~ ., method="rf", 
#                                 data = training, trainControl=tc)
# end.train <- Sys.time()
# train.dur <- end.train - start.train

modFitRF <- readRDS("modFitRF_75_training_500_trees.rds")

# Predicting on the training set ----
trainPred    <- predict(modFitRF, training[,-length(training)])
trainCorrect <- trainPred == training$classe
trainAcc     <- sum(trainCorrect)/length(training$classe)

table(training$classe, trainPred)
confusionMatrix(training$classe, trainPred)

# Predicting on the validating set
validPred    <- predict(modFitRF, validating[,-length(validating)])
validCorrect <- validPred == validating$classe
validAcc     <- sum(validCorrect)/length(validating$classe)

table(validating$classe, validPred)
confusionMatrix(validating$classe, validPred)

validWrong <- dim(validating)[1] - sum(validCorrect)

validAccuracy <- sum(validCorrect)/dim(validating)[1]

# Predicting on the validating set ----
rawTestingData  <- read_csv("pml-testing.csv")
testing         <- rawTestingData[,names(training[, -length(training)])]

testing$testPred <- predict(modFitRF, testing)

plot(modFitRF$finalModel)
plot(modFitRF)
#write_csv(testing, "RF_temp_percent_training_cv_3.csv")

#saveRDS(modFitRF, "modFitRF_75_training_500_trees.rds")

