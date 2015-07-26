# This example is derived from the following url:
# https://topepo.github.io/caret/training.html
#

# Load the data 
library(mlbench)
data(Sonar)
str(Sonar[, 1:10])

# Create data partitions

library(caret)
set.seed(998)
inTraining <- createDataPartition(Sonar$Class, p = .75, list = FALSE)
training <- Sonar[ inTraining,]
testing  <- Sonar[-inTraining,]

# Basic parameters tuning "gradient boosting machine (GBM)" gbm

fitControl <- trainControl(## 10-fold CV
        method = "repeatedcv",
        number = 10,
        ## repeated ten times
        repeats = 10)

set.seed(825)
gbmFit1 <- train(Class ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)
gbmFit1


# Alternate Tuning Grids 

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode = 10)

nrow(gbmGrid)

set.seed(825)
gbmFit2 <- train(Class ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE,
                 ## Now specify the exact models 
                 ## to evaludate:
                 tuneGrid = gbmGrid)
gbmFit2

# Plotting the Resampling Profile

# Simple invokation of the function shows the results 
# for the first performance measure

trellis.par.set(caretTheme())
plot(gbmFit2)

# Other performance metrics can be shown using the metric option

trellis.par.set(caretTheme())
plot(gbmFit2, metric = 'Kappa')

# Other types of plot are also available. See ?plot.train for more details. 
# The code below shows a heatmap of the results:

trellis.par.set(caretTheme())
plot(gbmFit2, metric = "Kappa", plotType = "level",
     scales = list(x = list(rot = 90)))

plot(gbmFit2, plotType = "level",
     scales = list(x = list(rot = 90)))

ggplot(gbmFit2)

# ?xyplot.train 
# xyplot(gbmFit2,
#         metric = "Kappa",
#         type = c("n.trees", "interaction.depth"))

# From these plots, a different set of tuning parameters may be desired.
# To change the final values without starting the whole process again, 
# the update.train can be used to refit the final model. See ?update.train


# Alternative performance metrics and two class Probability

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

set.seed(825)
gbmFit3 <- train(Class ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE,
                 tuneGrid = gbmGrid,
                 ## Specify which metric to optimize
                 metric = "ROC")
gbmFit3

# Extracting Predictions and Class Probabilities

predict(gbmFit3, newdata = head(testing))

# Between-Models

set.seed(825)
svmFit <- train(Class ~ ., data = training,
                method = "svmRadial",
                trControl = fitControl,
                preProc = c("center", "scale"),
                tuneLength = 8,
                metric = "ROC")
svmFit

# Regularized discriminant analysis model fit

set.seed(825)
rdaFit <- train(Class ~ ., data = training,
                method = "rda",
                trControl = fitControl,
                tuneLength = 4,
                metric = "ROC")
rdaFit

# Given these models, can we make statistical statements about their performance differences? 
# To do this, we first collect the resampling results using resamples.

resamps <- resamples(list(GBM = gbmFit3,
                          SVM = svmFit,
                          RDA = rdaFit))
resamps

summary(resamps)

# Plotting the result

trellis.par.set("theme1")
bwplot(resamps, layout = c(3, 1))

trellis.par.set(caretTheme())
dotplot(resamps, metric = "ROC")

trellis.par.set(theme1)
xyplot(resamps, what = "BlandAltman")

# Comparing the models
difValues <- diff(resamps)
difValues

# Fitting models without parameter tuning

fitControl <- trainControl(method = "none", classProbs = TRUE)

set.seed(825)
gbmFit4 <- train(Class ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE,
                 ## Only a single model can be passed to the
                 ## function when no resampling is used:
                 tuneGrid = data.frame(interaction.depth = 4,
                                       n.trees = 100,
                                       shrinkage = .1,
                                       n.minobsinnode = 10),
                 metric = "ROC")
gbmFit4
