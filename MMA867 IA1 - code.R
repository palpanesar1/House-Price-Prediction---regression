library(tidyverse)
library(sqldf)
library(dplyr)
library(ggplot2)
library(tidyr)
library(stringr)
library(readxl)
library(dummies)
library(mice)
library(caret)
library(robustbase)
library(sandwich)
library(lmtest)
library(modelr)
library(broom)
library(estimatr)
library(glmnet)

#______________________________________________________________________DATA IMPORT_______________________________________________

# import the two datasets 1. Train dataset: used to train our linear regression model 2. Test dataset: use created linear regression model to predict house prices in this dataset
train_data = read.csv('C:\\Users\\panes\\Desktop\\Jaspal\\Queens MMA\\MMA 867 - Predictive Modelling\\A1\\a1 - house prices\\train.csv')

predict_data = read.csv('C:\\Users\\panes\\Desktop\\Jaspal\\Queens MMA\\MMA 867 - Predictive Modelling\\A1\\a1 - house prices\\test.csv')


#______________________________________________________________________DATA EXPLORATION & MANIPULATION_____________________________________________________
# add SalesPrice to predict data to ensure we can combine data and work on the datset together to ensure data is unform. we will then split the data later on once done.
# combine the two datasets to ensure uniformity and then deal with missing values, outliers, feature engineering etc.
predict_data$SalePrice <- NA
join_data <- rbind(train_data, predict_data)


head(join_data)
str(join_data)


# analyze the cumulative dataset, there are a variety of missing values that we will need to address...
join_data %>%
  select(everything()) %>%
  summarise_all(funs(sum(is.na(.))))


# need to address NAs, we can first replace NAs with 0 or another value as per data_description file provided for columns such as alley
# first add factor of "None" to applicable columns, this will then be used as part of the factor and replace the NAs

levels <- levels(join_data$Alley)
levels[length(levels)+1] <- "None"
# refactor dataset to include None as a factor level & replace NA with "None"
join_data$Alley <- factor(join_data$Alley, levels = levels)
join_data$Alley[is.na(join_data$Alley)] <- "None"

# repeat re-assignmnet of NAs for other fields based on data_description file
levels <- levels(join_data$GarageQual)
levels[length(levels)+1] <- "None"
join_data$GarageQual <- factor(join_data$GarageQual, levels = levels)
join_data$GarageQual[is.na(join_data$GarageQual)] <- "None"

levels <- levels(join_data$GarageCond)
levels[length(levels)+1] <- "None"
join_data$GarageCond <- factor(join_data$GarageCond, levels = levels)
join_data$GarageCond[is.na(join_data$GarageCond)] <- "None"

levels <- levels(join_data$FireplaceQu)
levels[length(levels)+1] <- "None"
join_data$FireplaceQu <- factor(join_data$FireplaceQu, levels = levels)
join_data$FireplaceQu[is.na(join_data$FireplaceQu)] <- "None"

levels <- levels(join_data$GarageType)
levels[length(levels)+1] <- "None"
join_data$GarageType <- factor(join_data$GarageType, levels = levels)
join_data$GarageType[is.na(join_data$GarageType)] <- "None"

levels <- levels(join_data$GarageYrBlt)
levels[length(levels)+1] <- "None"
join_data$GarageYrBlt <- factor(join_data$GarageYrBlt, levels = levels)
join_data$GarageYrBlt[is.na(join_data$GarageYrBlt)] <- "None"

levels <- levels(join_data$GarageFinish)
levels[length(levels)+1] <- "None"
join_data$GarageFinish <- factor(join_data$GarageFinish, levels = levels)
join_data$GarageFinish[is.na(join_data$GarageFinish)] <- "None"

levels <- levels(join_data$PoolQC)
levels[length(levels)+1] <- "None"
join_data$PoolQC <- factor(join_data$PoolQC, levels = levels)
join_data$PoolQC[is.na(join_data$PoolQC)] <- "None"

levels <- levels(join_data$BsmtExposure)
levels[length(levels)+1] <- "None"
join_data$BsmtExposure <- factor(join_data$BsmtExposure, levels = levels)
join_data$BsmtExposure[is.na(join_data$BsmtExposure)] <- "None"

levels <- levels(join_data$BsmtFinType1)
levels[length(levels)+1] <- "None"
join_data$BsmtFinType1 <- factor(join_data$BsmtFinType1, levels = levels)
join_data$BsmtFinType1[is.na(join_data$BsmtFinType1)] <- "None"

levels <- levels(join_data$BsmtFinType2)
levels[length(levels)+1] <- "None"
join_data$BsmtFinType2 <- factor(join_data$BsmtFinType2, levels = levels)
join_data$BsmtFinType2[is.na(join_data$BsmtFinType2)] <- "None"


levels <- levels(join_data$Fence)
levels[length(levels)+1] <- "None"
join_data$Fence <- factor(join_data$Fence, levels = levels)
join_data$Fence[is.na(join_data$Fence)] <- "None"

levels <- levels(join_data$MiscFeature)
levels[length(levels)+1] <- "None"
join_data$MiscFeature <- factor(join_data$MiscFeature, levels = levels)
join_data$MiscFeature[is.na(join_data$MiscFeature)] <- "None"


levels <- levels(join_data$BsmtQual)
levels[length(levels)+1] <- "None"
join_data$BsmtQual <- factor(join_data$BsmtQual, levels = levels)
join_data$BsmtQual[is.na(join_data$BsmtQual)] <- "None"

levels <- levels(join_data$BsmtCond)
levels[length(levels)+1] <- "None"
join_data$BsmtCond <- factor(join_data$BsmtCond, levels = levels)
join_data$BsmtCond[is.na(join_data$BsmtCond)] <- "None"


# use mean imputation to replace the lotFrontage
join_data$LotFrontage[is.na(join_data$LotFrontage)] = mean(join_data$LotFrontage, na.rm=TRUE)

join_data$BsmtFinSF1[is.na(join_data$BsmtFinSF1)] = 0
join_data$BsmtFinSF2[is.na(join_data$BsmtFinSF2)] = 0
join_data$BsmtUnfSF[is.na(join_data$BsmtUnfSF)] =  0  
join_data$TotalBsmtSF[is.na(join_data$TotalBsmtSF)] =  0 
join_data$BsmtFullBath[is.na(join_data$BsmtFullBath)] =  0 
join_data$BsmtHalfBath[is.na(join_data$BsmtHalfBath)] =  0 
join_data$GarageCars[is.na(join_data$GarageCars)] =  0 
join_data$GarageArea[is.na(join_data$GarageArea)] =  0 


# function to get mode of vector to be used for mode imputation
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}


# replace rest of categorical variables with mode (mode imputation)
join_data$MSZoning[is.na(join_data$MSZoning)] =  getmode(join_data$MSZoning) 
join_data$Utilities[is.na(join_data$Utilities)] =  getmode(join_data$Utilities) 
join_data$MasVnrType[is.na(join_data$MasVnrType)] =  getmode(join_data$MasVnrType) 
join_data$MasVnrArea[is.na(join_data$MasVnrArea)] =  getmode(join_data$MasVnrArea) 
join_data$KitchenQual[is.na(join_data$KitchenQual)] =  getmode(join_data$KitchenQual) 
join_data$Functional[is.na(join_data$Functional)] =  getmode(join_data$Functional) 
join_data$SaleType[is.na(join_data$SaleType)] =  getmode(join_data$SaleType) 
join_data$Exterior1st[is.na(join_data$Exterior1st)] =  getmode(join_data$Exterior1st) 
join_data$Exterior2nd[is.na(join_data$Exterior2nd)] =  getmode(join_data$Exterior2nd) 
join_data$Electrical[is.na(join_data$Electrical)] =  getmode(join_data$Electrical) 


# Sublclass and Month need to be factors
join_data$MSSubClass <- factor(join_data$MSSubClass)   
join_data$MoSold <- factor(join_data$MoSold)

# remove the Id column as this won't help with regression model
Id <- join_data$Id[join_data$Id>=1461]

join_data$Id <- NULL

#remove year Garage built as all are nulls and this should technically be the same as YearlBuilt
join_data$GarageYrBlt <- NULL

# analyze the cumulative dataset, 
join_data %>%
  select(everything()) %>%
  summarise_all(funs(sum(is.na(.))))

str(join_data)


join_data$PropertyAge <- (join_data$YrSold - join_data$YearBuilt)
join_data$Year_SRemodel <- (join_data$YrSold-join_data$YearRemodAdd)

# factorizing year columns
join_data$YearBuilt <- factor(join_data$YearBuilt)
join_data$YearRemodAdd <- factor(join_data$YearRemodAdd)



sample_data <- filter(join_data, !is.na(join_data$SalePrice))
vis1 <- sqldf("select MoSold, AVG(SalePrice) from sample_data GROUP BY MoSold")

vis2 <- sqldf("select Neighborhood, AVG(SalePrice) from sample_data GROUP BY Neighborhood ORDER BY AVG(SalePrice) DESC")
highPN <- c('NoRidge', 'NridgHt','StoneBr', 'Timber', 'Veenker', 'Somerst', 'ClearCr', 'Crawfor','CollgCr', 'Blmngtn','Gilbert', 'NWAmes')


# feature engineering (add new variable based on higher than average home prices)
join_data['NeighborhoodHP'] <- (join_data$Neighborhood %in% highPN) *1



# adding dummy variables for all categorical vs
join_data <- dummy.data.frame(join_data, sep=".")


# visualize the data, check for outliers, linearity between variables, plot density etc.

boxplot(join_data$LotArea)
plot(density(join_data$LotArea))

boxplot(join_data$LotFrontage)
plot(density(join_data$LotFrontage))

boxplot(join_data$TotalBsmtSF)
plot(density(join_data$TotalBsmtSF))

boxplot(join_data$X1stFlrSF)
plot(density(join_data$X1stFlrSF))


bargraph <- ggplot(data=join_data, aes(x=join_data$SaleCondition, y=join_data$SalePrice)) +
  geom_bar(stat="identity")
bargraph


# there are multiple outliers, we replace these using mean imp. later on (on the train_data2 set)



#______________________________________________________________________LINEAR REGRESSION_____________________________________________________
train_data2 <- filter(join_data, !is.na(join_data$SalePrice))
predict_data2 <- filter(join_data, is.na(join_data$SalePrice))

train_data2$LotArea[train_data2$LotArea>100000] <- mean(train_data2$LotArea)
train_data2$LotFrontage[train_data2$LotFrontage>250] <- mean(train_data2$LotFrontage)
train_data2$LotFrontage[train_data2$TotalBsmtSF>4000] <- mean(train_data2$TotalBsmtSF)
train_data2$LotFrontage[train_data2$X1stFlrSF>3000] <- mean(train_data2$X1stFlrSF)

# holdout some of the train data in order to validate our LR model
# Create Training and Test data -
set.seed(100)  # setting seed to reproduce results of random sampling
trainingRowIndex <- sample(1:nrow(train_data2), 0.8*nrow(train_data2))  # row indices for training data
train_data3 <- train_data2[trainingRowIndex, ]  # model training data
test_data3  <- train_data2[-trainingRowIndex, ]   # test data


#regression
reg1 <- lm_robust(log(SalePrice) ~ ., train_data3, se_type="HC2")



prediction1<-exp(predict(reg1, test_data3)) 

percent.errors <- abs((test_data3$SalePrice-prediction1)/test_data3$SalePrice)*100 #calculate absolute percentage errors
mean(percent.errors) #display Mean Absolute Percentage Error (MAPE)


#______________________________________________________________________LASSO (LINEAR) REGRESSION_____________________________________________________

x <- data.matrix(train_data3[, - which(names(train_data3) %in% c('SalePrice'))])
x2 <- data.matrix(test_data3[, - which(names(test_data3) %in% c('SalePrice'))])
y <- log(train_data3$SalePrice)

crossval <-  cv.glmnet(x = x, y = y, alpha = 1)
penalty.lasso <- crossval$lambda.min
lassofit <-glmnet(x = x, y = y, alpha = 1, lambda = penalty.lasso) #estimate the model with the optimal penalty
coef(lassofit) #resultant model coefficients

# predicting the performance on the testing set
lassopredictions <- exp(predict(lassofit, s = penalty.lasso, newx =x2))

percent.errors <- abs((test_data3$SalePrice-lassopredictions)/test_data3$SalePrice)*100 #calculate absolute percentage errors
mean(percent.errors) #display Mean Absolute Percentage Error (MAPE)

#______________________________________________________________________FINAL RESULTS_____________________________________________________

# construct new lasso regression using all results from test data

x <- data.matrix(train_data2[, - which(names(train_data2) %in% c('SalePrice'))])
xfinal <- data.matrix(predict_data2[, - which(names(predict_data2) %in% c('SalePrice'))])
y <- log(train_data2$SalePrice)

crossval <-  cv.glmnet(x = x, y = y, alpha = 1)
penalty.lasso <- crossval$lambda.min
lassofit <-glmnet(x = x, y = y, alpha = 1, lambda = penalty.lasso) #estimate the model with the optimal penalty
coef(lassofit) #resultant model coefficients

# predicting the performance on the testing set
SalePrice <- formatC(as.numeric(exp(predict(lassofit, s = penalty.lasso, newx =xfinal))),digits = 2, format = "f")




#finalprediction <- formatC(as.numeric(lassopredictionsfinal,digits = 2, format = "f")
lassopredictionsfinal <- cbind(Id, SalePrice)

write.csv(lassopredictionsfinal,"C:\\Users\\panes\\Desktop\\Jaspal\\Queens MMA\\MMA 867 - Predictive Modelling\\A1\\a1 - house prices\\prediction.csv", row.names = FALSE)

