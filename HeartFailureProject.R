##Heart Failure Clinical Records Dataset

##First we load our csv dataset into R
dataf <- read.csv("~/Downloads/heart_failure_clinical_records_dataset.csv")

##Packages used:
##install.packages('MASS')
library(MASS)
##install.packages('mgcv')
library(mgcv)
##install.packages('mda')
library(mda)
##install.packages('tree')
library(tree)
##install.packages('matrixStats')
library(matrixStats)
##install.packages('caret')
library(caret)
##install.packages('pROC')
library(pROC)
##install.packages('ggplot2')
library(ggplot2)

##Commentary: from an article link on the dataset repository
##we are given the following logical indicators and/or units of measurement of the data
##anemia: 0 is false & 1 is true
##high blood pressure: follows anemia logic
##diabetes: follows anemia logic
##sex: 0 is woman & 1 is man
##smoking: follows anemia logic
##CPK: mcg/L units
##Ejection fraction: Percentage
##Platelets: kiloplateletes/mL
##Serum creatine: mg/dL
##Serum sodium: mEq/L
##Time: number of days
##DEATH_EVENT: 0 is false & 1 is true

##In this specific script, we explore different types of regression on the data
##Note that since DEATH_EVENT is either 0 or 1, our regression etimates will
##our response variables will be equivalent to percentage of DEATH_EVENT happening

##Let us start by splitting the data into 10 groups
set.seed(5)
groups <- split(dataf, sample(1:10, nrow(dataf), replace = T))
##we will perform cross-validation for each model on the 8 first groups
##the remaining two groups will consist of our test sets

##let us create our error metrics
##our first model is a simple linear regression on all predictors
simplelinear.RMSE <- c()
##our second model is a also a simple linear regression, but,
##this time, we only include the predictors with high significance
summary(lm(DEATH_EVENT~., data = dataf))
anova(lm(DEATH_EVENT~., data = dataf))
##Note those predictors are age, ejection_fraction, serum_creatinine,
##& time
revisedlinear.RMSE <- c()
##our third model is generalized linear model, and since we are working
##with binary {0,1} results we must choose the bernouilli distribution
generlinear.model <- glm(DEATH_EVENT~., family = binomial(link = "logit"), data = dataf)
stepAIC(generlinear.model, direction = 'both')
##Note the predictors used for the generalized linear model are age, 
##ejection_fraction, serum_creatinine, serum_sodium, & time
generlinear.RMSE <- c()
##our fourth model is a generalized additive model
generaddit.model <- gam(DEATH_EVENT ~ s(age) + anaemia + s(creatinine_phosphokinase) + diabetes + s(ejection_fraction) + high_blood_pressure + s(platelets) + s(serum_creatinine) + s(serum_sodium) + sex + smoking + s(time), data = dataf, method = 'REML')
summary(generaddit.model)
##Note we use only the predictors that are significant; for our 
##generalized additive model, they are s(age), s(creatinine_phosphokinase),
##s(ejection_fraction), s(serum_creatinine), & s(time)
generaddit.RMSE <- c()
##our fifth model is a linear discriminant analysis model
lindiscr.RMSE <- c()
##our sixth model is a quadratic discriminant analysis model
quaddiscr.RMSE <- c()
##our seventh model is a mixed discriminant analysis model
mixeddiscr.RMSE <- c()
##our eigth model is a flexible discriminant analysis model
flexdiscr.RMSE <- c()
##our ninth model is a simple decision tree model
decisiontree.RMSE <- c()
##our tenth model is a pruned version of our ninth models
prunedectree.RMSE <- c()
##our last model is a K nearest neighbours model
knearest.RMSE <- c()


##1st fold-out training
trainset <- data.frame()
for (i in 2:8){
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ ., data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[1]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[1]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[1]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[1]]))

generlinear.model <- glm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + serum_sodium + time, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[1]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[1]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine) + s(time), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[1]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[1]]))

lindiscr.model <- lda(DEATH_EVENT ~., data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[1]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[1]]))

quaddiscr.model <- qda(DEATH_EVENT ~., data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[1]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[1]]))

mixeddiscr.model <- mda(DEATH_EVENT ~., data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[1]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[1]]))

flexdiscr.model <- fda(DEATH_EVENT ~., data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[1]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[1]]))

decisiontree.model <- tree(DEATH_EVENT ~., data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[1]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[1]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[1]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[1]]))


##2nd fold-out training
trainset <- data.frame()
for (i in c(1,3:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ ., data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[2]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[2]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[2]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[2]]))

generlinear.model <- glm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + serum_sodium + time, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[2]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[2]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine) + s(time), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[2]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[2]]))

lindiscr.model <- lda(DEATH_EVENT ~., data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[2]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[2]]))

quaddiscr.model <- qda(DEATH_EVENT ~., data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[2]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[2]]))

mixeddiscr.model <- mda(DEATH_EVENT ~., data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[2]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[2]]))

flexdiscr.model <- fda(DEATH_EVENT ~., data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[2]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[2]]))

decisiontree.model <- tree(DEATH_EVENT ~., data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[2]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[2]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[2]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[2]]))


##3rd fold-out training
trainset <- data.frame()
for (i in c(1:2,4:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ ., data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[3]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[3]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[3]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[3]]))

generlinear.model <- glm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + serum_sodium + time, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[3]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[3]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine) + s(time), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[3]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[3]]))

lindiscr.model <- lda(DEATH_EVENT ~., data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[3]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[3]]))

quaddiscr.model <- qda(DEATH_EVENT ~., data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[3]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[3]]))

mixeddiscr.model <- mda(DEATH_EVENT ~., data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[3]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[3]]))

flexdiscr.model <- fda(DEATH_EVENT ~., data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[3]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[3]]))

decisiontree.model <- tree(DEATH_EVENT ~., data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[3]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[3]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[3]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[3]]))


##4th fold-out training
trainset <- data.frame()
for (i in c(1:3,5:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ ., data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[4]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[4]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[4]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[4]]))

generlinear.model <- glm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + serum_sodium + time, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[4]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[4]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine) + s(time), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[4]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[4]]))

lindiscr.model <- lda(DEATH_EVENT ~., data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[4]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[4]]))

quaddiscr.model <- qda(DEATH_EVENT ~., data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[4]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[4]]))

mixeddiscr.model <- mda(DEATH_EVENT ~., data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[4]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[4]]))

flexdiscr.model <- fda(DEATH_EVENT ~., data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[4]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[4]]))

decisiontree.model <- tree(DEATH_EVENT ~., data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[4]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[4]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[4]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[4]]))


##5th fold-out training
trainset <- data.frame()
for (i in c(1:4,6:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ ., data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[5]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[5]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[5]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[5]]))

generlinear.model <- glm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + serum_sodium + time, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[5]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[5]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine) + s(time), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[5]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[5]]))

lindiscr.model <- lda(DEATH_EVENT ~., data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[5]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[5]]))

quaddiscr.model <- qda(DEATH_EVENT ~., data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[5]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[5]]))

mixeddiscr.model <- mda(DEATH_EVENT ~., data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[5]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[5]]))

flexdiscr.model <- fda(DEATH_EVENT ~., data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[5]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[5]]))

decisiontree.model <- tree(DEATH_EVENT ~., data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[5]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[5]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[5]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[5]]))


##6th fold-out training
trainset <- data.frame()
for (i in c(1:5,7:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ ., data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[6]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[6]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[6]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[6]]))

generlinear.model <- glm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + serum_sodium + time, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[6]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[6]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine) + s(time), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[6]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[6]]))

lindiscr.model <- lda(DEATH_EVENT ~., data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[6]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[6]]))

quaddiscr.model <- qda(DEATH_EVENT ~., data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[6]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[6]]))

mixeddiscr.model <- mda(DEATH_EVENT ~., data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[6]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[6]]))

flexdiscr.model <- fda(DEATH_EVENT ~., data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[6]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[6]]))

decisiontree.model <- tree(DEATH_EVENT ~., data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[6]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[6]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[6]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[6]]))


##7th fold-out training
trainset <- data.frame()
for (i in c(1:6, 8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ ., data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[7]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[7]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[7]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[7]]))

generlinear.model <- glm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + serum_sodium + time, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[7]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[7]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine) + s(time), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[7]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[7]]))

lindiscr.model <- lda(DEATH_EVENT ~., data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[7]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[7]]))

quaddiscr.model <- qda(DEATH_EVENT ~., data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[7]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[7]]))

mixeddiscr.model <- mda(DEATH_EVENT ~., data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[7]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[7]]))

flexdiscr.model <- fda(DEATH_EVENT ~., data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[7]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[7]]))

decisiontree.model <- tree(DEATH_EVENT ~., data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[7]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[7]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[7]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[7]]))


##8th fold-out training
trainset <- data.frame()
for (i in c(1:7)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ ., data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[8]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[8]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[8]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[8]]))

generlinear.model <- glm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + serum_sodium + time, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[8]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[8]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine) + s(time), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[8]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[8]]))

lindiscr.model <- lda(DEATH_EVENT ~., data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[8]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[8]]))

quaddiscr.model <- qda(DEATH_EVENT ~., data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[8]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[8]]))

mixeddiscr.model <- mda(DEATH_EVENT ~., data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[8]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[8]]))

flexdiscr.model <- fda(DEATH_EVENT ~., data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[8]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[8]]))

decisiontree.model <- tree(DEATH_EVENT ~., data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[8]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[8]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[8]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[8]]))


##Model comparison over training cross-validation
sum(simplelinear.RMSE)/8
sum(revisedlinear.RMSE)/8
sum(generlinear.RMSE)/8
sum(generaddit.RMSE)/8
sum(lindiscr.RMSE)/8
sum(quaddiscr.RMSE)/8
sum(mixeddiscr.RMSE)/8
sum(flexdiscr.RMSE)/8
sum(decisiontree.RMSE)/8
sum(prunedectree.RMSE)/8

##Conclusions from Model Comparision over Training Cross-Validation

##We can note that the Revised Linear Regression has an improvement
##over the Simple Linear Regression, when it comes to Error Rate. 
##The Best Discriminant Analysis Types are Linear and Flexible,
##which is somewhat predictable for Linear over Quadratic when taking
##into account the small size of the dataset. Surprisingly, the Pruned 
##Tree is as good as most models, even with with the relatively low 
##amount of predictors it uses. However, the Generalized Additive Model
##seems to be the best model so far, being the only model with mean
##error under 0.065 for the cross-validation comparison

##Applications over Test set (groups 9 & 10)

trainset <- data.frame()
for (i in c(1:8)) {
  trainset <- rbind(trainset, groups[[i]])
}
testset <- data.frame()
for (i in c(9,10)) {
  testset <- rbind(testset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ ., data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.test <- predict(simplelinear.model, testset)
simplelinear.test
simplelinear.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - simplelinear.test)^2))/nrow(testset)

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.test <- predict(revisedlinear.model, testset)
revisedlinear.test
revisedlinear.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - revisedlinear.test)^2))/nrow(testset)

generlinear.model <- glm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + serum_sodium + time, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.test <- predict(generlinear.model, testset, type = 'response')
generlinear.test
generlinear.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - generlinear.test)^2))/nrow(testset)

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine) + s(time), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.test <- predict(generaddit.model, testset, type = 'response')
generaddit.test
generaddit.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - generaddit.test)^2))/nrow(testset)

lindiscr.model <- lda(DEATH_EVENT ~., data = trainset)
lindiscr.model
lindiscr.test <- predict(lindiscr.model, testset, type = 'response')
lindiscr.test <- lindiscr.test$posterior[,2]
lindiscr.test
lindiscr.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - lindiscr.test)^2))/nrow(testset)

quaddiscr.model <- qda(DEATH_EVENT ~., data = trainset)
quaddiscr.model
quaddiscr.test <- predict(quaddiscr.model, testset, type = 'response')
quaddiscr.test <- quaddiscr.test$posterior[,2]
quaddiscr.test
quaddiscr.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - quaddiscr.test)^2))/nrow(testset)

mixeddiscr.model <- mda(DEATH_EVENT ~., data = trainset)
mixeddiscr.model
mixeddiscr.test <- predict(mixeddiscr.model, testset, type = 'posterior')
mixeddiscr.test <- mixeddiscr.test[,2]
mixeddiscr.test
mixeddiscr.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - mixeddiscr.test)^2))/nrow(testset)

flexdiscr.model <- fda(DEATH_EVENT ~., data = trainset)
flexdiscr.model
flexdiscr.test <- predict(flexdiscr.model, testset, type = 'posterior')
flexdiscr.test <- flexdiscr.test[,2]
flexdiscr.test
flexdiscr.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - flexdiscr.test)^2))/nrow(testset)

decisiontree.model <- tree(DEATH_EVENT ~., data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.test <- predict(decisiontree.model, testset, type = 'vector')
decisiontree.test
decisiontree.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - decisiontree.test)^2))/nrow(testset)

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.test <- predict(prunedectree.model, testset, type = 'vector')
prunedectree.test
prunedectree.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - prunedectree.test)^2))/nrow(testset)

##Model Comparison over Test Set
lindiscr.test.RMSE
revisedlinear.test.RMSE
generlinear.test.RMSE
generaddit.test.RMSE
lindiscr.test.RMSE
quaddiscr.test.RMSE
mixeddiscr.test.RMSE
flexdiscr.test.RMSE
decisiontree.test.RMSE
prunedectree.test.RMSE

##Conclusions from Testing

##Most Model Conclusions seem in line with our comments from 
##earlier cross-validation. We still note that the decision trees
##seem like a less reliable option compared to other models now.
##We can also notice there is very little difference between linear
##models now (with the Simple Linear, Revised Linear and Generalized
##Linear Models all hovering around 0.044). We can note that only 
##Quadratic Discriminant is over the 0.05 mark (again, likely due
##to the effect of a small dataset on Quadratic Discrimnant Analysis)
##Finally, similarly to our Cross-Validation experiments, the 
##Generalized Additive Model has best performance with a Root Mean
##Square Error which rounds up to 4.1% with the Mixed Discriminant
##Analysis Model with 4.3% and all others equal or above 4.4%.

##However we were building models which would give us the percentage 
##of dying from heart failure
##One of the issues is that the models above are purely analytical
##Since we can not predict the follow-up period duration when 
##patients are admitted, we cannot predict the actual risk of
##passing away from heart failure. We do however know that the longer
##the follow-up is, the better the chance of survival is.

##Then how about focusing on a purely predictive model ?
##Let us try all this again with the follow-up period to build our
##analytical models

##Let us reset our error metrics
simplelinear.RMSE <- c()
##Once again, we only include the predictors with high significance
summary(lm(DEATH_EVENT~.-time, data = dataf))
anova(lm(DEATH_EVENT~.-time, data = dataf))
##Note those predictors are age, ejection_fraction, & serum_creatinine
revisedlinear.RMSE <- c()
##Once again, for the Generalized Linear Model
generlinear.model <- glm(DEATH_EVENT~.-time, family = binomial(link = "logit"), data = dataf)
stepAIC(generlinear.model, direction = 'both')
##Note now the predictors are age, anaemia, creatinine_phosphokinase,
##ejection_fraction, high_blood_pressure, serum_creatinine, & serum_sodium
generlinear.RMSE <- c()
##Once again, for the Generalized Additive Model
generaddit.model <- gam(DEATH_EVENT ~ s(age) + anaemia + s(creatinine_phosphokinase) + diabetes + s(ejection_fraction) + high_blood_pressure + s(platelets) + s(serum_creatinine) + s(serum_sodium) + sex + smoking, data = dataf, method = 'REML')
summary(generaddit.model)
##Note we use only the predictors that are significant; for our 
##generalized additive model, they are s(age), s(creatinine_phosphokinase),
##s(ejection_fraction), & s(serum_creatinine)
generaddit.RMSE <- c()
lindiscr.RMSE <- c()
quaddiscr.RMSE <- c()
mixeddiscr.RMSE <- c()
flexdiscr.RMSE <- c()
decisiontree.RMSE <- c()
prunedectree.RMSE <- c()

##1st fold-out training
trainset <- data.frame()
for (i in 2:8){
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[1]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[1]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[1]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[1]]))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[1]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[1]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[1]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[1]]))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[1]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[1]]))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[1]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[1]]))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[1]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[1]]))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[1]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[1]]))

decisiontree.model <- tree(DEATH_EVENT ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[1]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[1]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[1]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[1]]))


##2nd fold-out training
trainset <- data.frame()
for (i in c(1,3:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[2]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[2]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[2]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[2]]))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[2]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[2]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[2]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[2]]))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[2]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[2]]))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[2]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[2]]))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[2]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[2]]))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[2]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[2]]))

decisiontree.model <- tree(DEATH_EVENT ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[2]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[2]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[2]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[2]]))


##3rd fold-out training
trainset <- data.frame()
for (i in c(1:2,4:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[3]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[3]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[3]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[3]]))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[3]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[3]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[3]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[3]]))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[3]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[3]]))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[3]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[3]]))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[3]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[3]]))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[3]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[3]]))

decisiontree.model <- tree(DEATH_EVENT ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[3]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[3]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[3]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[3]]))


##4th fold-out training
trainset <- data.frame()
for (i in c(1:3,5:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[4]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[4]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[4]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[4]]))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[4]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[4]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[4]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[4]]))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[4]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[4]]))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[4]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[4]]))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[4]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[4]]))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[4]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[4]]))

decisiontree.model <- tree(DEATH_EVENT ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[4]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[4]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[4]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[4]]))


##5th fold-out training
trainset <- data.frame()
for (i in c(1:4,6:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[5]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[5]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[5]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[5]]))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[5]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[5]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[5]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[5]]))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[5]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[5]]))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[5]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[5]]))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[5]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[5]]))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[5]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[5]]))

decisiontree.model <- tree(DEATH_EVENT ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[5]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[5]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[5]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[5]]))


##6th fold-out training
trainset <- data.frame()
for (i in c(1:5,7:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[6]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[6]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[6]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[6]]))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[6]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[6]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[6]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[6]]))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[6]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[6]]))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[6]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[6]]))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[6]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[6]]))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[6]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[6]]))

decisiontree.model <- tree(DEATH_EVENT ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[6]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[6]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[6]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[6]]))


##7th fold-out training
trainset <- data.frame()
for (i in c(1:6, 8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[7]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[7]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[7]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[7]]))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[7]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[7]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[7]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[7]]))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[7]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[7]]))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[7]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[7]]))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[7]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[7]]))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[7]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[7]]))

decisiontree.model <- tree(DEATH_EVENT ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[7]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[7]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[7]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[7]]))


##8th fold-out training
trainset <- data.frame()
for (i in c(1:7)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[8]])
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[8]]))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[8]])
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[8]]))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[8]], type = 'response')
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[8]]))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[8]], type = 'response')
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[8]]))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[8]], type = 'response')
lindiscr.valid <- lindiscr.valid$posterior[,2]
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[8]]))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[8]], type = 'response')
quaddiscr.valid <- quaddiscr.valid$posterior[,2]
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[8]]))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[8]], type = 'posterior')
mixeddiscr.valid <- mixeddiscr.valid[,2]
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[8]]))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[8]], type = 'posterior')
flexdiscr.valid <- flexdiscr.valid[,2]
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[8]]))

decisiontree.model <- tree(DEATH_EVENT ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[8]], type = 'vector')
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[8]]))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[8]], type = 'vector')
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[8]]))


##Model comparison over training cross-validation
sum(simplelinear.RMSE)/8
sum(revisedlinear.RMSE)/8
sum(generlinear.RMSE)/8
sum(generaddit.RMSE)/8
sum(lindiscr.RMSE)/8
sum(quaddiscr.RMSE)/8
sum(mixeddiscr.RMSE)/8
sum(flexdiscr.RMSE)/8
sum(decisiontree.RMSE)/8
sum(prunedectree.RMSE)/8

##Conclusions from Model Comparision over Training Cross-Validation
##We note that our error rates are higher compared to previously; this
##is because the longer a patient's follow-up period, the better chance
##of survival is. However, some models seem less prone to error than
## all of our models; our Revised Linear, General Additive, & Pruned
##Decision Tree stand out from the rest (Average Cross-Validated RMSE of
##7.8% or less). Let us do this again on the Test to verify these results.

##Applications over Test set (groups 9 & 10)

trainset <- data.frame()
for (i in c(1:8)) {
  trainset <- rbind(trainset, groups[[i]])
}
testset <- data.frame()
for (i in c(9,10)) {
  testset <- rbind(testset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.test <- predict(simplelinear.model, testset)
simplelinear.test
simplelinear.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - simplelinear.test)^2))/nrow(testset)

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.test <- predict(revisedlinear.model, testset)
revisedlinear.test
revisedlinear.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - revisedlinear.test)^2))/nrow(testset)

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.test <- predict(generlinear.model, testset, type = 'response')
generlinear.test
generlinear.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - generlinear.test)^2))/nrow(testset)

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.test <- predict(generaddit.model, testset, type = 'response')
generaddit.test
generaddit.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - generaddit.test)^2))/nrow(testset)

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.test <- predict(lindiscr.model, testset, type = 'response')
lindiscr.test <- lindiscr.test$posterior[,2]
lindiscr.test
lindiscr.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - lindiscr.test)^2))/nrow(testset)

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.test <- predict(quaddiscr.model, testset, type = 'response')
quaddiscr.test <- quaddiscr.test$posterior[,2]
quaddiscr.test
quaddiscr.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - quaddiscr.test)^2))/nrow(testset)

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.test <- predict(mixeddiscr.model, testset, type = 'posterior')
mixeddiscr.test <- mixeddiscr.test[,2]
mixeddiscr.test
mixeddiscr.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - mixeddiscr.test)^2))/nrow(testset)

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.test <- predict(flexdiscr.model, testset, type = 'posterior')
flexdiscr.test <- flexdiscr.test[,2]
flexdiscr.test
flexdiscr.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - flexdiscr.test)^2))/nrow(testset)

decisiontree.model <- tree(DEATH_EVENT ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.test <- predict(decisiontree.model, testset, type = 'vector')
decisiontree.test
decisiontree.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - decisiontree.test)^2))/nrow(testset)

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.test <- predict(prunedectree.model, testset, type = 'vector')
prunedectree.test
prunedectree.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - prunedectree.test)^2))/nrow(testset)

##Model Comparison over Test Set
lindiscr.test.RMSE
revisedlinear.test.RMSE
generlinear.test.RMSE
generaddit.test.RMSE
lindiscr.test.RMSE
quaddiscr.test.RMSE
mixeddiscr.test.RMSE
flexdiscr.test.RMSE
decisiontree.test.RMSE
prunedectree.test.RMSE

##Conclusions from Testing
##We can note an inmprovement for all models here with the worst
##being the Original Decision Tree with a 5.99% RMSE. While many
##of our models are close to its RMSE, the best model we have
##is our Generalized Additive model, since it had a low error score
##in our cross-validation as well as in our test phase (others either
##have a great test score, with unremarkable cross-validation results,
##or great cross-validation results with a test score in the same range
##as other models)

##However what we get our approximate percentages of dying from Heart
##Failure according to biological indicators used as predictors (they are
##approximate because some cases may give us negative probabilities or 
##probabilities above 100% [which could be categorised as impossible to
##die from Heart Failure or death from Heart Failure inevitable respectively])
##Thus, instead of relying on these approximate scores, what if we tried 
##classification instead.

##Classification section:

##Set up error metrics
simplelinear.RMSE <- c()
simplelinear.accur <- c()
simplelinear.precis <- c()
simplelinear.recall <- c()
simplelinear.F1score <- c()
simplelinear.AUCsc <- c()
simplelinear.ROCcurv <- vector(mode = 'list', length = 8)
simplelinear.ConfMat <- vector(mode = 'list', length = 8)

revisedlinear.RMSE <- c()
revisedlinear.accur <- c()
revisedlinear.precis <- c()
revisedlinear.recall <- c()
revisedlinear.F1score <- c()
revisedlinear.AUCsc <- c()
revisedlinear.ROCcurv <- vector(mode = 'list', length = 8)
revisedlinear.ConfMat <- vector(mode = 'list', length = 8)

generlinear.RMSE <- c()
generlinear.accur <- c()
generlinear.precis <- c()
generlinear.recall <- c()
generlinear.F1score <- c()
generlinear.AUCsc <- c()
generlinear.ROCcurv <- vector(mode = 'list', length = 8)
generlinear.ConfMat <- vector(mode = 'list', length = 8)

generaddit.RMSE <- c()
generaddit.accur <- c()
generaddit.precis <- c()
generaddit.recall <- c()
generaddit.F1score <- c()
generaddit.AUCsc <- c()
generaddit.ROCcurv <- vector(mode = 'list', length = 8)
generaddit.ConfMat <- vector(mode = 'list', length = 8)

lindiscr.RMSE <- c()
lindiscr.accur <- c()
lindiscr.precis <- c()
lindiscr.recall <- c()
lindiscr.F1score <- c()
lindiscr.AUCsc <- c()
lindiscr.ROCcurv <- vector(mode = 'list', length = 8)
lindiscr.ConfMat <- vector(mode = 'list', length = 8)

quaddiscr.RMSE <- c()
quaddiscr.accur <- c()
quaddiscr.precis <- c()
quaddiscr.recall <- c()
quaddiscr.F1score <- c()
quaddiscr.AUCsc <- c()
quaddiscr.ROCcurv <- vector(mode = 'list', length = 8)
quaddiscr.ConfMat <- vector(mode = 'list', length = 8)

mixeddiscr.RMSE <- c()
mixeddiscr.accur <- c()
mixeddiscr.precis <- c()
mixeddiscr.recall <- c()
mixeddiscr.F1score <- c()
mixeddiscr.AUCsc <- c()
mixeddiscr.ROCcurv <- vector(mode = 'list', length = 8)
mixeddiscr.ConfMat <- vector(mode = 'list', length = 8)

flexdiscr.RMSE <- c()
flexdiscr.accur <- c()
flexdiscr.precis <- c()
flexdiscr.recall <- c()
flexdiscr.F1score <- c()
flexdiscr.AUCsc <- c()
flexdiscr.ROCcurv <- vector(mode = 'list', length = 8)
flexdiscr.ConfMat <- vector(mode = 'list', length = 8)

decisiontree.RMSE <- c()
decisiontree.accur <- c()
decisiontree.precis <- c()
decisiontree.recall <- c()
decisiontree.F1score <- c()
decisiontree.AUCsc <- c()
decisiontree.ROCcurv <- vector(mode = 'list', length = 8)
decisiontree.ConfMat <- vector(mode = 'list', length = 8)

prunedectree.RMSE <- c()
prunedectree.accur <- c()
prunedectree.precis <- c()
prunedectree.recall <- c()
prunedectree.F1score <- c()
prunedectree.AUCsc <- c()
prunedectree.ROCcurv <- vector(mode = 'list', length = 8)
prunedectree.ConfMat <- vector(mode = 'list', length = 8)

##1st fold-out training
trainset <- data.frame()
for (i in 2:8){
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[1]])
for (j in 1:length(simplelinear.valid)) {
  if(simplelinear.valid[j] < 0.5){
    simplelinear.valid[j] <- 0
  }
  if(simplelinear.valid[j] >= 0.5){
    simplelinear.valid[j] <- 1
  }
}
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[1]]))
simplelinear.ConfMat[[1]] <- confusionMatrix(factor(simplelinear.valid), factor(groups[[1]]$DEATH_EVENT))
simplelinear.accur <- c(simplelinear.accur, simplelinear.ConfMat[[1]]$overall[1])
simplelinear.precis <- c(simplelinear.precis, simplelinear.ConfMat[[1]]$byClass[5])
simplelinear.recall <- c(simplelinear.recall, simplelinear.ConfMat[[1]]$byClass[6])
simplelinear.F1score <- c(simplelinear.F1score, simplelinear.ConfMat[[1]]$byClass[7])
simplelinear.ConfMat[[1]] <- simplelinear.ConfMat[[1]]$table
simplelinear.ROCcurv[[1]] <- roc(groups[[1]]$DEATH_EVENT, simplelinear.valid)
simplelinear.AUCsc <- c(simplelinear.AUCsc, auc(groups[[1]]$DEATH_EVENT, simplelinear.valid))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[1]])
for (j in 1:length(revisedlinear.valid)) {
  if(revisedlinear.valid[j] < 0.5){
    revisedlinear.valid[j] <- 0
  }
  if(revisedlinear.valid[j] >= 0.5){
    revisedlinear.valid[j] <- 1
  }
}
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[1]]))
revisedlinear.ConfMat[[1]] <- confusionMatrix(factor(revisedlinear.valid), factor(groups[[1]]$DEATH_EVENT))
revisedlinear.accur <- c(revisedlinear.accur, revisedlinear.ConfMat[[1]]$overall[1])
revisedlinear.precis <- c(revisedlinear.precis, revisedlinear.ConfMat[[1]]$byClass[5])
revisedlinear.recall <- c(revisedlinear.recall, revisedlinear.ConfMat[[1]]$byClass[6])
revisedlinear.F1score <- c(revisedlinear.F1score, revisedlinear.ConfMat[[1]]$byClass[7])
revisedlinear.ConfMat[[1]] <- revisedlinear.ConfMat[[1]]$table
revisedlinear.ROCcurv[[1]] <- roc(groups[[1]]$DEATH_EVENT, revisedlinear.valid)
revisedlinear.AUCsc <- c(revisedlinear.AUCsc, auc(groups[[1]]$DEATH_EVENT, revisedlinear.valid))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[1]], type = 'response')
for (j in 1:length(generlinear.valid)) {
  if(generlinear.valid[j] < 0.5){
    generlinear.valid[j] <- 0
  }
  if(generlinear.valid[j] >= 0.5){
    generlinear.valid[j] <- 1
  }
}
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[1]]))
generlinear.ConfMat[[1]] <- confusionMatrix(factor(generlinear.valid), factor(groups[[1]]$DEATH_EVENT))
generlinear.accur <- c(generlinear.accur, generlinear.ConfMat[[1]]$overall[1])
generlinear.precis <- c(generlinear.precis, generlinear.ConfMat[[1]]$byClass[5])
generlinear.recall <- c(generlinear.recall, generlinear.ConfMat[[1]]$byClass[6])
generlinear.F1score <- c(generlinear.F1score, generlinear.ConfMat[[1]]$byClass[7])
generlinear.ConfMat[[1]] <- generlinear.ConfMat[[1]]$table
generlinear.ROCcurv[[1]] <- roc(groups[[1]]$DEATH_EVENT, generlinear.valid)
generlinear.AUCsc <- c(generlinear.AUCsc, auc(groups[[1]]$DEATH_EVENT, generlinear.valid))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[1]], type = 'response')
for (j in 1:length(generaddit.valid)) {
  if(generaddit.valid[j] < 0.5){
    generaddit.valid[j] <- 0
  }
  if(generaddit.valid[j] >= 0.5){
    generaddit.valid[j] <- 1
  }
}
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[1]]))
generaddit.ConfMat[[1]] <- confusionMatrix(factor(generaddit.valid), factor(groups[[1]]$DEATH_EVENT))
generaddit.accur <- c(generaddit.accur, generaddit.ConfMat[[1]]$overall[1])
generaddit.precis <- c(generaddit.precis, generaddit.ConfMat[[1]]$byClass[5])
generaddit.recall <- c(generaddit.recall, generaddit.ConfMat[[1]]$byClass[6])
generaddit.F1score <- c(generaddit.F1score, generaddit.ConfMat[[1]]$byClass[7])
generaddit.ConfMat[[1]] <- generaddit.ConfMat[[1]]$table
generaddit.ROCcurv[[1]] <- roc(groups[[1]]$DEATH_EVENT, generaddit.valid)
generaddit.AUCsc <- c(generaddit.AUCsc, auc(groups[[1]]$DEATH_EVENT, generaddit.valid))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[1]], type = 'response')
lindiscr.valid <- as.numeric(lindiscr.valid$class)-1
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[1]]))
lindiscr.ConfMat[[1]] <- confusionMatrix(factor(lindiscr.valid), factor(groups[[1]]$DEATH_EVENT))
lindiscr.accur <- c(lindiscr.accur, lindiscr.ConfMat[[1]]$overall[1])
lindiscr.precis <- c(lindiscr.precis, lindiscr.ConfMat[[1]]$byClass[5])
lindiscr.recall <- c(lindiscr.recall, lindiscr.ConfMat[[1]]$byClass[6])
lindiscr.F1score <- c(lindiscr.F1score, lindiscr.ConfMat[[1]]$byClass[7])
lindiscr.ConfMat[[1]] <- lindiscr.ConfMat[[1]]$table
lindiscr.ROCcurv[[1]] <- roc(groups[[1]]$DEATH_EVENT, lindiscr.valid)
lindiscr.AUCsc <- c(lindiscr.AUCsc, auc(groups[[1]]$DEATH_EVENT, lindiscr.valid))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[1]], type = 'response')
quaddiscr.valid <- as.numeric(quaddiscr.valid$class)-1
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[1]]))
quaddiscr.ConfMat[[1]] <- confusionMatrix(factor(quaddiscr.valid), factor(groups[[1]]$DEATH_EVENT))
quaddiscr.accur <- c(quaddiscr.accur, quaddiscr.ConfMat[[1]]$overall[1])
quaddiscr.precis <- c(quaddiscr.precis, quaddiscr.ConfMat[[1]]$byClass[5])
quaddiscr.recall <- c(quaddiscr.recall, quaddiscr.ConfMat[[1]]$byClass[6])
quaddiscr.F1score <- c(quaddiscr.F1score, quaddiscr.ConfMat[[1]]$byClass[7])
quaddiscr.ConfMat[[1]] <- quaddiscr.ConfMat[[1]]$table
quaddiscr.ROCcurv[[1]] <- roc(groups[[1]]$DEATH_EVENT, quaddiscr.valid)
quaddiscr.AUCsc <- c(quaddiscr.AUCsc, auc(groups[[1]]$DEATH_EVENT, quaddiscr.valid))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[1]], type = 'class')
mixeddiscr.valid <- as.numeric(mixeddiscr.valid)-1
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[1]]))
mixeddiscr.ConfMat[[1]] <- confusionMatrix(factor(mixeddiscr.valid), factor(groups[[1]]$DEATH_EVENT))
mixeddiscr.accur <- c(mixeddiscr.accur, mixeddiscr.ConfMat[[1]]$overall[1])
mixeddiscr.precis <- c(mixeddiscr.precis, mixeddiscr.ConfMat[[1]]$byClass[5])
mixeddiscr.recall <- c(mixeddiscr.recall, mixeddiscr.ConfMat[[1]]$byClass[6])
mixeddiscr.F1score <- c(mixeddiscr.F1score, mixeddiscr.ConfMat[[1]]$byClass[7])
mixeddiscr.ConfMat[[1]] <- mixeddiscr.ConfMat[[1]]$table
mixeddiscr.ROCcurv[[1]] <- roc(groups[[1]]$DEATH_EVENT, mixeddiscr.valid)
mixeddiscr.AUCsc <- c(mixeddiscr.AUCsc, auc(groups[[1]]$DEATH_EVENT, mixeddiscr.valid))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[1]], type = 'class')
flexdiscr.valid <- as.numeric(flexdiscr.valid)-1
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[1]]))
flexdiscr.ConfMat[[1]] <- confusionMatrix(factor(flexdiscr.valid), factor(groups[[1]]$DEATH_EVENT))
flexdiscr.accur <- c(flexdiscr.accur, flexdiscr.ConfMat[[1]]$overall[1])
flexdiscr.precis <- c(flexdiscr.precis, flexdiscr.ConfMat[[1]]$byClass[5])
flexdiscr.recall <- c(flexdiscr.recall, flexdiscr.ConfMat[[1]]$byClass[6])
flexdiscr.F1score <- c(flexdiscr.F1score, flexdiscr.ConfMat[[1]]$byClass[7])
flexdiscr.ConfMat[[1]] <- flexdiscr.ConfMat[[1]]$table
flexdiscr.ROCcurv[[1]] <- roc(groups[[1]]$DEATH_EVENT, flexdiscr.valid)
flexdiscr.AUCsc <- c(flexdiscr.AUCsc, auc(groups[[1]]$DEATH_EVENT, flexdiscr.valid))

decisiontree.model <- tree(as.factor(DEATH_EVENT) ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[1]], type = 'class')
decisiontree.valid <- as.numeric(decisiontree.valid)-1
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[1]]))
decisiontree.ConfMat[[1]] <- confusionMatrix(factor(decisiontree.valid), factor(groups[[1]]$DEATH_EVENT))
decisiontree.accur <- c(decisiontree.accur, decisiontree.ConfMat[[1]]$overall[1])
decisiontree.precis <- c(decisiontree.precis, decisiontree.ConfMat[[1]]$byClass[5])
decisiontree.recall <- c(decisiontree.recall, decisiontree.ConfMat[[1]]$byClass[6])
decisiontree.F1score <- c(decisiontree.F1score, decisiontree.ConfMat[[1]]$byClass[7])
decisiontree.ConfMat[[1]] <- decisiontree.ConfMat[[1]]$table
decisiontree.ROCcurv[[1]] <- roc(groups[[1]]$DEATH_EVENT, decisiontree.valid)
decisiontree.AUCsc <- c(decisiontree.AUCsc, auc(groups[[1]]$DEATH_EVENT, decisiontree.valid))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[1]], type = 'class')
prunedectree.valid <- as.numeric(prunedectree.valid)-1
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[1]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[1]]))
prunedectree.ConfMat[[1]] <- confusionMatrix(factor(prunedectree.valid), factor(groups[[1]]$DEATH_EVENT))
prunedectree.accur <- c(prunedectree.accur, prunedectree.ConfMat[[1]]$overall[1])
prunedectree.precis <- c(prunedectree.precis, prunedectree.ConfMat[[1]]$byClass[5])
prunedectree.recall <- c(prunedectree.recall, prunedectree.ConfMat[[1]]$byClass[6])
prunedectree.F1score <- c(prunedectree.F1score, prunedectree.ConfMat[[1]]$byClass[7])
prunedectree.ConfMat[[1]] <- prunedectree.ConfMat[[1]]$table
prunedectree.ROCcurv[[1]] <- roc(groups[[1]]$DEATH_EVENT, prunedectree.valid)
prunedectree.AUCsc <- c(prunedectree.AUCsc, auc(groups[[1]]$DEATH_EVENT, prunedectree.valid))


##2nd fold-out training
trainset <- data.frame()
for (i in c(1,3:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[2]])
for (j in 1:length(simplelinear.valid)) {
  if(simplelinear.valid[j] < 0.5){
    simplelinear.valid[j] <- 0
  }
  if(simplelinear.valid[j] >= 0.5){
    simplelinear.valid[j] <- 1
  }
}
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[2]]))
simplelinear.ConfMat[[2]] <- confusionMatrix(factor(simplelinear.valid), factor(groups[[2]]$DEATH_EVENT))
simplelinear.accur <- c(simplelinear.accur, simplelinear.ConfMat[[2]]$overall[1])
simplelinear.precis <- c(simplelinear.precis, simplelinear.ConfMat[[2]]$byClass[5])
simplelinear.recall <- c(simplelinear.recall, simplelinear.ConfMat[[2]]$byClass[6])
simplelinear.F1score <- c(simplelinear.F1score, simplelinear.ConfMat[[2]]$byClass[7])
simplelinear.ConfMat[[2]] <- simplelinear.ConfMat[[2]]$table
simplelinear.ROCcurv[[2]] <- roc(groups[[2]]$DEATH_EVENT, simplelinear.valid)
simplelinear.AUCsc <- c(simplelinear.AUCsc, auc(groups[[2]]$DEATH_EVENT, simplelinear.valid))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[2]])
for (j in 1:length(revisedlinear.valid)) {
  if(revisedlinear.valid[j] < 0.5){
    revisedlinear.valid[j] <- 0
  }
  if(revisedlinear.valid[j] >= 0.5){
    revisedlinear.valid[j] <- 1
  }
}
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[2]]))
revisedlinear.ConfMat[[2]] <- confusionMatrix(factor(revisedlinear.valid), factor(groups[[2]]$DEATH_EVENT))
revisedlinear.accur <- c(revisedlinear.accur, revisedlinear.ConfMat[[2]]$overall[1])
revisedlinear.precis <- c(revisedlinear.precis, revisedlinear.ConfMat[[2]]$byClass[5])
revisedlinear.recall <- c(revisedlinear.recall, revisedlinear.ConfMat[[2]]$byClass[6])
revisedlinear.F1score <- c(revisedlinear.F1score, revisedlinear.ConfMat[[2]]$byClass[7])
revisedlinear.ConfMat[[2]] <- revisedlinear.ConfMat[[2]]$table
revisedlinear.ROCcurv[[2]] <- roc(groups[[2]]$DEATH_EVENT, revisedlinear.valid)
revisedlinear.AUCsc <- c(revisedlinear.AUCsc, auc(groups[[2]]$DEATH_EVENT, revisedlinear.valid))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[2]], type = 'response')
for (j in 1:length(generlinear.valid)) {
  if(generlinear.valid[j] < 0.5){
    generlinear.valid[j] <- 0
  }
  if(generlinear.valid[j] >= 0.5){
    generlinear.valid[j] <- 1
  }
}
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[2]]))
generlinear.ConfMat[[2]] <- confusionMatrix(factor(generlinear.valid), factor(groups[[2]]$DEATH_EVENT))
generlinear.accur <- c(generlinear.accur, generlinear.ConfMat[[2]]$overall[1])
generlinear.precis <- c(generlinear.precis, generlinear.ConfMat[[2]]$byClass[5])
generlinear.recall <- c(generlinear.recall, generlinear.ConfMat[[2]]$byClass[6])
generlinear.F1score <- c(generlinear.F1score, generlinear.ConfMat[[2]]$byClass[7])
generlinear.ConfMat[[2]] <- generlinear.ConfMat[[2]]$table
generlinear.ROCcurv[[2]] <- roc(groups[[2]]$DEATH_EVENT, generlinear.valid)
generlinear.AUCsc <- c(generlinear.AUCsc, auc(groups[[2]]$DEATH_EVENT, generlinear.valid))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[2]], type = 'response')
for (j in 1:length(generaddit.valid)) {
  if(generaddit.valid[j] < 0.5){
    generaddit.valid[j] <- 0
  }
  if(generaddit.valid[j] >= 0.5){
    generaddit.valid[j] <- 1
  }
}
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[2]]))
generaddit.ConfMat[[2]] <- confusionMatrix(factor(generaddit.valid), factor(groups[[2]]$DEATH_EVENT))
generaddit.accur <- c(generaddit.accur, generaddit.ConfMat[[2]]$overall[1])
generaddit.precis <- c(generaddit.precis, generaddit.ConfMat[[2]]$byClass[5])
generaddit.recall <- c(generaddit.recall, generaddit.ConfMat[[2]]$byClass[6])
generaddit.F1score <- c(generaddit.F1score, generaddit.ConfMat[[2]]$byClass[7])
generaddit.ConfMat[[2]] <- generaddit.ConfMat[[2]]$table
generaddit.ROCcurv[[2]] <- roc(groups[[2]]$DEATH_EVENT, generaddit.valid)
generaddit.AUCsc <- c(generaddit.AUCsc, auc(groups[[2]]$DEATH_EVENT, generaddit.valid))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[2]], type = 'response')
lindiscr.valid <- as.numeric(lindiscr.valid$class)-1
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[2]]))
lindiscr.ConfMat[[2]] <- confusionMatrix(factor(lindiscr.valid), factor(groups[[2]]$DEATH_EVENT))
lindiscr.accur <- c(lindiscr.accur, lindiscr.ConfMat[[2]]$overall[1])
lindiscr.precis <- c(lindiscr.precis, lindiscr.ConfMat[[2]]$byClass[5])
lindiscr.recall <- c(lindiscr.recall, lindiscr.ConfMat[[2]]$byClass[6])
lindiscr.F1score <- c(lindiscr.F1score, lindiscr.ConfMat[[2]]$byClass[7])
lindiscr.ConfMat[[2]] <- lindiscr.ConfMat[[2]]$table
lindiscr.ROCcurv[[2]] <- roc(groups[[2]]$DEATH_EVENT, lindiscr.valid)
lindiscr.AUCsc <- c(lindiscr.AUCsc, auc(groups[[2]]$DEATH_EVENT, lindiscr.valid))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[2]], type = 'response')
quaddiscr.valid <- as.numeric(quaddiscr.valid$class)-1
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[2]]))
quaddiscr.ConfMat[[2]] <- confusionMatrix(factor(quaddiscr.valid), factor(groups[[2]]$DEATH_EVENT))
quaddiscr.accur <- c(quaddiscr.accur, quaddiscr.ConfMat[[2]]$overall[1])
quaddiscr.precis <- c(quaddiscr.precis, quaddiscr.ConfMat[[2]]$byClass[5])
quaddiscr.recall <- c(quaddiscr.recall, quaddiscr.ConfMat[[2]]$byClass[6])
quaddiscr.F1score <- c(quaddiscr.F1score, quaddiscr.ConfMat[[2]]$byClass[7])
quaddiscr.ConfMat[[2]] <- quaddiscr.ConfMat[[2]]$table
quaddiscr.ROCcurv[[2]] <- roc(groups[[2]]$DEATH_EVENT, quaddiscr.valid)
quaddiscr.AUCsc <- c(quaddiscr.AUCsc, auc(groups[[2]]$DEATH_EVENT, quaddiscr.valid))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[2]], type = 'class')
mixeddiscr.valid <- as.numeric(mixeddiscr.valid)-1
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[2]]))
mixeddiscr.ConfMat[[2]] <- confusionMatrix(factor(mixeddiscr.valid), factor(groups[[2]]$DEATH_EVENT))
mixeddiscr.accur <- c(mixeddiscr.accur, mixeddiscr.ConfMat[[2]]$overall[1])
mixeddiscr.precis <- c(mixeddiscr.precis, mixeddiscr.ConfMat[[2]]$byClass[5])
mixeddiscr.recall <- c(mixeddiscr.recall, mixeddiscr.ConfMat[[2]]$byClass[6])
mixeddiscr.F1score <- c(mixeddiscr.F1score, mixeddiscr.ConfMat[[2]]$byClass[7])
mixeddiscr.ConfMat[[2]] <- mixeddiscr.ConfMat[[2]]$table
mixeddiscr.ROCcurv[[2]] <- roc(groups[[2]]$DEATH_EVENT, mixeddiscr.valid)
mixeddiscr.AUCsc <- c(mixeddiscr.AUCsc, auc(groups[[2]]$DEATH_EVENT, mixeddiscr.valid))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[2]], type = 'class')
flexdiscr.valid <- as.numeric(flexdiscr.valid)-1
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[2]]))
flexdiscr.ConfMat[[2]] <- confusionMatrix(factor(flexdiscr.valid), factor(groups[[2]]$DEATH_EVENT))
flexdiscr.accur <- c(flexdiscr.accur, flexdiscr.ConfMat[[2]]$overall[1])
flexdiscr.precis <- c(flexdiscr.precis, flexdiscr.ConfMat[[2]]$byClass[5])
flexdiscr.recall <- c(flexdiscr.recall, flexdiscr.ConfMat[[2]]$byClass[6])
flexdiscr.F1score <- c(flexdiscr.F1score, flexdiscr.ConfMat[[2]]$byClass[7])
flexdiscr.ConfMat[[2]] <- flexdiscr.ConfMat[[2]]$table
flexdiscr.ROCcurv[[2]] <- roc(groups[[2]]$DEATH_EVENT, flexdiscr.valid)
flexdiscr.AUCsc <- c(flexdiscr.AUCsc, auc(groups[[2]]$DEATH_EVENT, flexdiscr.valid))

decisiontree.model <- tree(as.factor(DEATH_EVENT) ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[2]], type = 'class')
decisiontree.valid <- as.numeric(decisiontree.valid)-1
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[2]]))
decisiontree.ConfMat[[2]] <- confusionMatrix(factor(decisiontree.valid), factor(groups[[2]]$DEATH_EVENT))
decisiontree.accur <- c(decisiontree.accur, decisiontree.ConfMat[[2]]$overall[1])
decisiontree.precis <- c(decisiontree.precis, decisiontree.ConfMat[[2]]$byClass[5])
decisiontree.recall <- c(decisiontree.recall, decisiontree.ConfMat[[2]]$byClass[6])
decisiontree.F1score <- c(decisiontree.F1score, decisiontree.ConfMat[[2]]$byClass[7])
decisiontree.ConfMat[[2]] <- decisiontree.ConfMat[[2]]$table
decisiontree.ROCcurv[[2]] <- roc(groups[[2]]$DEATH_EVENT, decisiontree.valid)
decisiontree.AUCsc <- c(decisiontree.AUCsc, auc(groups[[2]]$DEATH_EVENT, decisiontree.valid))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[2]], type = 'class')
prunedectree.valid <- as.numeric(prunedectree.valid)-1
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[2]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[2]]))
prunedectree.ConfMat[[2]] <- confusionMatrix(factor(prunedectree.valid), factor(groups[[2]]$DEATH_EVENT))
prunedectree.accur <- c(prunedectree.accur, prunedectree.ConfMat[[2]]$overall[1])
prunedectree.precis <- c(prunedectree.precis, prunedectree.ConfMat[[2]]$byClass[5])
prunedectree.recall <- c(prunedectree.recall, prunedectree.ConfMat[[2]]$byClass[6])
prunedectree.F1score <- c(prunedectree.F1score, prunedectree.ConfMat[[2]]$byClass[7])
prunedectree.ConfMat[[2]] <- prunedectree.ConfMat[[2]]$table
prunedectree.ROCcurv[[2]] <- roc(groups[[2]]$DEATH_EVENT, prunedectree.valid)
prunedectree.AUCsc <- c(prunedectree.AUCsc, auc(groups[[2]]$DEATH_EVENT, prunedectree.valid))


##3rd fold-out training
trainset <- data.frame()
for (i in c(1:2,4:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[3]])
for (j in 1:length(simplelinear.valid)) {
  if(simplelinear.valid[j] < 0.5){
    simplelinear.valid[j] <- 0
  }
  if(simplelinear.valid[j] >= 0.5){
    simplelinear.valid[j] <- 1
  }
}
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[3]]))
simplelinear.ConfMat[[3]] <- confusionMatrix(factor(simplelinear.valid), factor(groups[[3]]$DEATH_EVENT))
simplelinear.accur <- c(simplelinear.accur, simplelinear.ConfMat[[3]]$overall[1])
simplelinear.precis <- c(simplelinear.precis, simplelinear.ConfMat[[3]]$byClass[5])
simplelinear.recall <- c(simplelinear.recall, simplelinear.ConfMat[[3]]$byClass[6])
simplelinear.F1score <- c(simplelinear.F1score, simplelinear.ConfMat[[3]]$byClass[7])
simplelinear.ConfMat[[3]] <- simplelinear.ConfMat[[3]]$table
simplelinear.ROCcurv[[3]] <- roc(groups[[3]]$DEATH_EVENT, simplelinear.valid)
simplelinear.AUCsc <- c(simplelinear.AUCsc, auc(groups[[3]]$DEATH_EVENT, simplelinear.valid))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[3]])
for (j in 1:length(revisedlinear.valid)) {
  if(revisedlinear.valid[j] < 0.5){
    revisedlinear.valid[j] <- 0
  }
  if(revisedlinear.valid[j] >= 0.5){
    revisedlinear.valid[j] <- 1
  }
}
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[3]]))
revisedlinear.ConfMat[[3]] <- confusionMatrix(factor(revisedlinear.valid), factor(groups[[3]]$DEATH_EVENT))
revisedlinear.accur <- c(revisedlinear.accur, revisedlinear.ConfMat[[3]]$overall[1])
revisedlinear.precis <- c(revisedlinear.precis, revisedlinear.ConfMat[[3]]$byClass[5])
revisedlinear.recall <- c(revisedlinear.recall, revisedlinear.ConfMat[[3]]$byClass[6])
revisedlinear.F1score <- c(revisedlinear.F1score, revisedlinear.ConfMat[[3]]$byClass[7])
revisedlinear.ConfMat[[3]] <- revisedlinear.ConfMat[[3]]$table
revisedlinear.ROCcurv[[3]] <- roc(groups[[3]]$DEATH_EVENT, revisedlinear.valid)
revisedlinear.AUCsc <- c(revisedlinear.AUCsc, auc(groups[[3]]$DEATH_EVENT, revisedlinear.valid))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[3]], type = 'response')
for (j in 1:length(generlinear.valid)) {
  if(generlinear.valid[j] < 0.5){
    generlinear.valid[j] <- 0
  }
  if(generlinear.valid[j] >= 0.5){
    generlinear.valid[j] <- 1
  }
}
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[3]]))
generlinear.ConfMat[[3]] <- confusionMatrix(factor(generlinear.valid), factor(groups[[3]]$DEATH_EVENT))
generlinear.accur <- c(generlinear.accur, generlinear.ConfMat[[3]]$overall[1])
generlinear.precis <- c(generlinear.precis, generlinear.ConfMat[[3]]$byClass[5])
generlinear.recall <- c(generlinear.recall, generlinear.ConfMat[[3]]$byClass[6])
generlinear.F1score <- c(generlinear.F1score, generlinear.ConfMat[[3]]$byClass[7])
generlinear.ConfMat[[3]] <- generlinear.ConfMat[[3]]$table
generlinear.ROCcurv[[3]] <- roc(groups[[3]]$DEATH_EVENT, generlinear.valid)
generlinear.AUCsc <- c(generlinear.AUCsc, auc(groups[[3]]$DEATH_EVENT, generlinear.valid))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[3]], type = 'response')
for (j in 1:length(generaddit.valid)) {
  if(generaddit.valid[j] < 0.5){
    generaddit.valid[j] <- 0
  }
  if(generaddit.valid[j] >= 0.5){
    generaddit.valid[j] <- 1
  }
}
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[3]]))
generaddit.ConfMat[[3]] <- confusionMatrix(factor(generaddit.valid), factor(groups[[3]]$DEATH_EVENT))
generaddit.accur <- c(generaddit.accur, generaddit.ConfMat[[3]]$overall[1])
generaddit.precis <- c(generaddit.precis, generaddit.ConfMat[[3]]$byClass[5])
generaddit.recall <- c(generaddit.recall, generaddit.ConfMat[[3]]$byClass[6])
generaddit.F1score <- c(generaddit.F1score, generaddit.ConfMat[[3]]$byClass[7])
generaddit.ConfMat[[3]] <- generaddit.ConfMat[[3]]$table
generaddit.ROCcurv[[3]] <- roc(groups[[3]]$DEATH_EVENT, generaddit.valid)
generaddit.AUCsc <- c(generaddit.AUCsc, auc(groups[[3]]$DEATH_EVENT, generaddit.valid))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[3]], type = 'response')
lindiscr.valid <- as.numeric(lindiscr.valid$class)-1
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[3]]))
lindiscr.ConfMat[[3]] <- confusionMatrix(factor(lindiscr.valid), factor(groups[[3]]$DEATH_EVENT))
lindiscr.accur <- c(lindiscr.accur, lindiscr.ConfMat[[3]]$overall[1])
lindiscr.precis <- c(lindiscr.precis, lindiscr.ConfMat[[3]]$byClass[5])
lindiscr.recall <- c(lindiscr.recall, lindiscr.ConfMat[[3]]$byClass[6])
lindiscr.F1score <- c(lindiscr.F1score, lindiscr.ConfMat[[3]]$byClass[7])
lindiscr.ConfMat[[3]] <- lindiscr.ConfMat[[3]]$table
lindiscr.ROCcurv[[3]] <- roc(groups[[3]]$DEATH_EVENT, lindiscr.valid)
lindiscr.AUCsc <- c(lindiscr.AUCsc, auc(groups[[3]]$DEATH_EVENT, lindiscr.valid))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[3]], type = 'response')
quaddiscr.valid <- as.numeric(quaddiscr.valid$class)-1
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[3]]))
quaddiscr.ConfMat[[3]] <- confusionMatrix(factor(quaddiscr.valid), factor(groups[[3]]$DEATH_EVENT))
quaddiscr.accur <- c(quaddiscr.accur, quaddiscr.ConfMat[[3]]$overall[1])
quaddiscr.precis <- c(quaddiscr.precis, quaddiscr.ConfMat[[3]]$byClass[5])
quaddiscr.recall <- c(quaddiscr.recall, quaddiscr.ConfMat[[3]]$byClass[6])
quaddiscr.F1score <- c(quaddiscr.F1score, quaddiscr.ConfMat[[3]]$byClass[7])
quaddiscr.ConfMat[[3]] <- quaddiscr.ConfMat[[3]]$table
quaddiscr.ROCcurv[[3]] <- roc(groups[[3]]$DEATH_EVENT, quaddiscr.valid)
quaddiscr.AUCsc <- c(quaddiscr.AUCsc, auc(groups[[3]]$DEATH_EVENT, quaddiscr.valid))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[3]], type = 'class')
mixeddiscr.valid <- as.numeric(mixeddiscr.valid)-1
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[3]]))
mixeddiscr.ConfMat[[3]] <- confusionMatrix(factor(mixeddiscr.valid), factor(groups[[3]]$DEATH_EVENT))
mixeddiscr.accur <- c(mixeddiscr.accur, mixeddiscr.ConfMat[[3]]$overall[1])
mixeddiscr.precis <- c(mixeddiscr.precis, mixeddiscr.ConfMat[[3]]$byClass[5])
mixeddiscr.recall <- c(mixeddiscr.recall, mixeddiscr.ConfMat[[3]]$byClass[6])
mixeddiscr.F1score <- c(mixeddiscr.F1score, mixeddiscr.ConfMat[[3]]$byClass[7])
mixeddiscr.ConfMat[[3]] <- mixeddiscr.ConfMat[[3]]$table
mixeddiscr.ROCcurv[[3]] <- roc(groups[[3]]$DEATH_EVENT, mixeddiscr.valid)
mixeddiscr.AUCsc <- c(mixeddiscr.AUCsc, auc(groups[[3]]$DEATH_EVENT, mixeddiscr.valid))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[3]], type = 'class')
flexdiscr.valid <- as.numeric(flexdiscr.valid)-1
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[3]]))
flexdiscr.ConfMat[[3]] <- confusionMatrix(factor(flexdiscr.valid), factor(groups[[3]]$DEATH_EVENT))
flexdiscr.accur <- c(flexdiscr.accur, flexdiscr.ConfMat[[3]]$overall[1])
flexdiscr.precis <- c(flexdiscr.precis, flexdiscr.ConfMat[[3]]$byClass[5])
flexdiscr.recall <- c(flexdiscr.recall, flexdiscr.ConfMat[[3]]$byClass[6])
flexdiscr.F1score <- c(flexdiscr.F1score, flexdiscr.ConfMat[[3]]$byClass[7])
flexdiscr.ConfMat[[3]] <- flexdiscr.ConfMat[[3]]$table
flexdiscr.ROCcurv[[3]] <- roc(groups[[3]]$DEATH_EVENT, flexdiscr.valid)
flexdiscr.AUCsc <- c(flexdiscr.AUCsc, auc(groups[[3]]$DEATH_EVENT, flexdiscr.valid))

decisiontree.model <- tree(as.factor(DEATH_EVENT) ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[3]], type = 'class')
decisiontree.valid <- as.numeric(decisiontree.valid)-1
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[3]]))
decisiontree.ConfMat[[3]] <- confusionMatrix(factor(decisiontree.valid), factor(groups[[3]]$DEATH_EVENT))
decisiontree.accur <- c(decisiontree.accur, decisiontree.ConfMat[[3]]$overall[1])
decisiontree.precis <- c(decisiontree.precis, decisiontree.ConfMat[[3]]$byClass[5])
decisiontree.recall <- c(decisiontree.recall, decisiontree.ConfMat[[3]]$byClass[6])
decisiontree.F1score <- c(decisiontree.F1score, decisiontree.ConfMat[[3]]$byClass[7])
decisiontree.ConfMat[[3]] <- decisiontree.ConfMat[[3]]$table
decisiontree.ROCcurv[[3]] <- roc(groups[[3]]$DEATH_EVENT, decisiontree.valid)
decisiontree.AUCsc <- c(decisiontree.AUCsc, auc(groups[[3]]$DEATH_EVENT, decisiontree.valid))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[3]], type = 'class')
prunedectree.valid <- as.numeric(prunedectree.valid)-1
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[3]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[3]]))
prunedectree.ConfMat[[3]] <- confusionMatrix(factor(prunedectree.valid), factor(groups[[3]]$DEATH_EVENT))
prunedectree.accur <- c(prunedectree.accur, prunedectree.ConfMat[[3]]$overall[1])
prunedectree.precis <- c(prunedectree.precis, prunedectree.ConfMat[[3]]$byClass[5])
prunedectree.recall <- c(prunedectree.recall, prunedectree.ConfMat[[3]]$byClass[6])
prunedectree.F1score <- c(prunedectree.F1score, prunedectree.ConfMat[[3]]$byClass[7])
prunedectree.ConfMat[[3]] <- prunedectree.ConfMat[[3]]$table
prunedectree.ROCcurv[[3]] <- roc(groups[[3]]$DEATH_EVENT, prunedectree.valid)
prunedectree.AUCsc <- c(prunedectree.AUCsc, auc(groups[[3]]$DEATH_EVENT, prunedectree.valid))


##4th fold-out training
trainset <- data.frame()
for (i in c(1:3,5:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[4]])
for (j in 1:length(simplelinear.valid)) {
  if(simplelinear.valid[j] < 0.5){
    simplelinear.valid[j] <- 0
  }
  if(simplelinear.valid[j] >= 0.5){
    simplelinear.valid[j] <- 1
  }
}
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[4]]))
simplelinear.ConfMat[[4]] <- confusionMatrix(factor(simplelinear.valid), factor(groups[[4]]$DEATH_EVENT))
simplelinear.accur <- c(simplelinear.accur, simplelinear.ConfMat[[4]]$overall[1])
simplelinear.precis <- c(simplelinear.precis, simplelinear.ConfMat[[4]]$byClass[5])
simplelinear.recall <- c(simplelinear.recall, simplelinear.ConfMat[[4]]$byClass[6])
simplelinear.F1score <- c(simplelinear.F1score, simplelinear.ConfMat[[4]]$byClass[7])
simplelinear.ConfMat[[4]] <- simplelinear.ConfMat[[4]]$table
simplelinear.ROCcurv[[4]] <- roc(groups[[4]]$DEATH_EVENT, simplelinear.valid)
simplelinear.AUCsc <- c(simplelinear.AUCsc, auc(groups[[4]]$DEATH_EVENT, simplelinear.valid))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[4]])
for (j in 1:length(revisedlinear.valid)) {
  if(revisedlinear.valid[j] < 0.5){
    revisedlinear.valid[j] <- 0
  }
  if(revisedlinear.valid[j] >= 0.5){
    revisedlinear.valid[j] <- 1
  }
}
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[4]]))
revisedlinear.ConfMat[[4]] <- confusionMatrix(factor(revisedlinear.valid), factor(groups[[4]]$DEATH_EVENT))
revisedlinear.accur <- c(revisedlinear.accur, revisedlinear.ConfMat[[4]]$overall[1])
revisedlinear.precis <- c(revisedlinear.precis, revisedlinear.ConfMat[[4]]$byClass[5])
revisedlinear.recall <- c(revisedlinear.recall, revisedlinear.ConfMat[[4]]$byClass[6])
revisedlinear.F1score <- c(revisedlinear.F1score, revisedlinear.ConfMat[[4]]$byClass[7])
revisedlinear.ConfMat[[4]] <- revisedlinear.ConfMat[[4]]$table
revisedlinear.ROCcurv[[4]] <- roc(groups[[4]]$DEATH_EVENT, revisedlinear.valid)
revisedlinear.AUCsc <- c(revisedlinear.AUCsc, auc(groups[[4]]$DEATH_EVENT, revisedlinear.valid))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[4]], type = 'response')
for (j in 1:length(generlinear.valid)) {
  if(generlinear.valid[j] < 0.5){
    generlinear.valid[j] <- 0
  }
  if(generlinear.valid[j] >= 0.5){
    generlinear.valid[j] <- 1
  }
}
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[4]]))
generlinear.ConfMat[[4]] <- confusionMatrix(factor(generlinear.valid), factor(groups[[4]]$DEATH_EVENT))
generlinear.accur <- c(generlinear.accur, generlinear.ConfMat[[4]]$overall[1])
generlinear.precis <- c(generlinear.precis, generlinear.ConfMat[[4]]$byClass[5])
generlinear.recall <- c(generlinear.recall, generlinear.ConfMat[[4]]$byClass[6])
generlinear.F1score <- c(generlinear.F1score, generlinear.ConfMat[[4]]$byClass[7])
generlinear.ConfMat[[4]] <- generlinear.ConfMat[[4]]$table
generlinear.ROCcurv[[4]] <- roc(groups[[4]]$DEATH_EVENT, generlinear.valid)
generlinear.AUCsc <- c(generlinear.AUCsc, auc(groups[[4]]$DEATH_EVENT, generlinear.valid))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[4]], type = 'response')
for (j in 1:length(generaddit.valid)) {
  if(generaddit.valid[j] < 0.5){
    generaddit.valid[j] <- 0
  }
  if(generaddit.valid[j] >= 0.5){
    generaddit.valid[j] <- 1
  }
}
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[4]]))
generaddit.ConfMat[[4]] <- confusionMatrix(factor(generaddit.valid), factor(groups[[4]]$DEATH_EVENT))
generaddit.accur <- c(generaddit.accur, generaddit.ConfMat[[4]]$overall[1])
generaddit.precis <- c(generaddit.precis, generaddit.ConfMat[[4]]$byClass[5])
generaddit.recall <- c(generaddit.recall, generaddit.ConfMat[[4]]$byClass[6])
generaddit.F1score <- c(generaddit.F1score, generaddit.ConfMat[[4]]$byClass[7])
generaddit.ConfMat[[4]] <- generaddit.ConfMat[[4]]$table
generaddit.ROCcurv[[4]] <- roc(groups[[4]]$DEATH_EVENT, generaddit.valid)
generaddit.AUCsc <- c(generaddit.AUCsc, auc(groups[[4]]$DEATH_EVENT, generaddit.valid))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[4]], type = 'response')
lindiscr.valid <- as.numeric(lindiscr.valid$class)-1
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[4]]))
lindiscr.ConfMat[[4]] <- confusionMatrix(factor(lindiscr.valid), factor(groups[[4]]$DEATH_EVENT))
lindiscr.accur <- c(lindiscr.accur, lindiscr.ConfMat[[4]]$overall[1])
lindiscr.precis <- c(lindiscr.precis, lindiscr.ConfMat[[4]]$byClass[5])
lindiscr.recall <- c(lindiscr.recall, lindiscr.ConfMat[[4]]$byClass[6])
lindiscr.F1score <- c(lindiscr.F1score, lindiscr.ConfMat[[4]]$byClass[7])
lindiscr.ConfMat[[4]] <- lindiscr.ConfMat[[4]]$table
lindiscr.ROCcurv[[4]] <- roc(groups[[4]]$DEATH_EVENT, lindiscr.valid)
lindiscr.AUCsc <- c(lindiscr.AUCsc, auc(groups[[4]]$DEATH_EVENT, lindiscr.valid))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[4]], type = 'response')
quaddiscr.valid <- as.numeric(quaddiscr.valid$class)-1
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[4]]))
quaddiscr.ConfMat[[4]] <- confusionMatrix(factor(quaddiscr.valid), factor(groups[[4]]$DEATH_EVENT))
quaddiscr.accur <- c(quaddiscr.accur, quaddiscr.ConfMat[[4]]$overall[1])
quaddiscr.precis <- c(quaddiscr.precis, quaddiscr.ConfMat[[4]]$byClass[5])
quaddiscr.recall <- c(quaddiscr.recall, quaddiscr.ConfMat[[4]]$byClass[6])
quaddiscr.F1score <- c(quaddiscr.F1score, quaddiscr.ConfMat[[4]]$byClass[7])
quaddiscr.ConfMat[[4]] <- quaddiscr.ConfMat[[4]]$table
quaddiscr.ROCcurv[[4]] <- roc(groups[[4]]$DEATH_EVENT, quaddiscr.valid)
quaddiscr.AUCsc <- c(quaddiscr.AUCsc, auc(groups[[4]]$DEATH_EVENT, quaddiscr.valid))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[4]], type = 'class')
mixeddiscr.valid <- as.numeric(mixeddiscr.valid)-1
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[4]]))
mixeddiscr.ConfMat[[4]] <- confusionMatrix(factor(mixeddiscr.valid), factor(groups[[4]]$DEATH_EVENT))
mixeddiscr.accur <- c(mixeddiscr.accur, mixeddiscr.ConfMat[[4]]$overall[1])
mixeddiscr.precis <- c(mixeddiscr.precis, mixeddiscr.ConfMat[[4]]$byClass[5])
mixeddiscr.recall <- c(mixeddiscr.recall, mixeddiscr.ConfMat[[4]]$byClass[6])
mixeddiscr.F1score <- c(mixeddiscr.F1score, mixeddiscr.ConfMat[[4]]$byClass[7])
mixeddiscr.ConfMat[[4]] <- mixeddiscr.ConfMat[[4]]$table
mixeddiscr.ROCcurv[[4]] <- roc(groups[[4]]$DEATH_EVENT, mixeddiscr.valid)
mixeddiscr.AUCsc <- c(mixeddiscr.AUCsc, auc(groups[[4]]$DEATH_EVENT, mixeddiscr.valid))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[4]], type = 'class')
flexdiscr.valid <- as.numeric(flexdiscr.valid)-1
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[4]]))
flexdiscr.ConfMat[[4]] <- confusionMatrix(factor(flexdiscr.valid), factor(groups[[4]]$DEATH_EVENT))
flexdiscr.accur <- c(flexdiscr.accur, flexdiscr.ConfMat[[4]]$overall[1])
flexdiscr.precis <- c(flexdiscr.precis, flexdiscr.ConfMat[[4]]$byClass[5])
flexdiscr.recall <- c(flexdiscr.recall, flexdiscr.ConfMat[[4]]$byClass[6])
flexdiscr.F1score <- c(flexdiscr.F1score, flexdiscr.ConfMat[[4]]$byClass[7])
flexdiscr.ConfMat[[4]] <- flexdiscr.ConfMat[[4]]$table
flexdiscr.ROCcurv[[4]] <- roc(groups[[4]]$DEATH_EVENT, flexdiscr.valid)
flexdiscr.AUCsc <- c(flexdiscr.AUCsc, auc(groups[[4]]$DEATH_EVENT, flexdiscr.valid))

decisiontree.model <- tree(as.factor(DEATH_EVENT) ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[4]], type = 'class')
decisiontree.valid <- as.numeric(decisiontree.valid)-1
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[4]]))
decisiontree.ConfMat[[4]] <- confusionMatrix(factor(decisiontree.valid), factor(groups[[4]]$DEATH_EVENT))
decisiontree.accur <- c(decisiontree.accur, decisiontree.ConfMat[[4]]$overall[1])
decisiontree.precis <- c(decisiontree.precis, decisiontree.ConfMat[[4]]$byClass[5])
decisiontree.recall <- c(decisiontree.recall, decisiontree.ConfMat[[4]]$byClass[6])
decisiontree.F1score <- c(decisiontree.F1score, decisiontree.ConfMat[[4]]$byClass[7])
decisiontree.ConfMat[[4]] <- decisiontree.ConfMat[[4]]$table
decisiontree.ROCcurv[[4]] <- roc(groups[[4]]$DEATH_EVENT, decisiontree.valid)
decisiontree.AUCsc <- c(decisiontree.AUCsc, auc(groups[[4]]$DEATH_EVENT, decisiontree.valid))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[4]], type = 'class')
prunedectree.valid <- as.numeric(prunedectree.valid)-1
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[4]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[4]]))
prunedectree.ConfMat[[4]] <- confusionMatrix(factor(prunedectree.valid), factor(groups[[4]]$DEATH_EVENT))
prunedectree.accur <- c(prunedectree.accur, prunedectree.ConfMat[[4]]$overall[1])
prunedectree.precis <- c(prunedectree.precis, prunedectree.ConfMat[[4]]$byClass[5])
prunedectree.recall <- c(prunedectree.recall, prunedectree.ConfMat[[4]]$byClass[6])
prunedectree.F1score <- c(prunedectree.F1score, prunedectree.ConfMat[[4]]$byClass[7])
prunedectree.ConfMat[[4]] <- prunedectree.ConfMat[[4]]$table
prunedectree.ROCcurv[[4]] <- roc(groups[[4]]$DEATH_EVENT, prunedectree.valid)
prunedectree.AUCsc <- c(prunedectree.AUCsc, auc(groups[[4]]$DEATH_EVENT, prunedectree.valid))


##5th fold-out training
trainset <- data.frame()
for (i in c(1:4,6:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[5]])
for (j in 1:length(simplelinear.valid)) {
  if(simplelinear.valid[j] < 0.5){
    simplelinear.valid[j] <- 0
  }
  if(simplelinear.valid[j] >= 0.5){
    simplelinear.valid[j] <- 1
  }
}
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[5]]))
simplelinear.ConfMat[[5]] <- confusionMatrix(factor(simplelinear.valid), factor(groups[[5]]$DEATH_EVENT))
simplelinear.accur <- c(simplelinear.accur, simplelinear.ConfMat[[5]]$overall[1])
simplelinear.precis <- c(simplelinear.precis, simplelinear.ConfMat[[5]]$byClass[5])
simplelinear.recall <- c(simplelinear.recall, simplelinear.ConfMat[[5]]$byClass[6])
simplelinear.F1score <- c(simplelinear.F1score, simplelinear.ConfMat[[5]]$byClass[7])
simplelinear.ConfMat[[5]] <- simplelinear.ConfMat[[5]]$table
simplelinear.ROCcurv[[5]] <- roc(groups[[5]]$DEATH_EVENT, simplelinear.valid)
simplelinear.AUCsc <- c(simplelinear.AUCsc, auc(groups[[5]]$DEATH_EVENT, simplelinear.valid))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[5]])
for (j in 1:length(revisedlinear.valid)) {
  if(revisedlinear.valid[j] < 0.5){
    revisedlinear.valid[j] <- 0
  }
  if(revisedlinear.valid[j] >= 0.5){
    revisedlinear.valid[j] <- 1
  }
}
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[5]]))
revisedlinear.ConfMat[[5]] <- confusionMatrix(factor(revisedlinear.valid), factor(groups[[5]]$DEATH_EVENT))
revisedlinear.accur <- c(revisedlinear.accur, revisedlinear.ConfMat[[5]]$overall[1])
revisedlinear.precis <- c(revisedlinear.precis, revisedlinear.ConfMat[[5]]$byClass[5])
revisedlinear.recall <- c(revisedlinear.recall, revisedlinear.ConfMat[[5]]$byClass[6])
revisedlinear.F1score <- c(revisedlinear.F1score, revisedlinear.ConfMat[[5]]$byClass[7])
revisedlinear.ConfMat[[5]] <- revisedlinear.ConfMat[[5]]$table
revisedlinear.ROCcurv[[5]] <- roc(groups[[5]]$DEATH_EVENT, revisedlinear.valid)
revisedlinear.AUCsc <- c(revisedlinear.AUCsc, auc(groups[[5]]$DEATH_EVENT, revisedlinear.valid))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[5]], type = 'response')
for (j in 1:length(generlinear.valid)) {
  if(generlinear.valid[j] < 0.5){
    generlinear.valid[j] <- 0
  }
  if(generlinear.valid[j] >= 0.5){
    generlinear.valid[j] <- 1
  }
}
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[5]]))
generlinear.ConfMat[[5]] <- confusionMatrix(factor(generlinear.valid), factor(groups[[5]]$DEATH_EVENT))
generlinear.accur <- c(generlinear.accur, generlinear.ConfMat[[5]]$overall[1])
generlinear.precis <- c(generlinear.precis, generlinear.ConfMat[[5]]$byClass[5])
generlinear.recall <- c(generlinear.recall, generlinear.ConfMat[[5]]$byClass[6])
generlinear.F1score <- c(generlinear.F1score, generlinear.ConfMat[[5]]$byClass[7])
generlinear.ConfMat[[5]] <- generlinear.ConfMat[[5]]$table
generlinear.ROCcurv[[5]] <- roc(groups[[5]]$DEATH_EVENT, generlinear.valid)
generlinear.AUCsc <- c(generlinear.AUCsc, auc(groups[[5]]$DEATH_EVENT, generlinear.valid))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[5]], type = 'response')
for (j in 1:length(generaddit.valid)) {
  if(generaddit.valid[j] < 0.5){
    generaddit.valid[j] <- 0
  }
  if(generaddit.valid[j] >= 0.5){
    generaddit.valid[j] <- 1
  }
}
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[5]]))
generaddit.ConfMat[[5]] <- confusionMatrix(factor(generaddit.valid), factor(groups[[5]]$DEATH_EVENT))
generaddit.accur <- c(generaddit.accur, generaddit.ConfMat[[5]]$overall[1])
generaddit.precis <- c(generaddit.precis, generaddit.ConfMat[[5]]$byClass[5])
generaddit.recall <- c(generaddit.recall, generaddit.ConfMat[[5]]$byClass[6])
generaddit.F1score <- c(generaddit.F1score, generaddit.ConfMat[[5]]$byClass[7])
generaddit.ConfMat[[5]] <- generaddit.ConfMat[[5]]$table
generaddit.ROCcurv[[5]] <- roc(groups[[5]]$DEATH_EVENT, generaddit.valid)
generaddit.AUCsc <- c(generaddit.AUCsc, auc(groups[[5]]$DEATH_EVENT, generaddit.valid))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[5]], type = 'response')
lindiscr.valid <- as.numeric(lindiscr.valid$class)-1
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[5]]))
lindiscr.ConfMat[[5]] <- confusionMatrix(factor(lindiscr.valid), factor(groups[[5]]$DEATH_EVENT))
lindiscr.accur <- c(lindiscr.accur, lindiscr.ConfMat[[5]]$overall[1])
lindiscr.precis <- c(lindiscr.precis, lindiscr.ConfMat[[5]]$byClass[5])
lindiscr.recall <- c(lindiscr.recall, lindiscr.ConfMat[[5]]$byClass[6])
lindiscr.F1score <- c(lindiscr.F1score, lindiscr.ConfMat[[5]]$byClass[7])
lindiscr.ConfMat[[5]] <- lindiscr.ConfMat[[5]]$table
lindiscr.ROCcurv[[5]] <- roc(groups[[5]]$DEATH_EVENT, lindiscr.valid)
lindiscr.AUCsc <- c(lindiscr.AUCsc, auc(groups[[5]]$DEATH_EVENT, lindiscr.valid))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[5]], type = 'response')
quaddiscr.valid <- as.numeric(quaddiscr.valid$class)-1
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[5]]))
quaddiscr.ConfMat[[5]] <- confusionMatrix(factor(quaddiscr.valid), factor(groups[[5]]$DEATH_EVENT))
quaddiscr.accur <- c(quaddiscr.accur, quaddiscr.ConfMat[[5]]$overall[1])
quaddiscr.precis <- c(quaddiscr.precis, quaddiscr.ConfMat[[5]]$byClass[5])
quaddiscr.recall <- c(quaddiscr.recall, quaddiscr.ConfMat[[5]]$byClass[6])
quaddiscr.F1score <- c(quaddiscr.F1score, quaddiscr.ConfMat[[5]]$byClass[7])
quaddiscr.ConfMat[[5]] <- quaddiscr.ConfMat[[5]]$table
quaddiscr.ROCcurv[[5]] <- roc(groups[[5]]$DEATH_EVENT, quaddiscr.valid)
quaddiscr.AUCsc <- c(quaddiscr.AUCsc, auc(groups[[5]]$DEATH_EVENT, quaddiscr.valid))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[5]], type = 'class')
mixeddiscr.valid <- as.numeric(mixeddiscr.valid)-1
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[5]]))
mixeddiscr.ConfMat[[5]] <- confusionMatrix(factor(mixeddiscr.valid), factor(groups[[5]]$DEATH_EVENT))
mixeddiscr.accur <- c(mixeddiscr.accur, mixeddiscr.ConfMat[[5]]$overall[1])
mixeddiscr.precis <- c(mixeddiscr.precis, mixeddiscr.ConfMat[[5]]$byClass[5])
mixeddiscr.recall <- c(mixeddiscr.recall, mixeddiscr.ConfMat[[5]]$byClass[6])
mixeddiscr.F1score <- c(mixeddiscr.F1score, mixeddiscr.ConfMat[[5]]$byClass[7])
mixeddiscr.ConfMat[[5]] <- mixeddiscr.ConfMat[[5]]$table
mixeddiscr.ROCcurv[[5]] <- roc(groups[[5]]$DEATH_EVENT, mixeddiscr.valid)
mixeddiscr.AUCsc <- c(mixeddiscr.AUCsc, auc(groups[[5]]$DEATH_EVENT, mixeddiscr.valid))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[5]], type = 'class')
flexdiscr.valid <- as.numeric(flexdiscr.valid)-1
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[5]]))
flexdiscr.ConfMat[[5]] <- confusionMatrix(factor(flexdiscr.valid), factor(groups[[5]]$DEATH_EVENT))
flexdiscr.accur <- c(flexdiscr.accur, flexdiscr.ConfMat[[5]]$overall[1])
flexdiscr.precis <- c(flexdiscr.precis, flexdiscr.ConfMat[[5]]$byClass[5])
flexdiscr.recall <- c(flexdiscr.recall, flexdiscr.ConfMat[[5]]$byClass[6])
flexdiscr.F1score <- c(flexdiscr.F1score, flexdiscr.ConfMat[[5]]$byClass[7])
flexdiscr.ConfMat[[5]] <- flexdiscr.ConfMat[[5]]$table
flexdiscr.ROCcurv[[5]] <- roc(groups[[5]]$DEATH_EVENT, flexdiscr.valid)
flexdiscr.AUCsc <- c(flexdiscr.AUCsc, auc(groups[[5]]$DEATH_EVENT, flexdiscr.valid))

decisiontree.model <- tree(as.factor(DEATH_EVENT) ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[5]], type = 'class')
decisiontree.valid <- as.numeric(decisiontree.valid)-1
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[5]]))
decisiontree.ConfMat[[5]] <- confusionMatrix(factor(decisiontree.valid), factor(groups[[5]]$DEATH_EVENT))
decisiontree.accur <- c(decisiontree.accur, decisiontree.ConfMat[[5]]$overall[1])
decisiontree.precis <- c(decisiontree.precis, decisiontree.ConfMat[[5]]$byClass[5])
decisiontree.recall <- c(decisiontree.recall, decisiontree.ConfMat[[5]]$byClass[6])
decisiontree.F1score <- c(decisiontree.F1score, decisiontree.ConfMat[[5]]$byClass[7])
decisiontree.ConfMat[[5]] <- decisiontree.ConfMat[[5]]$table
decisiontree.ROCcurv[[5]] <- roc(groups[[5]]$DEATH_EVENT, decisiontree.valid)
decisiontree.AUCsc <- c(decisiontree.AUCsc, auc(groups[[5]]$DEATH_EVENT, decisiontree.valid))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[5]], type = 'class')
prunedectree.valid <- as.numeric(prunedectree.valid)-1
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[5]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[5]]))
prunedectree.ConfMat[[5]] <- confusionMatrix(factor(prunedectree.valid), factor(groups[[5]]$DEATH_EVENT))
prunedectree.accur <- c(prunedectree.accur, prunedectree.ConfMat[[5]]$overall[1])
prunedectree.precis <- c(prunedectree.precis, prunedectree.ConfMat[[5]]$byClass[5])
prunedectree.recall <- c(prunedectree.recall, prunedectree.ConfMat[[5]]$byClass[6])
prunedectree.F1score <- c(prunedectree.F1score, prunedectree.ConfMat[[5]]$byClass[7])
prunedectree.ConfMat[[5]] <- prunedectree.ConfMat[[5]]$table
prunedectree.ROCcurv[[5]] <- roc(groups[[5]]$DEATH_EVENT, prunedectree.valid)
prunedectree.AUCsc <- c(prunedectree.AUCsc, auc(groups[[5]]$DEATH_EVENT, prunedectree.valid))


##6th fold-out training
trainset <- data.frame()
for (i in c(1:5,7:8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[6]])
for (j in 1:length(simplelinear.valid)) {
  if(simplelinear.valid[j] < 0.5){
    simplelinear.valid[j] <- 0
  }
  if(simplelinear.valid[j] >= 0.5){
    simplelinear.valid[j] <- 1
  }
}
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[6]]))
simplelinear.ConfMat[[6]] <- confusionMatrix(factor(simplelinear.valid), factor(groups[[6]]$DEATH_EVENT))
simplelinear.accur <- c(simplelinear.accur, simplelinear.ConfMat[[6]]$overall[1])
simplelinear.precis <- c(simplelinear.precis, simplelinear.ConfMat[[6]]$byClass[5])
simplelinear.recall <- c(simplelinear.recall, simplelinear.ConfMat[[6]]$byClass[6])
simplelinear.F1score <- c(simplelinear.F1score, simplelinear.ConfMat[[6]]$byClass[7])
simplelinear.ConfMat[[6]] <- simplelinear.ConfMat[[6]]$table
simplelinear.ROCcurv[[6]] <- roc(groups[[6]]$DEATH_EVENT, simplelinear.valid)
simplelinear.AUCsc <- c(simplelinear.AUCsc, auc(groups[[6]]$DEATH_EVENT, simplelinear.valid))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[6]])
for (j in 1:length(revisedlinear.valid)) {
  if(revisedlinear.valid[j] < 0.5){
    revisedlinear.valid[j] <- 0
  }
  if(revisedlinear.valid[j] >= 0.5){
    revisedlinear.valid[j] <- 1
  }
}
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[6]]))
revisedlinear.ConfMat[[6]] <- confusionMatrix(factor(revisedlinear.valid), factor(groups[[6]]$DEATH_EVENT))
revisedlinear.accur <- c(revisedlinear.accur, revisedlinear.ConfMat[[6]]$overall[1])
revisedlinear.precis <- c(revisedlinear.precis, revisedlinear.ConfMat[[6]]$byClass[5])
revisedlinear.recall <- c(revisedlinear.recall, revisedlinear.ConfMat[[6]]$byClass[6])
revisedlinear.F1score <- c(revisedlinear.F1score, revisedlinear.ConfMat[[6]]$byClass[7])
revisedlinear.ConfMat[[6]] <- revisedlinear.ConfMat[[6]]$table
revisedlinear.ROCcurv[[6]] <- roc(groups[[6]]$DEATH_EVENT, revisedlinear.valid)
revisedlinear.AUCsc <- c(revisedlinear.AUCsc, auc(groups[[6]]$DEATH_EVENT, revisedlinear.valid))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[6]], type = 'response')
for (j in 1:length(generlinear.valid)) {
  if(generlinear.valid[j] < 0.5){
    generlinear.valid[j] <- 0
  }
  if(generlinear.valid[j] >= 0.5){
    generlinear.valid[j] <- 1
  }
}
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[6]]))
generlinear.ConfMat[[6]] <- confusionMatrix(factor(generlinear.valid), factor(groups[[6]]$DEATH_EVENT))
generlinear.accur <- c(generlinear.accur, generlinear.ConfMat[[6]]$overall[1])
generlinear.precis <- c(generlinear.precis, generlinear.ConfMat[[6]]$byClass[5])
generlinear.recall <- c(generlinear.recall, generlinear.ConfMat[[6]]$byClass[6])
generlinear.F1score <- c(generlinear.F1score, generlinear.ConfMat[[6]]$byClass[7])
generlinear.ConfMat[[6]] <- generlinear.ConfMat[[6]]$table
generlinear.ROCcurv[[6]] <- roc(groups[[6]]$DEATH_EVENT, generlinear.valid)
generlinear.AUCsc <- c(generlinear.AUCsc, auc(groups[[6]]$DEATH_EVENT, generlinear.valid))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[6]], type = 'response')
for (j in 1:length(generaddit.valid)) {
  if(generaddit.valid[j] < 0.5){
    generaddit.valid[j] <- 0
  }
  if(generaddit.valid[j] >= 0.5){
    generaddit.valid[j] <- 1
  }
}
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[6]]))
generaddit.ConfMat[[6]] <- confusionMatrix(factor(generaddit.valid), factor(groups[[6]]$DEATH_EVENT))
generaddit.accur <- c(generaddit.accur, generaddit.ConfMat[[6]]$overall[1])
generaddit.precis <- c(generaddit.precis, generaddit.ConfMat[[6]]$byClass[5])
generaddit.recall <- c(generaddit.recall, generaddit.ConfMat[[6]]$byClass[6])
generaddit.F1score <- c(generaddit.F1score, generaddit.ConfMat[[6]]$byClass[7])
generaddit.ConfMat[[6]] <- generaddit.ConfMat[[6]]$table
generaddit.ROCcurv[[6]] <- roc(groups[[6]]$DEATH_EVENT, generaddit.valid)
generaddit.AUCsc <- c(generaddit.AUCsc, auc(groups[[6]]$DEATH_EVENT, generaddit.valid))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[6]], type = 'response')
lindiscr.valid <- as.numeric(lindiscr.valid$class)-1
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[6]]))
lindiscr.ConfMat[[6]] <- confusionMatrix(factor(lindiscr.valid), factor(groups[[6]]$DEATH_EVENT))
lindiscr.accur <- c(lindiscr.accur, lindiscr.ConfMat[[6]]$overall[1])
lindiscr.precis <- c(lindiscr.precis, lindiscr.ConfMat[[6]]$byClass[5])
lindiscr.recall <- c(lindiscr.recall, lindiscr.ConfMat[[6]]$byClass[6])
lindiscr.F1score <- c(lindiscr.F1score, lindiscr.ConfMat[[6]]$byClass[7])
lindiscr.ConfMat[[6]] <- lindiscr.ConfMat[[6]]$table
lindiscr.ROCcurv[[6]] <- roc(groups[[6]]$DEATH_EVENT, lindiscr.valid)
lindiscr.AUCsc <- c(lindiscr.AUCsc, auc(groups[[6]]$DEATH_EVENT, lindiscr.valid))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[6]], type = 'response')
quaddiscr.valid <- as.numeric(quaddiscr.valid$class)-1
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[6]]))
quaddiscr.ConfMat[[6]] <- confusionMatrix(factor(quaddiscr.valid), factor(groups[[6]]$DEATH_EVENT))
quaddiscr.accur <- c(quaddiscr.accur, quaddiscr.ConfMat[[6]]$overall[1])
quaddiscr.precis <- c(quaddiscr.precis, quaddiscr.ConfMat[[6]]$byClass[5])
quaddiscr.recall <- c(quaddiscr.recall, quaddiscr.ConfMat[[6]]$byClass[6])
quaddiscr.F1score <- c(quaddiscr.F1score, quaddiscr.ConfMat[[6]]$byClass[7])
quaddiscr.ConfMat[[6]] <- quaddiscr.ConfMat[[6]]$table
quaddiscr.ROCcurv[[6]] <- roc(groups[[6]]$DEATH_EVENT, quaddiscr.valid)
quaddiscr.AUCsc <- c(quaddiscr.AUCsc, auc(groups[[6]]$DEATH_EVENT, quaddiscr.valid))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[6]], type = 'class')
mixeddiscr.valid <- as.numeric(mixeddiscr.valid)-1
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[6]]))
mixeddiscr.ConfMat[[6]] <- confusionMatrix(factor(mixeddiscr.valid), factor(groups[[6]]$DEATH_EVENT))
mixeddiscr.accur <- c(mixeddiscr.accur, mixeddiscr.ConfMat[[6]]$overall[1])
mixeddiscr.precis <- c(mixeddiscr.precis, mixeddiscr.ConfMat[[6]]$byClass[5])
mixeddiscr.recall <- c(mixeddiscr.recall, mixeddiscr.ConfMat[[6]]$byClass[6])
mixeddiscr.F1score <- c(mixeddiscr.F1score, mixeddiscr.ConfMat[[6]]$byClass[7])
mixeddiscr.ConfMat[[6]] <- mixeddiscr.ConfMat[[6]]$table
mixeddiscr.ROCcurv[[6]] <- roc(groups[[6]]$DEATH_EVENT, mixeddiscr.valid)
mixeddiscr.AUCsc <- c(mixeddiscr.AUCsc, auc(groups[[6]]$DEATH_EVENT, mixeddiscr.valid))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[6]], type = 'class')
flexdiscr.valid <- as.numeric(flexdiscr.valid)-1
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[6]]))
flexdiscr.ConfMat[[6]] <- confusionMatrix(factor(flexdiscr.valid), factor(groups[[6]]$DEATH_EVENT))
flexdiscr.accur <- c(flexdiscr.accur, flexdiscr.ConfMat[[6]]$overall[1])
flexdiscr.precis <- c(flexdiscr.precis, flexdiscr.ConfMat[[6]]$byClass[5])
flexdiscr.recall <- c(flexdiscr.recall, flexdiscr.ConfMat[[6]]$byClass[6])
flexdiscr.F1score <- c(flexdiscr.F1score, flexdiscr.ConfMat[[6]]$byClass[7])
flexdiscr.ConfMat[[6]] <- flexdiscr.ConfMat[[6]]$table
flexdiscr.ROCcurv[[6]] <- roc(groups[[6]]$DEATH_EVENT, flexdiscr.valid)
flexdiscr.AUCsc <- c(flexdiscr.AUCsc, auc(groups[[6]]$DEATH_EVENT, flexdiscr.valid))

decisiontree.model <- tree(as.factor(DEATH_EVENT) ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[6]], type = 'class')
decisiontree.valid <- as.numeric(decisiontree.valid)-1
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[6]]))
decisiontree.ConfMat[[6]] <- confusionMatrix(factor(decisiontree.valid), factor(groups[[6]]$DEATH_EVENT))
decisiontree.accur <- c(decisiontree.accur, decisiontree.ConfMat[[6]]$overall[1])
decisiontree.precis <- c(decisiontree.precis, decisiontree.ConfMat[[6]]$byClass[5])
decisiontree.recall <- c(decisiontree.recall, decisiontree.ConfMat[[6]]$byClass[6])
decisiontree.F1score <- c(decisiontree.F1score, decisiontree.ConfMat[[6]]$byClass[7])
decisiontree.ConfMat[[6]] <- decisiontree.ConfMat[[6]]$table
decisiontree.ROCcurv[[6]] <- roc(groups[[6]]$DEATH_EVENT, decisiontree.valid)
decisiontree.AUCsc <- c(decisiontree.AUCsc, auc(groups[[6]]$DEATH_EVENT, decisiontree.valid))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[6]], type = 'class')
prunedectree.valid <- as.numeric(prunedectree.valid)-1
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[6]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[6]]))
prunedectree.ConfMat[[6]] <- confusionMatrix(factor(prunedectree.valid), factor(groups[[6]]$DEATH_EVENT))
prunedectree.accur <- c(prunedectree.accur, prunedectree.ConfMat[[6]]$overall[1])
prunedectree.precis <- c(prunedectree.precis, prunedectree.ConfMat[[6]]$byClass[5])
prunedectree.recall <- c(prunedectree.recall, prunedectree.ConfMat[[6]]$byClass[6])
prunedectree.F1score <- c(prunedectree.F1score, prunedectree.ConfMat[[6]]$byClass[7])
prunedectree.ConfMat[[6]] <- prunedectree.ConfMat[[6]]$table
prunedectree.ROCcurv[[6]] <- roc(groups[[6]]$DEATH_EVENT, prunedectree.valid)
prunedectree.AUCsc <- c(prunedectree.AUCsc, auc(groups[[6]]$DEATH_EVENT, prunedectree.valid))


##7th fold-out training
trainset <- data.frame()
for (i in c(1:6, 8)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[7]])
for (j in 1:length(simplelinear.valid)) {
  if(simplelinear.valid[j] < 0.5){
    simplelinear.valid[j] <- 0
  }
  if(simplelinear.valid[j] >= 0.5){
    simplelinear.valid[j] <- 1
  }
}
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[7]]))
simplelinear.ConfMat[[7]] <- confusionMatrix(factor(simplelinear.valid), factor(groups[[7]]$DEATH_EVENT))
simplelinear.accur <- c(simplelinear.accur, simplelinear.ConfMat[[7]]$overall[1])
simplelinear.precis <- c(simplelinear.precis, simplelinear.ConfMat[[7]]$byClass[5])
simplelinear.recall <- c(simplelinear.recall, simplelinear.ConfMat[[7]]$byClass[6])
simplelinear.F1score <- c(simplelinear.F1score, simplelinear.ConfMat[[7]]$byClass[7])
simplelinear.ConfMat[[7]] <- simplelinear.ConfMat[[7]]$table
simplelinear.ROCcurv[[7]] <- roc(groups[[7]]$DEATH_EVENT, simplelinear.valid)
simplelinear.AUCsc <- c(simplelinear.AUCsc, auc(groups[[7]]$DEATH_EVENT, simplelinear.valid))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[7]])
for (j in 1:length(revisedlinear.valid)) {
  if(revisedlinear.valid[j] < 0.5){
    revisedlinear.valid[j] <- 0
  }
  if(revisedlinear.valid[j] >= 0.5){
    revisedlinear.valid[j] <- 1
  }
}
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[7]]))
revisedlinear.ConfMat[[7]] <- confusionMatrix(factor(revisedlinear.valid), factor(groups[[7]]$DEATH_EVENT))
revisedlinear.accur <- c(revisedlinear.accur, revisedlinear.ConfMat[[7]]$overall[1])
revisedlinear.precis <- c(revisedlinear.precis, revisedlinear.ConfMat[[7]]$byClass[5])
revisedlinear.recall <- c(revisedlinear.recall, revisedlinear.ConfMat[[7]]$byClass[6])
revisedlinear.F1score <- c(revisedlinear.F1score, revisedlinear.ConfMat[[7]]$byClass[7])
revisedlinear.ConfMat[[7]] <- revisedlinear.ConfMat[[7]]$table
revisedlinear.ROCcurv[[7]] <- roc(groups[[7]]$DEATH_EVENT, revisedlinear.valid)
revisedlinear.AUCsc <- c(revisedlinear.AUCsc, auc(groups[[7]]$DEATH_EVENT, revisedlinear.valid))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[7]], type = 'response')
for (j in 1:length(generlinear.valid)) {
  if(generlinear.valid[j] < 0.5){
    generlinear.valid[j] <- 0
  }
  if(generlinear.valid[j] >= 0.5){
    generlinear.valid[j] <- 1
  }
}
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[7]]))
generlinear.ConfMat[[7]] <- confusionMatrix(factor(generlinear.valid), factor(groups[[7]]$DEATH_EVENT))
generlinear.accur <- c(generlinear.accur, generlinear.ConfMat[[7]]$overall[1])
generlinear.precis <- c(generlinear.precis, generlinear.ConfMat[[7]]$byClass[5])
generlinear.recall <- c(generlinear.recall, generlinear.ConfMat[[7]]$byClass[6])
generlinear.F1score <- c(generlinear.F1score, generlinear.ConfMat[[7]]$byClass[7])
generlinear.ConfMat[[7]] <- generlinear.ConfMat[[7]]$table
generlinear.ROCcurv[[7]] <- roc(groups[[7]]$DEATH_EVENT, generlinear.valid)
generlinear.AUCsc <- c(generlinear.AUCsc, auc(groups[[7]]$DEATH_EVENT, generlinear.valid))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[7]], type = 'response')
for (j in 1:length(generaddit.valid)) {
  if(generaddit.valid[j] < 0.5){
    generaddit.valid[j] <- 0
  }
  if(generaddit.valid[j] >= 0.5){
    generaddit.valid[j] <- 1
  }
}
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[7]]))
generaddit.ConfMat[[7]] <- confusionMatrix(factor(generaddit.valid), factor(groups[[7]]$DEATH_EVENT))
generaddit.accur <- c(generaddit.accur, generaddit.ConfMat[[7]]$overall[1])
generaddit.precis <- c(generaddit.precis, generaddit.ConfMat[[7]]$byClass[5])
generaddit.recall <- c(generaddit.recall, generaddit.ConfMat[[7]]$byClass[6])
generaddit.F1score <- c(generaddit.F1score, generaddit.ConfMat[[7]]$byClass[7])
generaddit.ConfMat[[7]] <- generaddit.ConfMat[[7]]$table
generaddit.ROCcurv[[7]] <- roc(groups[[7]]$DEATH_EVENT, generaddit.valid)
generaddit.AUCsc <- c(generaddit.AUCsc, auc(groups[[7]]$DEATH_EVENT, generaddit.valid))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[7]], type = 'response')
lindiscr.valid <- as.numeric(lindiscr.valid$class)-1
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[7]]))
lindiscr.ConfMat[[7]] <- confusionMatrix(factor(lindiscr.valid), factor(groups[[7]]$DEATH_EVENT))
lindiscr.accur <- c(lindiscr.accur, lindiscr.ConfMat[[7]]$overall[1])
lindiscr.precis <- c(lindiscr.precis, lindiscr.ConfMat[[7]]$byClass[5])
lindiscr.recall <- c(lindiscr.recall, lindiscr.ConfMat[[7]]$byClass[6])
lindiscr.F1score <- c(lindiscr.F1score, lindiscr.ConfMat[[7]]$byClass[7])
lindiscr.ConfMat[[7]] <- lindiscr.ConfMat[[7]]$table
lindiscr.ROCcurv[[7]] <- roc(groups[[7]]$DEATH_EVENT, lindiscr.valid)
lindiscr.AUCsc <- c(lindiscr.AUCsc, auc(groups[[7]]$DEATH_EVENT, lindiscr.valid))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[7]], type = 'response')
quaddiscr.valid <- as.numeric(quaddiscr.valid$class)-1
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[7]]))
quaddiscr.ConfMat[[7]] <- confusionMatrix(factor(quaddiscr.valid), factor(groups[[7]]$DEATH_EVENT))
quaddiscr.accur <- c(quaddiscr.accur, quaddiscr.ConfMat[[7]]$overall[1])
quaddiscr.precis <- c(quaddiscr.precis, quaddiscr.ConfMat[[7]]$byClass[5])
quaddiscr.recall <- c(quaddiscr.recall, quaddiscr.ConfMat[[7]]$byClass[6])
quaddiscr.F1score <- c(quaddiscr.F1score, quaddiscr.ConfMat[[7]]$byClass[7])
quaddiscr.ConfMat[[7]] <- quaddiscr.ConfMat[[7]]$table
quaddiscr.ROCcurv[[7]] <- roc(groups[[7]]$DEATH_EVENT, quaddiscr.valid)
quaddiscr.AUCsc <- c(quaddiscr.AUCsc, auc(groups[[7]]$DEATH_EVENT, quaddiscr.valid))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[7]], type = 'class')
mixeddiscr.valid <- as.numeric(mixeddiscr.valid)-1
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[7]]))
mixeddiscr.ConfMat[[7]] <- confusionMatrix(factor(mixeddiscr.valid), factor(groups[[7]]$DEATH_EVENT))
mixeddiscr.accur <- c(mixeddiscr.accur, mixeddiscr.ConfMat[[7]]$overall[1])
mixeddiscr.precis <- c(mixeddiscr.precis, mixeddiscr.ConfMat[[7]]$byClass[5])
mixeddiscr.recall <- c(mixeddiscr.recall, mixeddiscr.ConfMat[[7]]$byClass[6])
mixeddiscr.F1score <- c(mixeddiscr.F1score, mixeddiscr.ConfMat[[7]]$byClass[7])
mixeddiscr.ConfMat[[7]] <- mixeddiscr.ConfMat[[7]]$table
mixeddiscr.ROCcurv[[7]] <- roc(groups[[7]]$DEATH_EVENT, mixeddiscr.valid)
mixeddiscr.AUCsc <- c(mixeddiscr.AUCsc, auc(groups[[7]]$DEATH_EVENT, mixeddiscr.valid))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[7]], type = 'class')
flexdiscr.valid <- as.numeric(flexdiscr.valid)-1
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[7]]))
flexdiscr.ConfMat[[7]] <- confusionMatrix(factor(flexdiscr.valid), factor(groups[[7]]$DEATH_EVENT))
flexdiscr.accur <- c(flexdiscr.accur, flexdiscr.ConfMat[[7]]$overall[1])
flexdiscr.precis <- c(flexdiscr.precis, flexdiscr.ConfMat[[7]]$byClass[5])
flexdiscr.recall <- c(flexdiscr.recall, flexdiscr.ConfMat[[7]]$byClass[6])
flexdiscr.F1score <- c(flexdiscr.F1score, flexdiscr.ConfMat[[7]]$byClass[7])
flexdiscr.ConfMat[[7]] <- flexdiscr.ConfMat[[7]]$table
flexdiscr.ROCcurv[[7]] <- roc(groups[[7]]$DEATH_EVENT, flexdiscr.valid)
flexdiscr.AUCsc <- c(flexdiscr.AUCsc, auc(groups[[7]]$DEATH_EVENT, flexdiscr.valid))

decisiontree.model <- tree(as.factor(DEATH_EVENT) ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[7]], type = 'class')
decisiontree.valid <- as.numeric(decisiontree.valid)-1
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[7]]))
decisiontree.ConfMat[[7]] <- confusionMatrix(factor(decisiontree.valid), factor(groups[[7]]$DEATH_EVENT))
decisiontree.accur <- c(decisiontree.accur, decisiontree.ConfMat[[7]]$overall[1])
decisiontree.precis <- c(decisiontree.precis, decisiontree.ConfMat[[7]]$byClass[5])
decisiontree.recall <- c(decisiontree.recall, decisiontree.ConfMat[[7]]$byClass[6])
decisiontree.F1score <- c(decisiontree.F1score, decisiontree.ConfMat[[7]]$byClass[7])
decisiontree.ConfMat[[7]] <- decisiontree.ConfMat[[7]]$table
decisiontree.ROCcurv[[7]] <- roc(groups[[7]]$DEATH_EVENT, decisiontree.valid)
decisiontree.AUCsc <- c(decisiontree.AUCsc, auc(groups[[7]]$DEATH_EVENT, decisiontree.valid))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[7]], type = 'class')
prunedectree.valid <- as.numeric(prunedectree.valid)-1
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[7]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[7]]))
prunedectree.ConfMat[[7]] <- confusionMatrix(factor(prunedectree.valid), factor(groups[[7]]$DEATH_EVENT))
prunedectree.accur <- c(prunedectree.accur, prunedectree.ConfMat[[7]]$overall[1])
prunedectree.precis <- c(prunedectree.precis, prunedectree.ConfMat[[7]]$byClass[5])
prunedectree.recall <- c(prunedectree.recall, prunedectree.ConfMat[[7]]$byClass[6])
prunedectree.F1score <- c(prunedectree.F1score, prunedectree.ConfMat[[7]]$byClass[7])
prunedectree.ConfMat[[7]] <- prunedectree.ConfMat[[7]]$table
prunedectree.ROCcurv[[7]] <- roc(groups[[7]]$DEATH_EVENT, prunedectree.valid)
prunedectree.AUCsc <- c(prunedectree.AUCsc, auc(groups[[7]]$DEATH_EVENT, prunedectree.valid))


##8th fold-out training
trainset <- data.frame()
for (i in c(1:7)) {
  trainset <- rbind(trainset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.valid <- predict(simplelinear.model, groups[[8]])
for (j in 1:length(simplelinear.valid)) {
  if(simplelinear.valid[j] < 0.5){
    simplelinear.valid[j] <- 0
  }
  if(simplelinear.valid[j] >= 0.5){
    simplelinear.valid[j] <- 1
  }
}
simplelinear.valid
simplelinear.RMSE <- c(simplelinear.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - simplelinear.valid)^2))/nrow(groups[[8]]))
simplelinear.ConfMat[[8]] <- confusionMatrix(factor(simplelinear.valid), factor(groups[[8]]$DEATH_EVENT))
simplelinear.accur <- c(simplelinear.accur, simplelinear.ConfMat[[8]]$overall[1])
simplelinear.precis <- c(simplelinear.precis, simplelinear.ConfMat[[8]]$byClass[5])
simplelinear.recall <- c(simplelinear.recall, simplelinear.ConfMat[[8]]$byClass[6])
simplelinear.F1score <- c(simplelinear.F1score, simplelinear.ConfMat[[8]]$byClass[7])
simplelinear.ConfMat[[8]] <- simplelinear.ConfMat[[8]]$table
simplelinear.ROCcurv[[8]] <- roc(groups[[8]]$DEATH_EVENT, simplelinear.valid)
simplelinear.AUCsc <- c(simplelinear.AUCsc, auc(groups[[8]]$DEATH_EVENT, simplelinear.valid))

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.valid <- predict(revisedlinear.model, groups[[8]])
for (j in 1:length(revisedlinear.valid)) {
  if(revisedlinear.valid[j] < 0.5){
    revisedlinear.valid[j] <- 0
  }
  if(revisedlinear.valid[j] >= 0.5){
    revisedlinear.valid[j] <- 1
  }
}
revisedlinear.valid
revisedlinear.RMSE <- c(revisedlinear.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - revisedlinear.valid)^2))/nrow(groups[[8]]))
revisedlinear.ConfMat[[8]] <- confusionMatrix(factor(revisedlinear.valid), factor(groups[[8]]$DEATH_EVENT))
revisedlinear.accur <- c(revisedlinear.accur, revisedlinear.ConfMat[[8]]$overall[1])
revisedlinear.precis <- c(revisedlinear.precis, revisedlinear.ConfMat[[8]]$byClass[5])
revisedlinear.recall <- c(revisedlinear.recall, revisedlinear.ConfMat[[8]]$byClass[6])
revisedlinear.F1score <- c(revisedlinear.F1score, revisedlinear.ConfMat[[8]]$byClass[7])
revisedlinear.ConfMat[[8]] <- revisedlinear.ConfMat[[8]]$table
revisedlinear.ROCcurv[[8]] <- roc(groups[[8]]$DEATH_EVENT, revisedlinear.valid)
revisedlinear.AUCsc <- c(revisedlinear.AUCsc, auc(groups[[8]]$DEATH_EVENT, revisedlinear.valid))

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.valid <- predict(generlinear.model, groups[[8]], type = 'response')
for (j in 1:length(generlinear.valid)) {
  if(generlinear.valid[j] < 0.5){
    generlinear.valid[j] <- 0
  }
  if(generlinear.valid[j] >= 0.5){
    generlinear.valid[j] <- 1
  }
}
generlinear.valid
generlinear.RMSE <- c(generlinear.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - generlinear.valid)^2))/nrow(groups[[8]]))
generlinear.ConfMat[[8]] <- confusionMatrix(factor(generlinear.valid), factor(groups[[8]]$DEATH_EVENT))
generlinear.accur <- c(generlinear.accur, generlinear.ConfMat[[8]]$overall[1])
generlinear.precis <- c(generlinear.precis, generlinear.ConfMat[[8]]$byClass[5])
generlinear.recall <- c(generlinear.recall, generlinear.ConfMat[[8]]$byClass[6])
generlinear.F1score <- c(generlinear.F1score, generlinear.ConfMat[[8]]$byClass[7])
generlinear.ConfMat[[8]] <- generlinear.ConfMat[[8]]$table
generlinear.ROCcurv[[8]] <- roc(groups[[8]]$DEATH_EVENT, generlinear.valid)
generlinear.AUCsc <- c(generlinear.AUCsc, auc(groups[[8]]$DEATH_EVENT, generlinear.valid))

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.valid <- predict(generaddit.model, groups[[8]], type = 'response')
for (j in 1:length(generaddit.valid)) {
  if(generaddit.valid[j] < 0.5){
    generaddit.valid[j] <- 0
  }
  if(generaddit.valid[j] >= 0.5){
    generaddit.valid[j] <- 1
  }
}
generaddit.valid
generaddit.RMSE <- c(generaddit.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - generaddit.valid)^2))/nrow(groups[[8]]))
generaddit.ConfMat[[8]] <- confusionMatrix(factor(generaddit.valid), factor(groups[[8]]$DEATH_EVENT))
generaddit.accur <- c(generaddit.accur, generaddit.ConfMat[[8]]$overall[1])
generaddit.precis <- c(generaddit.precis, generaddit.ConfMat[[8]]$byClass[5])
generaddit.recall <- c(generaddit.recall, generaddit.ConfMat[[8]]$byClass[6])
generaddit.F1score <- c(generaddit.F1score, generaddit.ConfMat[[8]]$byClass[7])
generaddit.ConfMat[[8]] <- generaddit.ConfMat[[8]]$table
generaddit.ROCcurv[[8]] <- roc(groups[[8]]$DEATH_EVENT, generaddit.valid)
generaddit.AUCsc <- c(generaddit.AUCsc, auc(groups[[8]]$DEATH_EVENT, generaddit.valid))

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.valid <- predict(lindiscr.model, groups[[8]], type = 'response')
lindiscr.valid <- as.numeric(lindiscr.valid$class)-1
lindiscr.valid
lindiscr.RMSE <- c(lindiscr.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - lindiscr.valid)^2))/nrow(groups[[8]]))
lindiscr.ConfMat[[8]] <- confusionMatrix(factor(lindiscr.valid), factor(groups[[8]]$DEATH_EVENT))
lindiscr.accur <- c(lindiscr.accur, lindiscr.ConfMat[[8]]$overall[1])
lindiscr.precis <- c(lindiscr.precis, lindiscr.ConfMat[[8]]$byClass[5])
lindiscr.recall <- c(lindiscr.recall, lindiscr.ConfMat[[8]]$byClass[6])
lindiscr.F1score <- c(lindiscr.F1score, lindiscr.ConfMat[[8]]$byClass[7])
lindiscr.ConfMat[[8]] <- lindiscr.ConfMat[[8]]$table
lindiscr.ROCcurv[[8]] <- roc(groups[[8]]$DEATH_EVENT, lindiscr.valid)
lindiscr.AUCsc <- c(lindiscr.AUCsc, auc(groups[[8]]$DEATH_EVENT, lindiscr.valid))

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.valid <- predict(quaddiscr.model, groups[[8]], type = 'response')
quaddiscr.valid <- as.numeric(quaddiscr.valid$class)-1
quaddiscr.valid
quaddiscr.RMSE <- c(quaddiscr.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - quaddiscr.valid)^2))/nrow(groups[[8]]))
quaddiscr.ConfMat[[8]] <- confusionMatrix(factor(quaddiscr.valid), factor(groups[[8]]$DEATH_EVENT))
quaddiscr.accur <- c(quaddiscr.accur, quaddiscr.ConfMat[[8]]$overall[1])
quaddiscr.precis <- c(quaddiscr.precis, quaddiscr.ConfMat[[8]]$byClass[5])
quaddiscr.recall <- c(quaddiscr.recall, quaddiscr.ConfMat[[8]]$byClass[6])
quaddiscr.F1score <- c(quaddiscr.F1score, quaddiscr.ConfMat[[8]]$byClass[7])
quaddiscr.ConfMat[[8]] <- quaddiscr.ConfMat[[8]]$table
quaddiscr.ROCcurv[[8]] <- roc(groups[[8]]$DEATH_EVENT, quaddiscr.valid)
quaddiscr.AUCsc <- c(quaddiscr.AUCsc, auc(groups[[8]]$DEATH_EVENT, quaddiscr.valid))

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.valid <- predict(mixeddiscr.model, groups[[8]], type = 'class')
mixeddiscr.valid <- as.numeric(mixeddiscr.valid)-1
mixeddiscr.valid
mixeddiscr.RMSE <- c(mixeddiscr.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - mixeddiscr.valid)^2))/nrow(groups[[8]]))
mixeddiscr.ConfMat[[8]] <- confusionMatrix(factor(mixeddiscr.valid), factor(groups[[8]]$DEATH_EVENT))
mixeddiscr.accur <- c(mixeddiscr.accur, mixeddiscr.ConfMat[[8]]$overall[1])
mixeddiscr.precis <- c(mixeddiscr.precis, mixeddiscr.ConfMat[[8]]$byClass[5])
mixeddiscr.recall <- c(mixeddiscr.recall, mixeddiscr.ConfMat[[8]]$byClass[6])
mixeddiscr.F1score <- c(mixeddiscr.F1score, mixeddiscr.ConfMat[[8]]$byClass[7])
mixeddiscr.ConfMat[[8]] <- mixeddiscr.ConfMat[[8]]$table
mixeddiscr.ROCcurv[[8]] <- roc(groups[[8]]$DEATH_EVENT, mixeddiscr.valid)
mixeddiscr.AUCsc <- c(mixeddiscr.AUCsc, auc(groups[[8]]$DEATH_EVENT, mixeddiscr.valid))

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.valid <- predict(flexdiscr.model, groups[[8]], type = 'class')
flexdiscr.valid <- as.numeric(flexdiscr.valid)-1
flexdiscr.valid
flexdiscr.RMSE <- c(flexdiscr.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - flexdiscr.valid)^2))/nrow(groups[[8]]))
flexdiscr.ConfMat[[8]] <- confusionMatrix(factor(flexdiscr.valid), factor(groups[[8]]$DEATH_EVENT))
flexdiscr.accur <- c(flexdiscr.accur, flexdiscr.ConfMat[[8]]$overall[1])
flexdiscr.precis <- c(flexdiscr.precis, flexdiscr.ConfMat[[8]]$byClass[5])
flexdiscr.recall <- c(flexdiscr.recall, flexdiscr.ConfMat[[8]]$byClass[6])
flexdiscr.F1score <- c(flexdiscr.F1score, flexdiscr.ConfMat[[8]]$byClass[7])
flexdiscr.ConfMat[[8]] <- flexdiscr.ConfMat[[8]]$table
flexdiscr.ROCcurv[[8]] <- roc(groups[[8]]$DEATH_EVENT, flexdiscr.valid)
flexdiscr.AUCsc <- c(flexdiscr.AUCsc, auc(groups[[8]]$DEATH_EVENT, flexdiscr.valid))

decisiontree.model <- tree(as.factor(DEATH_EVENT) ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.valid <- predict(decisiontree.model, groups[[8]], type = 'class')
decisiontree.valid <- as.numeric(decisiontree.valid)-1
decisiontree.valid
decisiontree.RMSE <- c(decisiontree.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - decisiontree.valid)^2))/nrow(groups[[8]]))
decisiontree.ConfMat[[8]] <- confusionMatrix(factor(decisiontree.valid), factor(groups[[8]]$DEATH_EVENT))
decisiontree.accur <- c(decisiontree.accur, decisiontree.ConfMat[[8]]$overall[1])
decisiontree.precis <- c(decisiontree.precis, decisiontree.ConfMat[[8]]$byClass[5])
decisiontree.recall <- c(decisiontree.recall, decisiontree.ConfMat[[8]]$byClass[6])
decisiontree.F1score <- c(decisiontree.F1score, decisiontree.ConfMat[[8]]$byClass[7])
decisiontree.ConfMat[[8]] <- decisiontree.ConfMat[[8]]$table
decisiontree.ROCcurv[[8]] <- roc(groups[[8]]$DEATH_EVENT, decisiontree.valid)
decisiontree.AUCsc <- c(decisiontree.AUCsc, auc(groups[[8]]$DEATH_EVENT, decisiontree.valid))

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.valid <- predict(prunedectree.model, groups[[8]], type = 'class')
prunedectree.valid <- as.numeric(prunedectree.valid)-1
prunedectree.valid
prunedectree.RMSE <- c(prunedectree.RMSE, sqrt(sum((groups[[8]]$DEATH_EVENT - prunedectree.valid)^2))/nrow(groups[[8]]))
prunedectree.ConfMat[[8]] <- confusionMatrix(factor(prunedectree.valid), factor(groups[[8]]$DEATH_EVENT))
prunedectree.accur <- c(prunedectree.accur, prunedectree.ConfMat[[8]]$overall[1])
prunedectree.precis <- c(prunedectree.precis, prunedectree.ConfMat[[8]]$byClass[5])
prunedectree.recall <- c(prunedectree.recall, prunedectree.ConfMat[[8]]$byClass[6])
prunedectree.F1score <- c(prunedectree.F1score, prunedectree.ConfMat[[8]]$byClass[7])
prunedectree.ConfMat[[8]] <- prunedectree.ConfMat[[8]]$table
prunedectree.ROCcurv[[8]] <- roc(groups[[8]]$DEATH_EVENT, prunedectree.valid)
prunedectree.AUCsc <- c(prunedectree.AUCsc, auc(groups[[8]]$DEATH_EVENT, prunedectree.valid))


##Model comparison over training cross-validation
sum(simplelinear.RMSE)/8
sum(simplelinear.accur)/8
sum(simplelinear.precis)/8
sum(simplelinear.recall)/8
sum(simplelinear.F1score)/8
sum(simplelinear.AUCsc)/8
ggroc(simplelinear.ROCcurv, aes = c('colour', linetype = 1, size = 2))+
  ggtitle('ROC curves from 8-fold cross-validation for simplelinear.model')+
  labs(colour = 'fold used for CV')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)+
  theme(legend.position = c(0.89, 0.25))
simplelinear.ConfMat

sum(revisedlinear.RMSE)/8
sum(revisedlinear.accur)/8
sum(revisedlinear.precis)/8
sum(revisedlinear.recall)/8
sum(revisedlinear.F1score)/8
sum(revisedlinear.AUCsc)/8
ggroc(revisedlinear.ROCcurv, aes = c('colour', linetype = 1, size = 2))+
  ggtitle('ROC curves from 8-fold cross-validation for revisedlinear.model')+
  labs(colour = 'fold used for CV')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)+
  theme(legend.position = c(0.89, 0.25))
revisedlinear.ConfMat

sum(generlinear.RMSE)/8
sum(generlinear.accur)/8
sum(generlinear.precis)/8
sum(generlinear.recall)/8
sum(generlinear.F1score)/8
sum(generlinear.AUCsc)/8
ggroc(generlinear.ROCcurv, aes = c('colour', linetype = 1, size = 2))+
  ggtitle('ROC curves from 8-fold cross-validation for generlinear.model')+
  labs(colour = 'fold used for CV')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)+
  theme(legend.position = c(0.89, 0.25))
generlinear.ConfMat

sum(generaddit.RMSE)/8
sum(generaddit.accur)/8
sum(generaddit.precis)/8
sum(generaddit.recall)/8
sum(generaddit.F1score)/8
sum(generaddit.AUCsc)/8
ggroc(generaddit.ROCcurv, aes = c('colour', linetype = 1, size = 2))+
  ggtitle('ROC curves from 8-fold cross-validation for generaddit.model')+
  labs(colour = 'fold used for CV')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)+
  theme(legend.position = c(0.89, 0.25))
generaddit.ConfMat

sum(lindiscr.RMSE)/8
sum(lindiscr.accur)/8
sum(lindiscr.precis)/8
sum(lindiscr.recall)/8
sum(lindiscr.F1score)/8
sum(lindiscr.AUCsc)/8
ggroc(lindiscr.ROCcurv, aes = c('colour', linetype = 1, size = 2))+
  ggtitle('ROC curves from 8-fold cross-validation for lindiscr.model')+
  labs(colour = 'fold used for CV')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)+
  theme(legend.position = c(0.89, 0.25))
lindiscr.ConfMat

sum(quaddiscr.RMSE)/8
sum(quaddiscr.accur)/8
sum(quaddiscr.precis)/8
sum(quaddiscr.recall)/8
sum(quaddiscr.F1score)/8
sum(quaddiscr.AUCsc)/8
ggroc(quaddiscr.ROCcurv, aes = c('colour', linetype = 1, size = 2))+
  ggtitle('ROC curves from 8-fold cross-validation for quaddiscr.model')+
  labs(colour = 'fold used for CV')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)+
  theme(legend.position = c(0.89, 0.25))
quaddiscr.ConfMat

sum(mixeddiscr.RMSE)/8
sum(mixeddiscr.accur)/8
sum(mixeddiscr.precis)/8
sum(mixeddiscr.recall)/8
sum(mixeddiscr.F1score)/8
sum(mixeddiscr.AUCsc)/8
ggroc(mixeddiscr.ROCcurv, aes = c('colour', linetype = 1, size = 2))+
  ggtitle('ROC curves from 8-fold cross-validation for mixeddiscr.model')+
  labs(colour = 'fold used for CV')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)+
  theme(legend.position = c(0.89, 0.25))
mixeddiscr.ConfMat

sum(flexdiscr.RMSE)/8
sum(flexdiscr.accur)/8
sum(flexdiscr.precis)/8
sum(flexdiscr.recall)/8
sum(flexdiscr.F1score)/8
sum(flexdiscr.AUCsc)/8
ggroc(flexdiscr.ROCcurv, aes = c('colour', linetype = 1, size = 2))+
  ggtitle('ROC curves from 8-fold cross-validation for flexdiscr.model')+
  labs(colour = 'fold used for CV')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)+
  theme(legend.position = c(0.89, 0.25))
flexdiscr.ConfMat

sum(decisiontree.RMSE)/8
sum(decisiontree.accur)/8
sum(decisiontree.precis)/8
sum(decisiontree.recall)/8
sum(decisiontree.F1score)/8
sum(decisiontree.AUCsc)/8
ggroc(decisiontree.ROCcurv, aes = c('colour', linetype = 1, size = 2))+
  ggtitle('ROC curves from 8-fold cross-validation for decisiontree.model')+
  labs(colour = 'fold used for CV')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)+
  theme(legend.position = c(0.89, 0.25))
decisiontree.ConfMat

sum(prunedectree.RMSE)/8
sum(prunedectree.accur)/8
sum(prunedectree.precis)/8
sum(prunedectree.recall)/8
sum(prunedectree.F1score)/8
sum(prunedectree.AUCsc)/8
ggroc(prunedectree.ROCcurv, aes = c('colour', linetype = 1, size = 2))+
  ggtitle('ROC curves from 8-fold cross-validation for prunedectree.model')+
  labs(colour = 'fold used for CV')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)+
  theme(legend.position = c(0.89, 0.25))
prunedectree.ConfMat

##Conclusions from Model Comparision over Training Cross-Validation
##From what we see on our graphs, and the average AUC score over 8-fold
##Cross-Validation, the best models in this case are the Revised Linear
##model, Generalized Additive model, Decision Tree model, & Pruned Decision
##Tree model (who all have average AUC above 0.6). We must however note that
##these models are very far from perfect (a random guess model would have n 
##AUC score near 0.5 while the perfect model would have a near or equal to 1
##AUC score).
##Thus, from our cross-validation, it doesn't seem like our classifications
##models are usable in any way from cross-validation results. Let us apply this
##on our test just to see if any significant changes appear

##Applied to test
trainset <- data.frame()
for (i in c(1:8)) {
  trainset <- rbind(trainset, groups[[i]])
}
testset <- data.frame()
for (i in c(9,10)) {
  testset <- rbind(testset, groups[[i]])
}

simplelinear.model <- lm(DEATH_EVENT ~ . -time, data = trainset)
simplelinear.model
summary(simplelinear.model)
anova(simplelinear.model)
simplelinear.test <- predict(simplelinear.model, testset)
for (j in 1:length(simplelinear.test)) {
  if(simplelinear.test[j] < 0.5){
    simplelinear.test[j] <- 0
  }
  if(simplelinear.test[j] >= 0.5){
    simplelinear.test[j] <- 1
  }
}
simplelinear.test
simplelinear.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - simplelinear.test)^2))/nrow(testset)
simplelinear.test.ConfMat <- confusionMatrix(factor(simplelinear.test), factor(testset$DEATH_EVENT))
simplelinear.test.accur <- simplelinear.test.ConfMat$overall[1]
simplelinear.test.precis <- simplelinear.test.ConfMat$byClass[5]
simplelinear.test.recall <- simplelinear.test.ConfMat$byClass[6]
simplelinear.test.F1score <- simplelinear.test.ConfMat$byClass[7]
simplelinear.test.ConfMat <- simplelinear.test.ConfMat$table
simplelinear.test.ROCcurv <- roc(testset$DEATH_EVENT, simplelinear.test)
simplelinear.test.AUCsc <- auc(testset$DEATH_EVENT, simplelinear.test)

revisedlinear.model <- lm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine, data = trainset)
revisedlinear.model
summary(revisedlinear.model)
anova(revisedlinear.model)
revisedlinear.test <- predict(revisedlinear.model, testset)
for (j in 1:length(revisedlinear.test)) {
  if(revisedlinear.test[j] < 0.5){
    revisedlinear.test[j] <- 0
  }
  if(revisedlinear.test[j] >= 0.5){
    revisedlinear.test[j] <- 1
  }
}
revisedlinear.test
revisedlinear.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - revisedlinear.test)^2))/nrow(testset)
revisedlinear.test.ConfMat <- confusionMatrix(factor(revisedlinear.test), factor(testset$DEATH_EVENT))
revisedlinear.test.accur <- revisedlinear.test.ConfMat$overall[1]
revisedlinear.test.precis <- revisedlinear.test.ConfMat$byClass[5]
revisedlinear.test.recall <- revisedlinear.test.ConfMat$byClass[6]
revisedlinear.test.F1score <- revisedlinear.test.ConfMat$byClass[7]
revisedlinear.test.ConfMat <- revisedlinear.test.ConfMat$table
revisedlinear.test.ROCcurv <- roc(testset$DEATH_EVENT, revisedlinear.test)
revisedlinear.test.AUCsc <- auc(testset$DEATH_EVENT, revisedlinear.test)

generlinear.model <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium, family = binomial(link = "logit"), data = trainset)
generlinear.model
summary(generlinear.model)
anova(generlinear.model)
generlinear.test <- predict(generlinear.model, testset, type = 'response')
for (j in 1:length(generlinear.test)) {
  if(generlinear.test[j] < 0.5){
    generlinear.test[j] <- 0
  }
  if(generlinear.test[j] >= 0.5){
    generlinear.test[j] <- 1
  }
}
generlinear.test
generlinear.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - generlinear.test)^2))/nrow(testset)
generlinear.test.ConfMat <- confusionMatrix(factor(generlinear.test), factor(testset$DEATH_EVENT))
generlinear.test.accur <- generlinear.test.ConfMat$overall[1]
generlinear.test.precis <- generlinear.test.ConfMat$byClass[5]
generlinear.test.recall <- generlinear.test.ConfMat$byClass[6]
generlinear.test.F1score <- generlinear.test.ConfMat$byClass[7]
generlinear.test.ConfMat <- generlinear.test.ConfMat$table
generlinear.test.ROCcurv <- roc(testset$DEATH_EVENT, generlinear.test)
generlinear.test.AUCsc <- auc(testset$DEATH_EVENT, generlinear.test)

generaddit.model <- gam(DEATH_EVENT ~ s(age)+ s(creatinine_phosphokinase) + s(ejection_fraction) + s(serum_creatinine), data = trainset, method = 'REML')
generaddit.model
summary(generaddit.model)
generaddit.test <- predict(generaddit.model, testset, type = 'response')
for (j in 1:length(generaddit.test)) {
  if(generaddit.test[j] < 0.5){
    generaddit.test[j] <- 0
  }
  if(generaddit.test[j] >= 0.5){
    generaddit.test[j] <- 1
  }
}
generaddit.test
generaddit.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - generaddit.test)^2))/nrow(testset)
generaddit.test.ConfMat <- confusionMatrix(factor(generaddit.test), factor(testset$DEATH_EVENT))
generaddit.test.accur <- generaddit.test.ConfMat$overall[1]
generaddit.test.precis <- generaddit.test.ConfMat$byClass[5]
generaddit.test.recall <- generaddit.test.ConfMat$byClass[6]
generaddit.test.F1score <- generaddit.test.ConfMat$byClass[7]
generaddit.test.ConfMat <- generaddit.test.ConfMat$table
generaddit.test.ROCcurv <- roc(testset$DEATH_EVENT, generaddit.test)
generaddit.test.AUCsc <- auc(testset$DEATH_EVENT, generaddit.test)

lindiscr.model <- lda(DEATH_EVENT ~ . -time, data = trainset)
lindiscr.model
lindiscr.test <- predict(lindiscr.model, testset, type = 'response')
lindiscr.test <- as.numeric(lindiscr.test$class)-1
lindiscr.test
lindiscr.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - lindiscr.test)^2))/nrow(testset)
lindiscr.test.ConfMat <- confusionMatrix(factor(lindiscr.test), factor(testset$DEATH_EVENT))
lindiscr.test.accur <- lindiscr.test.ConfMat$overall[1]
lindiscr.test.precis <- lindiscr.test.ConfMat$byClass[5]
lindiscr.test.recall <- lindiscr.test.ConfMat$byClass[6]
lindiscr.test.F1score <- lindiscr.test.ConfMat$byClass[7]
lindiscr.test.ConfMat <- lindiscr.test.ConfMat$table
lindiscr.test.ROCcurv <- roc(testset$DEATH_EVENT, lindiscr.test)
lindiscr.test.AUCsc <- auc(testset$DEATH_EVENT, lindiscr.test)

quaddiscr.model <- qda(DEATH_EVENT ~ . -time, data = trainset)
quaddiscr.model
quaddiscr.test <- predict(quaddiscr.model, testset, type = 'response')
quaddiscr.test <- as.numeric(quaddiscr.test$class)-1
quaddiscr.test
quaddiscr.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - quaddiscr.test)^2))/nrow(testset)
quaddiscr.test.ConfMat <- confusionMatrix(factor(quaddiscr.test), factor(testset$DEATH_EVENT))
quaddiscr.test.accur <- quaddiscr.test.ConfMat$overall[1]
quaddiscr.test.precis <- quaddiscr.test.ConfMat$byClass[5]
quaddiscr.test.recall <- quaddiscr.test.ConfMat$byClass[6]
quaddiscr.test.F1score <- quaddiscr.test.ConfMat$byClass[7]
quaddiscr.test.ConfMat <- quaddiscr.test.ConfMat$table
quaddiscr.test.ROCcurv <- roc(testset$DEATH_EVENT, quaddiscr.test)
quaddiscr.test.AUCsc <- auc(testset$DEATH_EVENT, quaddiscr.test)

mixeddiscr.model <- mda(DEATH_EVENT ~ . -time, data = trainset)
mixeddiscr.model
mixeddiscr.test <- predict(mixeddiscr.model, testset, type = 'class')
mixeddiscr.test <- as.numeric(mixeddiscr.test)-1
mixeddiscr.test
mixeddiscr.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - mixeddiscr.test)^2))/nrow(testset)
mixeddiscr.test.ConfMat <- confusionMatrix(factor(mixeddiscr.test), factor(testset$DEATH_EVENT))
mixeddiscr.test.accur <- mixeddiscr.test.ConfMat$overall[1]
mixeddiscr.test.precis <- mixeddiscr.test.ConfMat$byClass[5]
mixeddiscr.test.recall <- mixeddiscr.test.ConfMat$byClass[6]
mixeddiscr.test.F1score <- mixeddiscr.test.ConfMat$byClass[7]
mixeddiscr.test.ConfMat <- mixeddiscr.test.ConfMat$table
mixeddiscr.test.ROCcurv <- roc(testset$DEATH_EVENT, mixeddiscr.test)
mixeddiscr.test.AUCsc <- auc(testset$DEATH_EVENT, mixeddiscr.test)

flexdiscr.model <- fda(DEATH_EVENT ~ . -time, data = trainset)
flexdiscr.model
flexdiscr.test <- predict(flexdiscr.model, testset, type = 'class')
flexdiscr.test <- as.numeric(flexdiscr.test)-1
flexdiscr.test
flexdiscr.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - flexdiscr.test)^2))/nrow(testset)
flexdiscr.test.ConfMat <- confusionMatrix(factor(flexdiscr.test), factor(testset$DEATH_EVENT))
flexdiscr.test.accur <- flexdiscr.test.ConfMat$overall[1]
flexdiscr.test.precis <- flexdiscr.test.ConfMat$byClass[5]
flexdiscr.test.recall <- flexdiscr.test.ConfMat$byClass[6]
flexdiscr.test.F1score <- flexdiscr.test.ConfMat$byClass[7]
flexdiscr.test.ConfMat <- flexdiscr.test.ConfMat$table
flexdiscr.test.ROCcurv <- roc(testset$DEATH_EVENT, flexdiscr.test)
flexdiscr.test.AUCsc <- auc(testset$DEATH_EVENT, flexdiscr.test)

decisiontree.model <- tree(as.factor(DEATH_EVENT) ~ . -time, data = trainset)
plot(decisiontree.model)
text(decisiontree.model, pretty = 0)
decisiontree.test <- predict(decisiontree.model, testset, type = 'class')
decisiontree.test <- as.numeric(decisiontree.test)-1
decisiontree.test
decisiontree.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - decisiontree.test)^2))/nrow(testset)
decisiontree.test.ConfMat <- confusionMatrix(factor(decisiontree.test), factor(testset$DEATH_EVENT))
decisiontree.test.accur <- decisiontree.test.ConfMat$overall[1]
decisiontree.test.precis <- decisiontree.test.ConfMat$byClass[5]
decisiontree.test.recall <- decisiontree.test.ConfMat$byClass[6]
decisiontree.test.F1score <- decisiontree.test.ConfMat$byClass[7]
decisiontree.test.ConfMat <- decisiontree.test.ConfMat$table
decisiontree.test.ROCcurv <- roc(testset$DEATH_EVENT, decisiontree.test)
decisiontree.test.AUCsc <- auc(testset$DEATH_EVENT, decisiontree.test)

set.seed(5)
decisiontree.cv <- cv.tree(decisiontree.model)
plot(decisiontree.cv$size, decisiontree.cv$dev, type = 'b')
prunedectree.model <- prune.tree(decisiontree.model, best = decisiontree.cv$size[which.min(decisiontree.cv$dev)])
plot(prunedectree.model)
text(prunedectree.model, pretty = 0)
prunedectree.test <- predict(prunedectree.model, testset, type = 'class')
prunedectree.test <- as.numeric(prunedectree.test)-1
prunedectree.test
prunedectree.test.RMSE <- sqrt(sum((testset$DEATH_EVENT - prunedectree.test)^2))/nrow(testset)
prunedectree.test.ConfMat <- confusionMatrix(factor(prunedectree.test), factor(testset$DEATH_EVENT))
prunedectree.test.accur <- prunedectree.test.ConfMat$overall[1]
prunedectree.test.precis <- prunedectree.test.ConfMat$byClass[5]
prunedectree.test.recall <- prunedectree.test.ConfMat$byClass[6]
prunedectree.test.F1score <- prunedectree.test.ConfMat$byClass[7]
prunedectree.test.ConfMat <- prunedectree.test.ConfMat$table
prunedectree.test.ROCcurv <- roc(testset$DEATH_EVENT, prunedectree.test)
prunedectree.test.AUCsc <- auc(testset$DEATH_EVENT, prunedectree.test)

##Let us check results for our models once again

simplelinear.test.RMSE
simplelinear.test.accur
simplelinear.test.precis
simplelinear.test.recall
simplelinear.test.F1score
simplelinear.test.AUCsc
ggroc(simplelinear.test.ROCcurv, linetype = 1, size = 2)+
  ggtitle('ROC curve from test set for simplelinear.model')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)
simplelinear.test.ConfMat

revisedlinear.test.RMSE
revisedlinear.test.accur
revisedlinear.test.precis
revisedlinear.test.recall
revisedlinear.test.F1score
revisedlinear.test.AUCsc
ggroc(revisedlinear.test.ROCcurv, linetype = 1, size = 2)+
  ggtitle('ROC curve from test set for revisedlinear.model')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)
revisedlinear.test.ConfMat

generlinear.test.RMSE
generlinear.test.accur
generlinear.test.precis
generlinear.test.recall
generlinear.test.F1score
generlinear.test.AUCsc
ggroc(generlinear.test.ROCcurv, linetype = 1, size = 2)+
  ggtitle('ROC curve from test set for generlinear.model')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)
generlinear.test.ConfMat

generaddit.test.RMSE
generaddit.test.accur
generaddit.test.precis
generaddit.test.recall
generaddit.test.F1score
generaddit.test.AUCsc
ggroc(generaddit.test.ROCcurv, linetype = 1, size = 2)+
  ggtitle('ROC curve from test set for generaddit.model')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)
generaddit.test.ConfMat

lindiscr.test.RMSE
lindiscr.test.accur
lindiscr.test.precis
lindiscr.test.recall
lindiscr.test.F1score
lindiscr.test.AUCsc
ggroc(lindiscr.test.ROCcurv, linetype = 1, size = 2)+
  ggtitle('ROC curve from test set for lindiscr.model')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)
lindiscr.test.ConfMat

quaddiscr.test.RMSE
quaddiscr.test.accur
quaddiscr.test.precis
quaddiscr.test.recall
quaddiscr.test.F1score
quaddiscr.test.AUCsc
ggroc(quaddiscr.test.ROCcurv, linetype = 1, size = 2)+
  ggtitle('ROC curve from test set for quaddiscr.model')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)
quaddiscr.test.ConfMat

mixeddiscr.test.RMSE
mixeddiscr.test.accur
mixeddiscr.test.precis
mixeddiscr.test.recall
mixeddiscr.test.F1score
mixeddiscr.test.AUCsc
ggroc(mixeddiscr.test.ROCcurv, linetype = 1, size = 2)+
  ggtitle('ROC curve from test set for mixeddiscr.model')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)
mixeddiscr.test.ConfMat

flexdiscr.test.RMSE
flexdiscr.test.accur
flexdiscr.test.precis
flexdiscr.test.recall
flexdiscr.test.F1score
flexdiscr.test.AUCsc
ggroc(flexdiscr.test.ROCcurv, linetype = 1, size = 2)+
  ggtitle('ROC curve from test set for flexdiscr.model')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)
flexdiscr.test.ConfMat

decisiontree.test.RMSE
decisiontree.test.accur
decisiontree.test.precis
decisiontree.test.recall
decisiontree.test.F1score
decisiontree.test.AUCsc
ggroc(decisiontree.test.ROCcurv, linetype = 1, size = 2)+
  ggtitle('ROC curve from test set for decisiontree.model')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)
decisiontree.test.ConfMat

prunedectree.test.RMSE
prunedectree.test.accur
prunedectree.test.precis
prunedectree.test.recall
prunedectree.test.F1score
prunedectree.test.AUCsc
ggroc(prunedectree.test.ROCcurv, linetype = 1, size = 2)+
  ggtitle('ROC curve from test set for prunedectree.model')+
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), colour = 'black', linetype = 2)
prunedectree.test.ConfMat

##Conclusions from Tet Set
##As discussed above in the Cross Validation conclusions, it seems that
##there are no truly appropriate models here to classify the event of a
##death from heart failure. We can however note that all models have a
##F1 (harmonic) mean near 80% which is decent, but not amazing; and AUC
##scores 0.7 which gives the same conclusions. Since these methods rely
##on the same mathematical machine learning principals used for our
##probability regression, we can most likely assume that while our
##predicitive models are satisfactarily appropriate, they may also not be
##perfect probability predictors

##Ideally, this dataset would be enlarged in the future with more data being
##sporadically added (as this might help improve the models). It may also be
##interesting to attempt any kind of clustering analysis to see if there
##significant differences between clusters (whether it be predictor values
##or actual death event outcomes) before using our machine learning models
##(this may lead to better improved, but would also depend on cluster sizes;
##where a cluster being too small outweighs model improvement)
