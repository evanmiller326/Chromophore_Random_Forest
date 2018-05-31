library(ggplot2)
library(dplyr)
library(tidyr)
library(RSQLite)
#File folder 1
setwd("~/Documents/MATDAT/Data/")
#File folder 2
#setwd("~/Documents/MATDAT/Data/")

filename <- "p3ht_sc_sulfur.db"
sqlite.driver <- dbDriver("SQLite")
db <- dbConnect(sqlite.driver,dbname=filename)

dbListTables(db)
fulltab1 <- dbReadTable(db,"cryst_frame_7")

tab1 <- sample_frac(fulltab1,.1)
rm(fulltab1)
tab1_model <- tab1[,c(grep("chrom",colnames(tab1)),grep("pos",colnames(tab1)),grep("rot",colnames(tab1)))]

#Create new features (interactions between pos and rot)
int_inds <- expand.grid(3:8,3:8)
int_inds <- int_inds[which(int_inds[,1]>=int_inds[,2]),]
int_terms <- matrix(NA,nrow=nrow(tab1_model),ncol=nrow(int_inds))
names <- rep(NA,nrow(int_inds))
for(i in 1:nrow(int_inds)){
  int_terms[,i] <- tab1_model[,int_inds$Var1[i]]*tab1_model[,int_inds$Var2[i]]
  names[i] <- paste0(colnames(tab1_model)[int_inds$Var1[i]],colnames(tab1_model)[int_inds$Var2[i]])
}

colnames(int_terms) <- names
tab1_model <- cbind(tab1_model,int_terms)

tab1_model$totpos <- sqrt(tab1$posX^2+tab1$posY^2+tab1$posZ^2)
#dist one rot
#tab1_model$totpos_rotY <- tab1_model$totpos*tab1_model$rotY
#tab1_model$totpos_rotZ <- tab1_model$totpos*tab1_model$rotZ
#tab1_model$totpos_rotX <- tab1_model$totpos*tab1_model$rotX
#dist two rot
#tab1_model$totpos_rotXY <- tab1_model$totpos*tab1_model$rotYrotX
#tab1_model$totpos_rotYZ <- tab1_model$totpos*tab1_model$rotZrotY
#tab1_model$totpos_rotXZ <- tab1_model$totpos*tab1_model$rotZrotX

tab1_model$sulfur <- tab1$sulfur_distance
tab1_model$sulfur_sc <- tab1$sulfur_distance*tab1$same_chain
tab1_model$sc <- tab1$same_chain
tab1_model$dist_sc <- tab1$same_chain*tab1_model$totpos

#tab1_model$sc0_1 <- (tab1_model$sc==0 & tab1_model$totpos<6)
#tab1_model$sc01_dist <- tab1_model$sc0_1*tab1_model$totpos
tab1_model$TI <- tab1$TI

#write.csv(x = tab1_model,file = "~/Documents/MATDAT/Data/p3ht_test_data.csv",row.names = FALSE)
tab1_model_test <- tab1_model

qplot(totpos,TI,data=tab1_model,colour=sc)
qplot(totpos,TI,data=filter(tab1_model,sc==0),colour=rotY,alpha=I(.5))
qplot(totpos,TI,data=filter(tab1_model,sc==0),colour=sulfur,alpha=I(.3))
qplot(totpos,sulfur,data=filter(tab1_model,sc==1),alpha=I(.3))


##-----------##
library(dplyr)
library(tidyr)
setwd("~/Documents/MATDAT/Data")
tab1_model <- read.csv("~/Documents/MATDAT/Data/p3ht_test_data.csv")
library(ggplot2)


## rf
library(ranger)

train_dat <- tab1_model[,c(3:grep("TI",colnames(tab1_model)))]
for(i in 1:(ncol(train_dat)-1)){
  train_dat[,i] <- (train_dat[,i]-min(train_dat[,i]))/(max(train_dat[,i])-min(train_dat[,i]))
}

rf_res <- ranger(TI~.,data=train_dat,importance = "impurity",min.node.size = 6, mtry=8)
train_dat$Yhat <- rf_res$predictions
plot(train_dat$TI,train_dat$Yhat,main = rf_res$prediction.error);abline(0,1)

var_imp <- data.frame(Index=1:32,Imp=rf_res$variable.importance,Name=colnames(train_dat)[1:32])
qplot(Index,Imp,data=var_imp)+geom_text(aes(Index,Imp,label=Name),nudge_y = 7.5,angle=45)

qplot(TI,Yhat,colour=rotY,data=train_dat,alpha=I(.5))

#Try testing data
test_pred <- predict(rf_res,data=tab1_model_test)
tab1_model_test$Yhat <- test_pred$predictions
qplot(TI,Yhat,data=tab1_model_test)+geom_abline(aes(intercept=0,slope=1))

sqrt(mean((tab1_model_test$Yhat-tab1_model_test$TI)^2))

#Reduce to just chromophore A > chromophore B
tab1_modelAB <- filter(tab1_model,chromophoreA>chromophoreB)
rf_res_AB <- ranger(TI~.,data=tab1_modelAB[,c(3:grep("TI",colnames(tab1_modelAB)))],importance = "impurity")
pred <- rf_res_AB$predictions
plot(tab1_modelAB$TI,pred,main=rf_res_AB$prediction.error);abline(0,1)

var_impAB <- data.frame(Index=1:32,Imp=rf_res_AB$variable.importance,Name=colnames(tab1_modelAB)[3:34])
qplot(Index,Imp,data=var_impAB)+geom_text(aes(Index,Imp,label=Name),nudge_y = 5)


## nnet
library(nnet)
nn_res <- nnet(TI~.,data=tab1_model[,c(3:grep("TI",colnames(tab1_model)))],size=20,maxit=500)
pred <- nn_res$fitted.values[,1]
plot(tab1_model$TI,pred);abline(0,1)


#caret rf
library(caret)

rf_caret <- train(TI~.,data=tab1_model[,c(3:grep("TI",colnames(tab1_model)))],importance='impurity',
                  method='ranger',tuneLength=3)

#multivariate adaptive regression splies
library(earth)

eth_res <- earth((TI)~.,data=tab1_model[,c(3:grep("TI",colnames(tab1_model)))],degree=3,pmethod='cv',nfold=5)

plot((tab1_model$TI),eth_res$fitted.values[,1]);abline(0,1)
plot(tab1_model$TI,(eth_res$fitted.values[,1]));abline(0,1)

plot(eth_res$leverages)

tab1_model$Yhat <- eth_res$fitted.values[,1]
tab1_model$Resids <- (tab1_model$TI)-tab1_model$Yhat
tab1_model$Leverage <- eth_res$leverages

plot(tab1_model$totpos,tab1_model$Yhat)

plot(tab1_model$Leverage);abline(h=2/nrow(tab1_model))
qqnorm(tab1_model$Resids);qqline(tab1_model$Resids)

qplot(TI,Yhat,data=tab1_model,colour=Leverage)
tab1_model$Bad <- (tab1_model$Leverage>0.02)
qplot(TI,(Yhat),data=tab1_model,colour=Bad)


tab1_model$Bad <- tab1_model$Resids<(-10)
qplot(TI,(Yhat),data=tab1_model,colour=Bad)

