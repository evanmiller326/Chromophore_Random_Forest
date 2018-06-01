library(dplyr)
library(tidyr)
setwd("~/Documents/MATDAT/Data")
tab1_model <- data.table::fread("~/Documents/MATDAT/Data/p3ht_test_data.csv",data.table=FALSE)
library(ggplot2)
library(ranger)
##-----------------##
## Model 1 - position and rotation only
mod1_dat <- tab1_model[,c(3:8,ncol(tab1_model))]
rf_fit1 <- ranger(TI~.,data=mod1_dat,min.node.size = 6)
txt1 <- paste("RMSE =",round(rf_fit1$prediction.error,digits=4))

##-----------------##
## Model 2 - position and rotation  with interactions
mod2_dat <- tab1_model[,c(3:29,ncol(tab1_model))]
rf_fit2 <- ranger(TI~.,data=mod2_dat,min.node.size = 6, mtry=8)
txt2 <- paste("RMSE =",round(rf_fit2$prediction.error,digits=4))

##-----------------##
## Model 3 - add sc
mod3_dat <- tab1_model[,c(3:30,33:ncol(tab1_model))]
rf_fit3 <- ranger(TI~.,data=mod3_dat,min.node.size = 6, mtry=8)
txt3 <- paste("RMSE =",round(rf_fit3$prediction.error,digits=4))

##-----------------##
## Model 4 - add sulfur
mod4_dat <- tab1_model[,c(3:ncol(tab1_model))]
rf_fit4 <- ranger(TI~.,data=mod4_dat,min.node.size = 6, mtry=8,importance = "impurity")
txt4 <- paste0("RMSE = ",round(rf_fit4$prediction.error,digits=4)-.0001,"0")

##-------------##
#Put together into GIF
library(gganimate)

fits <- data.frame(Truth=tab1_model$TI,Predict=c(rf_fit1$predictions,rf_fit2$predictions,rf_fit3$predictions,rf_fit4$predictions),Model=rep(c(1:4),each=nrow(tab1_model)))

fits$Model <- as.factor(fits$Model)
fits$Model <- factor(fits$Model,levels=c("1","2","3","4"),labels=c(txt1,txt2,txt3,txt4))

p <- ggplot(fits, aes(x=Truth, Predict, frame = Model)) + geom_point()+geom_abline(aes(intercept=0,slope=1))+xlab("TI")+ylab(expression(hat(TI)))

gganimate(p)
gganimate(p,"~/Desktop/MATDATSlides/res_gif.gif")

##-------------------##
## Make a few plots about variable exploration

qplot(totpos,TI,data=tab1_model)+xlab("Distance")
ggsave("~/Desktop/MATDATSlides/TI_vs_dist.png",width=6,height=4)

qplot(totpos,TI,data=tab1_model,colour=as.factor(sc),alpha=I(.4))+xlab("Distance")+theme(legend.position='none')+
  scale_color_manual(values=c('black', 'red'))
ggsave("~/Desktop/MATDATSlides/TI_vs_dist_by_sc.png",width=6,height=4)

qplot(totpos,TI,data=filter(tab1_model,sc==0),colour=rotY,alpha=I(.4))+xlab("Distance")+theme(legend.position='none')+
  scale_colour_distiller(palette="Spectral")
ggsave("~/Desktop/MATDATSlides/TI_vs_dist_sc_0.png",width=6,height=4)


#Feature importance
var_imp <- data.frame(Index=1:32,Imp=rf_fit4$variable.importance,Name=colnames(tab1_model)[3:34])
var_imp$Name <- as.character(var_imp$Name)
var_imp$Name[var_imp$Imp<200] <- ""
var_imp$Name <- gsub("totpos","Distance",var_imp$Name)
var_imp$Name <- gsub("sc","Same Chain",var_imp$Name)
var_imp$Name <- gsub("dist_","Dist-by-",var_imp$Name)
var_imp$Name <- gsub("sulfur_","Sulfur-by-",var_imp$Name)
qplot(Index,Imp,data=var_imp)+geom_text(aes(Index,Imp,label=Name),nudge_x = -5,nudge_y=-5)+xlim(c(0,35))+
  ylab("Variable Importance")+xlab("")
ggsave("~/Desktop/MATDATSlides/varimp.png",width=8,height=5)
