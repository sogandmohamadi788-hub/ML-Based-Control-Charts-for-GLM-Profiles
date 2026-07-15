library(e1071)
library(svrpath)
library(Metrics)
library(nnet)
library(randomForest)

##### this is the function for generating data in case of logistic regression with two coefficients binomial_logit
binom_logit=function(n=10,m=30,shift0=0,shift1=0,number_iter=1){
  # creating a sequence of number of length n with last number 1 and step 0.1
  x=as.numeric(seq(0.1,1,length.out=n))
  # creating a vector of length n of numbers -2.8 +shift0*0.467 (1st coefficient)
  beta0=rep(-2.8,n)+shift0*0.4676
  # creating a vector of length n of numbers 1 +shift1*0.69 (2nd coefficient to x)
  beta1=rep(1,n)+shift1*0.6907
  pi<- exp(beta0+beta1*x)/(1+exp(beta0+beta1*x))
  # store a value of number of iterations
  nite=number_iter
  # creating a matrix of NA values dimension of (number_iter X n)
  y=matrix(NA,nite,n)
  # assigning values to matrix which are binomial distributed with m=30
  for (j in 1:nite) {
    for (i in 1:length(pi)) {
      y[j,i]=rbinom(1,30,pi[i])
    }
  }
  # calculate a mean value of matrix y
  ymean=apply(y, 1,mean)
  # vector of differences between m and y
  cc<- m-y
  coef=matrix(NA,nite,2)
  for (i in 1:nite) {
    # response matrix of values y and cc
    ynew=matrix(c(y[i,],cc[i,]),n,2)
    # model matrix first column are 1s and second column sequence defined before
    xstar=t(rbind(rep(1,n),x))
    # glm.fit is a function with which you can estimate GLM parameters for any distribution and with any link function
    glmmodel=glm.fit(xstar,ynew,family = binomial(link = "logit"))
    # assigning coefficients from glm model
    coef[i,]=as.numeric(glmmodel$coefficients)
  }
  # store means and estimated coefficients in matrix
  x_NN_test=cbind(ymean,coef)
  # change column names
  colnames(x_NN_test)=c("ymean","beta0","beta1")
  # return matrix of data
  return(x_NN_test)
}

############################### Generating data to train machine learning structures
# in control data
# when no shift - in control data
set.seed(10)
# generating data with function defined before
x_NN_in=binom_logit(n=10,number_iter=1350)
# adding to the generated data column of 0s (label in control data?)
x_NN_in_0=cbind(x_NN_in,0)

#out of control data
# when shift - out control data
set.seed(11)
# generating data with shift0 equal to 0.1
x_NN_out_shif0a=binom_logit(n=10,shift0=0.1,shift1=0,number_iter=150)
set.seed(12)
# generating data with shift1 equal to 0.1
x_NN_out_shif1a=binom_logit(n=10,shift0=0,shift1=0.1,number_iter=150)
set.seed(13)
# generating data with shift0 and shift1 equal to 0.1
x_NN_out_botha=binom_logit(n=10,shift0=0.1,shift1=0.1,number_iter=150)

set.seed(14)
# generating data with shift0 equal to 0.1
x_NN_out_shif0b=binom_logit(n=10,shift0=0.5,shift1=0,number_iter=150)
set.seed(15)
# generating data with shift1 equal to 0.1
x_NN_out_shif1b=binom_logit(n=10,shift0=0,shift1=0.5,number_iter=150)
set.seed(16)
# generating data with shift0 and shift1 equal to 0.1
x_NN_out_bothb=binom_logit(n=10,shift0=0.5,shift1=0.5,number_iter=150)

set.seed(17)
# generating data with shift0 equal to 0.1
x_NN_out_shif0c=binom_logit(n=10,shift0=1,shift1=0,number_iter=150)
set.seed(18)
# generating data with shift1 equal to 0.1
x_NN_out_shif1c=binom_logit(n=10,shift0=0,shift1=1,number_iter=150)
set.seed(19)
# generating data with shift0 and shift1 equal to 0.1
x_NN_out_bothc=binom_logit(n=10,shift0=1,shift1=1,number_iter=150)

# binding all data together
x_NN_out=rbind(x_NN_out_shif0a,x_NN_out_shif1a,x_NN_out_botha,
               x_NN_out_shif0b,x_NN_out_shif1b,x_NN_out_bothb,
               x_NN_out_shif0c,x_NN_out_shif1c,x_NN_out_bothc)
#in control and out of control combined
x_NN_out_1=cbind(x_NN_out,1)
input_NN=rbind(x_NN_in_0,x_NN_out_1)

# change column names 
colnames(input_NN)=c("ymean","beta0","beta1","Y")

list_s1 <- c(0.1,0.3,0.5,0.8,1,2,0,0,0,0,0,0,0.1,0.3,0.4,0.3,0.9,0.5,0.3,0.8,1,0.8)
list_s2 <- c(0,0,0,0,0,0,0.1,0.3,0.5,0.8,1,2,0.1,0.1,0.1,0.4,0.1,0.6,0.9,0.5,0.6,1)

SVR_binlogit <- matrix(NA,1,5)
colnames(SVR_binlogit) <- c("Type","RMSE","ARL","SDRL","UCL")

SVR_binlogit_shift <- matrix(NA,22,7)
colnames(SVR_binlogit_shift) <- c("Type","Error","ARL","SDRL","UCL","Shift 1","Shift 2")

set.seed(111)
SVR = svm(formula = Y ~ .,data = input_NN, scale =FALSE,type="eps-regression" ,kernel="linear")
ucl_SVR = 0.8031
arl=c()
s1=0
s2=0
for (k in 1:5000) {
  rl=0
  yhat=0
  while (yhat<ucl_SVR) {
    rl=rl+1
    x_NN_new=binom_logit(n=10,m=30,shift0=s1,shift1=s2)
    yhat=predict(SVR,x_NN_new)
  }
  arl[k]=rl
  print(paste("iter=",k," ","rl=",rl))
}

RMSE=rmse(SVR$fitted,input_NN[,4])
SVR_binlogit[1,]<-c('Eps linear',RMSE,mean(arl),sd(arl),ucl_SVR)

for (y in 1:22) {
  arl=c()
  s1=as.numeric(list_s1[y])
  s2=as.numeric(list_s2[y])
  for (k in 1:5000) {
    rl=0
    yhat=0
    while (yhat<ucl_SVR) {
      rl=rl+1
      x_NN_new=binom_logit(n=10,m=30,shift0=s1,shift1=s2)
      yhat=predict(SVR,x_NN_new)
    }
    arl[k]=rl
    print(paste("iter=",k," ","rl=",rl))
  }
  SVR_binlogit_shift[y,]<-c("Eps linear",SVR_binlogit[1,2],mean(arl),sd(arl),ucl_SVR,s1,s2)
}

RF_binlogit_combo <- matrix(NA,1,5)
colnames(RF_binlogit_combo) <- c("Number of trees","MSE","ARL","SDRL","UCL")

RF_binlogit_combo_shift <- matrix(NA,22,7)
colnames(RF_binlogit_combo_shift) <- c("Number of trees","MSE","ARL","SDRL","UCL","Shift 1","Shift 2")

set.seed(111)
RF=randomForest(Y~ .,data=input_NN,ntree=40,importance=FALSE)
ucl_RF =0.97325
arl=c()
s1=0
s2=0
for (k in 1:5000) {
  rl=0
  yhat=0
  while (yhat<ucl_RF) {
    rl=rl+1
    x_NN_new=binom_logit(n=10,m=30,shift0=s1,shift1=s2)
    yhat=predict(RF,x_NN_new)
  }
  arl[k]=rl
  print(paste("iter=",k," ","rl=",rl))
}

RF_binlogit_combo[1,] <- c(40,mean(RF$mse),mean(arl),sd(arl),ucl_RF)

for (y in 1:22) {
  arl=c()
  s1=as.numeric(list_s1[y])
  s2=as.numeric(list_s2[y])
  for (k in 1:5000) {
    rl=0
    yhat=0
    while (yhat<ucl_RF) {
      rl=rl+1
      x_NN_new=binom_logit(n=10,m=30,shift0=s1,shift1=s2)
      yhat=predict(RF,x_NN_new)
    }
    arl[k]=rl
    print(paste("iter=",k," ","rl=",rl))
  }
  RF_binlogit_combo_shift[y,] <- c(40,RF_binlogit_combo[1,2],mean(arl),sd(arl),ucl_RF,s1,s2)
}



NN_binlogit_combo <- matrix(NA,1,6)
colnames(NN_binlogit_combo) <- c("Nodes","Iterations","RMSE","ARL","SDRL","UCL")

NN_binlogit_combo_shift <- matrix(NA,22,8)

colnames(NN_binlogit_combo_shift) <- c("Nodes","Iterations","MSE","ARL","SDRL","UCL","Shift 1","Shift 2")

set.seed(111)
NN=nnet(Y~.,data=input_NN,size=7, linout = TRUE, maxit =200)
ucl_NN=0.8954
s1=0
s2=0
arl=c()
for (k in 1:5000) {
  rl=0
  yhat=0
  while (yhat<ucl_NN) {
    rl=rl+1
    x_NN_new=binom_logit(n=10,m=30,shift0=s1,shift1=s2)
    yhat=predict(NN,x_NN_new)
  }
  arl[k]=rl
  print(paste("iter=",k," ","rl=",rl))
}
RMSE=rmse(NN$fitted.values,input_NN[,4])
NN_binlogit_combo[1,] <- c(7,200,RMSE,mean(arl),sd(arl),ucl_NN)


for (y in 1:22) {
  arl=c()
  s1=as.numeric(list_s1[y])
  s2=as.numeric(list_s2[y])
  for (k in 1:5000) {
    rl=0
    yhat=0
    while (yhat<ucl_NN) {
      rl=rl+1
      x_NN_new=binom_logit(n=10,m=30,shift0=s1,shift1=s2)
      yhat=predict(NN,x_NN_new)
    }
    arl[k]=rl
    print(paste("iter=",k," ","rl=",rl))
  }
  NN_binlogit_combo_shift[y,]<- c(7,200,NN_binlogit_combo[1,3],mean(arl),sd(arl),ucl_NN,s1,s2)
}

write.csv(SVR_binlogit, "C:/R_CM/Lastcheck/SVR_binlogit_combo.csv", row.names=FALSE)
write.csv(SVR_binlogit_shift, "C:/R_CM/Lastcheck/SVR_binlogit_combo_shift.csv", row.names=FALSE)

write.csv(RF_binlogit_combo, "C:/R_CM/Lastcheck/RF_binlogit_combo.csv", row.names=FALSE)
write.csv(RF_binlogit_combo_shift, "C:/R_CM/Lastcheck/RF_binlogit_combo_shift.csv", row.names=FALSE)

write.csv(NN_binlogit_combo, "C:/R_CM/Lastcheck/NN_binlogit_combo.csv", row.names=FALSE)
write.csv(NN_binlogit_combo_shift, "C:/R_CM/Lastcheck/NN_binlogit_combo_shift.csv", row.names=FALSE)
