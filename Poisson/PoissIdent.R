library(e1071)
library(svrpath)
library(Metrics)
library(nnet)
library(randomForest)


##### this is the function for generating data in case of logistic regression with two coefficients poisson_log
poissident=function(n=10,shift0=0,shift1=0,number_iter=1){
  # creating a sequence of number of length n with last number 1 and step 0.1
  x=as.numeric(seq(0.1,1,length.out=n))
  beta0=rep(3,n)+shift0*0.375
  beta1=rep(2,n)+shift1*0.177
  mu=(beta0+beta1*x)
  # for identity mu =(beta0+beta1*x)/for sqrt mu = (beta0+beta1*x)^2
  # store a value of number of iterations
  nite=number_iter
  # creating a matrix of NA values dimension of (number_iter X n)
  y=matrix(NA,nite,n)
  for (j in 1:nite) {
    for (i in 1:length(mu)) {
      y[j,i]=rpois(1,mu[i])
    }
  }
  # calculate a mean value of matrix y
  ymean=apply(y, 1,mean)
  # creating matrix of NA values for storing coefficients of corresponding model
  coef=matrix(NA,nite,2)
  for (i in 1:nite) {
    # response matrix of values y 
    ynew=matrix(y[i,],n,1)
    # model matrix first column are 1s and second column sequence defined before
    xstar=t(rbind(rep(1,n),x))
    # glm.fit is a function with which you can estimate GLM parameters for any distribution and with any link function
    glmmodel=glm.fit(xstar,ynew,family = poisson(link = "identity"),start = c(0.5,0.5))
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
x_NN_in=poissident(n=10,number_iter=1350)
# adding to the generated data column of 0s (label in control data?)
x_NN_in_0=cbind(x_NN_in,0)

#out of control data
# when shift - out control data
set.seed(11)
# generating data with shift0 equal to 0.1
x_NN_out_shif0=poissident(n=10,shift0=0.1,shift1=0,number_iter=150)
set.seed(12)
# generating data with shift1 equal to 0.1
x_NN_out_shif1=poissident(n=10,shift0=0,shift1=0.1,number_iter=150)
set.seed(13)
# generating data with shift0 and shift1 equal to 0.1
x_NN_out_both=poissident(n=10,shift0=0.1,shift1=0.1,number_iter=150)

#out of control data
# when shift - out control data
set.seed(14)
# generating data with shift0 equal to 0.1
x_NN_out_shif0=poissident(n=10,shift0=0.5,shift1=0,number_iter=150)
set.seed(15)
# generating data with shift1 equal to 0.1
x_NN_out_shif1=poissident(n=10,shift0=0,shift1=0.5,number_iter=150)
set.seed(16)
# generating data with shift0 and shift1 equal to 0.1
x_NN_out_both=poissident(n=10,shift0=0.5,shift1=0.5,number_iter=150)

#out of control data
# when shift - out control data
set.seed(17)
# generating data with shift0 equal to 0.1
x_NN_out_shif0=poissident(n=10,shift0=1,shift1=0,number_iter=150)
set.seed(18)
# generating data with shift1 equal to 0.1
x_NN_out_shif1=poissident(n=10,shift0=0,shift1=1,number_iter=150)
set.seed(19)
# generating data with shift0 and shift1 equal to 0.1
x_NN_out_both=poissident(n=10,shift0=1,shift1=1,number_iter=150)

# binding all data together
x_NN_out=rbind(x_NN_out_shif0,x_NN_out_shif1,x_NN_out_both)
# adding to the generated data column of 1s (label out control data?)
x_NN_out_1=cbind(x_NN_out,1)

#in control and out of control combined
input_NN=rbind(x_NN_in_0,x_NN_out_1)

# change column names 
colnames(input_NN)=c("ymean","beta0","beta1","Y")

SVR_poissident <- matrix(NA,2,5)
colnames(SVR_poissident) <- c("Type","RMSE","ARL","SDRL","UCL")
SVR_poissident[1,1] <- "Eps linear"
SVR_poissident[2,1] <- "Eps radial"

set.seed(111)
gen_ucl=poissident(n=10,shift0=0,shift1=0,number_iter=10000) 
SVR = svm(formula = Y ~ .,data = input_NN, scale =FALSE,type="eps-regression" ,kernel="linear")
ucl_SVR=sort(predict(SVR,gen_ucl),decreasing = FALSE)[0.995*10000]
ucl_SVR= 0.10018725
arl=c()
s1=0
s2=0
for (k in 1:3000) {
  rl=0
  yhat=0
  while (yhat<ucl_SVR) {
    rl=rl+1
    x_NN_new=poissident(n=10,shift0=s1,shift1=s2)
    yhat=predict(SVR,x_NN_new)
  }
  arl[k]=rl
  print(paste("iter=",k," ","rl=",rl))
}


RMSE=rmse(SVR$fitted,input_NN[,4])
SVR_poissident[3,] <- c("Eps linear",RMSE,mean(arl),sd(arl),ucl_SVR)
RMSE=rmse(SVR$fitted,input_NN[,4])
SVR_poissident <- rbind(SVR_poissident, c("Eps linear",RMSE,mean(arl),sd(arl),ucl_SVR))

write.csv(SVR_poissident, "C:/Users/patri/OneDrive/Desktop/Master_thesis/Output/SVR_poissident_combo.csv", row.names=FALSE)


SVR_poissident_shift <- matrix(NA,22,7)
colnames(SVR_poissident_shift) <- c("Type","Error","ARL","SDRL","UCL","Shift 1","Shift 2")

set.seed(111)
gen_ucl=poissident(n=10,shift0=0,shift1=0,number_iter=10000) 
SVR = svm(formula = Y ~ .,data = input_NN, scale =FALSE,type="eps-regression" ,kernel="linear")
ucl_SVR=0.10018725
list_s1 <- c(0.1,0.3,0.5,0.8,1,2,0,0,0,0,0,0,0.1,0.3,0.4,0.3,0.9,0.5,0.3,0.8,1,0.8)
list_s2 <- c(0,0,0,0,0,0,0.1,0.3,0.5,0.8,1,2,0.1,0.1,0.1,0.4,0.1,0.6,0.9,0.5,0.6,1)
for (y in 1:22) {
  set.seed(111)
  arl=c()
  s1=as.numeric(list_s1[y])
  s2=as.numeric(list_s2[y])
  for (k in 1:3000) {
    rl=0
    yhat=0
    while (yhat<ucl_SVR) {
      rl=rl+1
      x_NN_new=poissident(n=10,shift0=s1,shift1=s2)
      yhat=predict(SVR,x_NN_new)
    }
    arl[k]=rl
    print(paste("iter=",k," ","rl=",rl))
  }
  SVR_poissident_shift[5,]<-c("Eps linear",SVR_poissident[1,2],mean(arl),sd(arl),ucl_SVR,s1,s2)
}

write.csv(SVR_poissident_shift, "C:/Users/patri/OneDrive/Desktop/Master_thesis/Output/SVR_poissident_combo_linear_shift.csv", row.names=FALSE)


RF_poissident <- matrix(NA,60,5)

colnames(RF_poissident) <- c("Number of trees","MSE","ARL","SDRL","UCL")
set.seed(111)
for (i in 1:60) {
  RF=randomForest(Y~ .,data=input_NN,ntree=i,importance=FALSE)
  RF_poissident[i,2] <-mean(RF$mse)
  RF_poissident[i,1] <- i
}

set.seed(111)
gen_ucl=poissident(n=10,shift0=0,shift1=0,number_iter=10000)
RF=randomForest(Y~ .,data=input_NN,ntree=44,importance=FALSE)
ucl_RF=sort(predict(RF,gen_ucl),decreasing = F)[0.995*10000]
ucl_RF =0.9341
arl=c()
s1=0
s2=0
for (k in 1:3000) {
  rl=0
  yhat=0
  while (yhat<ucl_RF) {
    rl=rl+1
    x_NN_new=poissident(n=10,shift0=s1,shift1=s2)
    yhat=predict(RF,x_NN_new)
  }
  arl[k]=rl
  print(paste("iter=",k," ","rl=",rl))
}


RF_poissident[44,c(3,4,5)]<- c(mean(arl),sd(arl),ucl_RF)
RF_poissident <- rbind(RF_poissident, c(54,RF_poissident[54,2],mean(arl),sd(arl),ucl_RF))


write.csv(RF_poissident, "C:/Users/patri/OneDrive/Desktop/Master_thesis/Output/RF_poissident.csv", row.names=FALSE)

NN_poissident <- matrix(NA,60,6)

colnames(NN_poissident) <- c("Nodes","Iterations","RMSE","ARL","SDRL","UCL")

max_iter = 100
set.seed(111)
for (i in 1:60) {
  
  if (i%%5==0){
    max_iter <- max_iter + 100
  }
  NN=nnet(Y~.,data=input_NN,size=i, linout = TRUE, maxit = max_iter)
  
  RMSE=rmse(NN$fitted.values,input_NN[,4])
  NN_poissident[i,1] <- i
  NN_poissident[i,2] <- max_iter
  NN_poissident[i,3] <- RMSE
  
}

set.seed(111)
s1=0
s2=0
arl=c()
NN=nnet(Y~.,data=input_NN,size=49, linout = TRUE, maxit =1000)
ucl_NN=sort(predict(NN,gen_ucl),decreasing = FALSE)[0.995*10000]
ucl_NN = 0.9017
for (k in 1:5000) {
  rl=0
  yhat=0
  while (yhat<ucl_NN) {
    rl=rl+1
    x_NN_new=poissident(n=10,shift0=s1,shift1=s2)
    yhat=predict(NN,x_NN_new)
  }
  arl[k]=rl
  print(paste("iter=",k," ","rl=",rl))
}

NN_poissident[49,c(4,5,6)] <-c(mean(arl),sd(arl),ucl_NN)
NN_poissident = rbind(NN_poissident,c(49,1000,NN_poissident[49,3],mean(arl),sd(arl),ucl_NN))

write.csv(NN_poissident, "C:/Users/patri/OneDrive/Desktop/Master_thesis/Output/NN_poissident.csv", row.names=FALSE)


NN_poissident[51,c(4,5,6)] <-c(mean(arl),sd(arl),ucl_NN)
