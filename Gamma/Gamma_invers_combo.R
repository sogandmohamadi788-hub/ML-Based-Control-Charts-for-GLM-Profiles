library(e1071)
library(svrpath)
library(Metrics)
library(nnet)
library(randomForest)

##### this is the function for generating data in case of logistic regression with two coefficients 
gamma_invers=function(n=9,shift0=0,shift1=0,number_iter=1){
  x= as.numeric(c(log(10),log(15),log(20),log(25),log(30),log(35),log(40),log(45),log(50)))
  beta0=rep(1,n)+shift0*1.4308
  beta1=rep(2,n)+shift1*0.3803
  pi<- (beta0+beta1*x)^(-1)
  # store a value of number of iterations
  nite=number_iter
  # creating a matrix of NA values dimension of (number_iter X n)
  y=matrix(NA,nite,n)
  # assigning values to matrix which are gamma distributed with m=30
  for (j in 1:nite) {
    for (i in 1:length(pi)) {
      y[j,i]=rgamma(1,30,30/pi[i])
    }
  }
  ymean=apply(y, 1,mean)
  coef=matrix(NA,nite,2)
  for (i in 1:nite) {
    ynew=y[i,]
    # model matrix first column are 1s and second column sequence defined before
    xstar=t(rbind(rep(1,n),x))
    # glm.fit is a function with which you can estimate GLM parameters for any distribution and with any link function
    glmmodel=glm.fit(xstar,ynew,family = Gamma(link = "inverse"))
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
x_NN_in=gamma_invers(n=9,number_iter=1350)
# adding to the generated data column of 0s (label in control data?)
x_NN_in_0=cbind(x_NN_in,0)

#out of control data
# when shift - out control data
set.seed(11)
# generating data with shift0 equal to 0.1
x_NN_out_shif0a=gamma_invers(n=9,shift0=0.1,shift1=0,number_iter=150)
set.seed(12)
# generating data with shift1 equal to 0.1
x_NN_out_shif1a=gamma_invers(n=9,shift0=0,shift1=0.1,number_iter=150)
set.seed(13)
# generating data with shift0 and shift1 equal to 0.1
x_NN_out_botha=gamma_invers(n=9,shift0=0.1,shift1=0.1,number_iter=150)

set.seed(14)
# generating data with shift0 equal to 0.1
x_NN_out_shif0b=gamma_invers(n=9,shift0=0.5,shift1=0,number_iter=150)
set.seed(15)
# generating data with shift1 equal to 0.1
x_NN_out_shif1b=gamma_invers(n=9,shift0=0,shift1=0.5,number_iter=150)
set.seed(16)
# generating data with shift0 and shift1 equal to 0.1
x_NN_out_bothb=gamma_invers(n=9,shift0=0.5,shift1=0.5,number_iter=150)

set.seed(17)
# generating data with shift0 equal to 0.1
x_NN_out_shif0c=gamma_invers(n=9,shift0=1,shift1=0,number_iter=150)
set.seed(18)
# generating data with shift1 equal to 0.1
x_NN_out_shif1c=gamma_invers(n=9,shift0=0,shift1=1,number_iter=150)
set.seed(19)
# generating data with shift0 and shift1 equal to 0.1
x_NN_out_bothc=gamma_invers(n=9,shift0=1,shift1=1,number_iter=150)

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

SVR_gammainvers <- matrix(NA,1,5)
colnames(SVR_gammainvers) <- c("Type","RMSE","ARL","SDRL","UCL")

SVR_gammainvers_shift <- matrix(NA,22,7)
colnames(SVR_gammainvers_shift) <- c("Type","RMSE","ARL","SDRL","UCL","Shift 1","Shift 2")

set.seed(111)
SVR = svm(formula = Y ~ .,data = input_NN, scale =FALSE,type="eps-regression" ,kernel="radial")
gen_ucl=gamma_invers(n=9,shift0=0,shift1=0,number_iter=10000) 
ucl_SVR=sort(predict(SVR,gen_ucl),decreasing = FALSE)[0.995*10000]
set.seed(111)
SVR = svm(formula = Y ~ .,data = input_NN, scale =FALSE,type="eps-regression" ,kernel="radial")
ucl_SVR = 0.8485
arl=c()
s1=0
s2=0
for (k in 1:3000) {
  rl=0
  yhat=0
  while (yhat<ucl_SVR) {
    rl=rl+1
    x_NN_new=gamma_invers(n=9,shift0=s1,shift1=s2)
    yhat=predict(SVR,x_NN_new)
  }
  arl[k]=rl
  print(paste("iter=",k," ","rl=",rl))
}

RMSE=rmse(SVR$fitted,input_NN[,4])
SVR_gammainvers[1,]<-c('Eps radial',RMSE,mean(arl),sd(arl),ucl_SVR)
SVR_gammainvers = rbind(SVR_gammainvers,c('Eps radial',RMSE,mean(arl),sd(arl),ucl_SVR))

for (y in 1:22) {
  arl=c()
  s1=as.numeric(list_s1[y])
  s2=as.numeric(list_s2[y])
  for (k in 1:3000) {
    rl=0
    yhat=0
    while (yhat<ucl_SVR) {
      rl=rl+1
      x_NN_new=gamma_invers(n=9,shift0=s1,shift1=s2)
      yhat=predict(SVR,x_NN_new)
    }
    arl[k]=rl
    print(paste("iter=",k," ","rl=",rl))
  }
  SVR_gammainvers_shift[y,]<-c("Eps linear",SVR_gammainvers[1,2],mean(arl),sd(arl),ucl_SVR,s1,s2)
}

RF_gammainvers <- matrix(NA,50,5)
colnames(RF_gammainvers) <- c("Number of trees","MSE","ARL","SDRL","UCL")

RF_gammainvers_shift <- matrix(NA,22,7)
colnames(RF_gammainvers_shift) <- c("Number of trees","MSE","ARL","SDRL","UCL","Shift 1","Shift 2")

for (i in 1:50) {
  set.seed(111)
  RF=randomForest(Y~ .,data=input_NN,ntree=i,importance=FALSE)
  RF_gammainvers[i,1]=i
  RF_gammainvers[i,2]=mean(RF$mse)
  
}

set.seed(111)
RF=randomForest(Y~ .,data=input_NN,ntree=35,importance=FALSE)
gen_ucl=gamma_invers(n=9,shift0=0,shift1=0,number_iter=10000) 
ucl_RF=sort(predict(RF,gen_ucl),decreasing = FALSE)[0.995*10000]
set.seed(111)
RF=randomForest(Y~ .,data=input_NN,ntree=35,importance=FALSE)
ucl_RF= 0.97145
arl=c()
s1=0
s2=0
for (k in 1:3000) {
  rl=0
  yhat=0
  while (yhat<ucl_RF) {
    rl=rl+1
    x_NN_new=gamma_invers(n=9,shift0=s1,shift1=s2)
    yhat=predict(RF,x_NN_new)
  }
  arl[k]=rl
  print(paste("iter=",k," ","rl=",rl))
}

RF_gammainvers[35 ,]<-c(35,mean(RF$mse),mean(arl),sd(arl),ucl_RF)
RF_gammainvers = rbind(RF_gammainvers,c(35,mean(RF$mse),mean(arl),sd(arl),ucl_RF))

for (y in 1:22) {
  arl=c()
  s1=as.numeric(list_s1[y])
  s2=as.numeric(list_s2[y])
  for (k in 1:3000) {
    rl=0
    yhat=0
    while (yhat<ucl_RF) {
      rl=rl+1
      x_NN_new=gamma_invers(n=9,shift0=s1,shift1=s2)
      yhat=predict(RF,x_NN_new)
    }
    arl[k]=rl
    print(paste("iter=",k," ","rl=",rl))
  }
  RF_gammainvers_shift[y,] <- c(35,RF_gammainvers[35,2],mean(arl),sd(arl),ucl_RF,s1,s2)
}



NN_gammainvers <- matrix(NA,12,6)
colnames(NN_gammainvers) <- c("Nodes","Iterations","RMSE","ARL","SDRL","UCL")

NN_gammainvers_shift <- matrix(NA,22,8)
colnames(NN_gammainvers_shift) <- c("Nodes","Iterations","MSE","ARL","SDRL","UCL","Shift 1","Shift 2")

max_iter = 100
for (i in 1:12) {
  
  if (i%%5==0){
    max_iter <- max_iter + 100
  }
  set.seed(111)
  NN=nnet(Y~.,data=input_NN,size=i, linout = TRUE, maxit = max_iter)
  RMSE=rmse(NN$fitted,input_NN[,4])
  NN_gammainvers[i,1] <- i
  NN_gammainvers[i,2] <- max_iter
  NN_gammainvers[i,3] <- RMSE
  
}

set.seed(111)
NN=nnet(Y~.,data=input_NN,size=8, linout = TRUE, maxit =200)
gen_ucl=gamma_invers(n=9,shift0=0,shift1=0,number_iter=10000) 
ucl_NN=sort(predict(NN,gen_ucl),decreasing = FALSE)[0.995*10000]
set.seed(111)
NN=nnet(Y~.,data=input_NN,size=8, linout = TRUE, maxit =200)
ucl_NN=0.915
s1=0
s2=0
arl=c()
for (k in 1:3000) {
  rl=0
  yhat=0
  while (yhat<ucl_NN) {
    rl=rl+1
    x_NN_new=gamma_invers(n=9,shift0=s1,shift1=s2)
    yhat=predict(NN,x_NN_new)
  }
  arl[k]=rl
  print(paste("iter=",k," ","rl=",rl))
}

RMSE=rmse(NN$fitted.values,input_NN[,4])
NN_gammainvers[8,] <- c(8,200,RMSE,mean(arl),sd(arl),ucl_NN)
NN_gammainvers=rbind(NN_gammainvers,c(8,200,RMSE,mean(arl),sd(arl),ucl_NN))

for (y in 1:22) {
  arl=c()
  s1=as.numeric(list_s1[y])
  s2=as.numeric(list_s2[y])
  for (k in 1:3000) {
    rl=0
    yhat=0
    while (yhat<ucl_NN) {
      rl=rl+1
      x_NN_new=gamma_invers(n=9,shift0=s1,shift1=s2)
      yhat=predict(NN,x_NN_new)
    }
    arl[k]=rl
    print(paste("iter=",k," ","rl=",rl))
  }
  NN_gammainvers_shift[y,]<- c(8,200,NN_gammainvers[1,3],mean(arl),sd(arl),ucl_NN,s1,s2)
}

write.csv(SVR_gammainvers, "C:/Users/Vaio/Dokumenty/SVR_gammainvers_combo.csv", row.names=FALSE)
write.csv(SVR_gammainvers_shift, "C:/Users/Vaio/Dokumenty/SVR_gammainvers_combo_shift.csv", row.names=FALSE)

write.csv(RF_gammainvers, "C:/Users/Vaio/Dokumenty/RF_gammainvers_combo.csv", row.names=FALSE)
write.csv(RF_gammainvers_shift, "C:/Users/Vaio/Dokumenty/RF_gammainvers_combo_shift.csv", row.names=FALSE)

write.csv(NN_gammainvers, "C:/Users/Vaio/Dokumenty/NN_gammainvers_combo.csv", row.names=FALSE)
write.csv(NN_gammainvers_shift, "C:/Users/Vaio/Dokumenty/NN_gammainvers_combo_shift.csv", row.names=FALSE)

