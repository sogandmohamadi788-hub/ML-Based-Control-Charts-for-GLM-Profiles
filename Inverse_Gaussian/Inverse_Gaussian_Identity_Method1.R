library(e1071)
library(svrpath)
library(Metrics)
library(nnet)
library(randomForest)
library(statmod)

# Function to generate Inverse Gaussian GLM profile data with identity link

invgaussi_identity=function(n=9,shift0=0,shift1=0,number_iter=1){
  x= as.numeric(c(log(10),log(15),log(20),log(25),log(30),log(35),log(40),log(45),log(50)))
  beta0=rep(1,n)+shift0*1.4308
  beta1=rep(2,n)+shift1*0.3803
  pi<- (beta0+beta1*x)
  
  # store a value of number of iterations
  
  nite=number_iter
  
  # creating a matrix of NA values dimension of (number_iter X n)
  
  y=matrix(NA,nite,n)
  
  # assigning values to matrix which are Inverse Gaussian distributed with m=30
  
  for (j in 1:nite) {
    for (i in 1:length(pi)) {
      y[j,i] = statmod::rinvgauss(
        1,
        mean = pi[i],
        shape = 30 * pi[i]
      )
    }
  }
  ymean=apply(y, 1,mean)
  coef=matrix(NA,nite,2)
  for (i in 1:nite) {
    ynew=y[i,]
    # model matrix first column are 1s and second column sequence defined before
    xstar=t(rbind(rep(1,n),x))
    # glm.fit is a function with which you can estimate GLM parameters for any distribution and with any link function
    glmmodel=glm.fit(xstar,ynew,family = inverse.gaussian(link = "identity"),start = c(1, 2),control = glm.control(maxit = 100))
    # assigning coefficients from GLM model
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

###############################
# First training method
###############################

# In-control data

set.seed(20)

x_NN_in_1 = invgaussi_identity(
  n = 9,
  number_iter = 1200
)

# Add class label for in-control data

x_NN_in_0_1 = cbind(x_NN_in_1, 0)


# Out-of-control data
# Shift size = 0.1

# Intercept shift

set.seed(21)

x_NN_out_shif0_1 = invgaussi_identity(
  n = 9,
  shift0 = 0.1,
  shift1 = 0,
  number_iter = 400
)


# Slope shift

set.seed(22)

x_NN_out_shif1_1 = invgaussi_identity(
  n = 9,
  shift0 = 0,
  shift1 = 0.1,
  number_iter = 400
)


# Simultaneous intercept and slope shifts

set.seed(23)

x_NN_out_both_1 = invgaussi_identity(
  n = 9,
  shift0 = 0.1,
  shift1 = 0.1,
  number_iter = 400
)


# Binding all out-of-control data together

x_NN_out = rbind(
  x_NN_out_shif0_1,
  x_NN_out_shif1_1,
  x_NN_out_both_1
)



# Combining in-control and out-of-control data

x_NN_out_1=cbind(x_NN_out,1)
input_NN=rbind(x_NN_in_0,x_NN_out_1)

# Change column names

colnames(input_NN)=c("ymean","beta0","beta1","Y")
list_s1 <- c(0.1,0.3,0.5,0.8,1,2,0,0,0,0,0,0,0.1,0.3,0.4,0.3,0.9,0.5,0.3,0.8,1,0.8)
list_s2 <- c(0,0,0,0,0,0,0.1,0.3,0.5,0.8,1,2,0.1,0.1,0.1,0.4,0.1,0.6,0.9,0.5,0.6,1)

SVR_invgaussi_identity <- matrix(NA,1,5)
colnames(SVR_invgaussi_identity) <- c("Type","RMSE","ARL","SDRL","UCL")

SVR_invgaussi_identity_shift <- matrix(NA,22,7)
colnames(SVR_invgaussi_identity_shift) <- c("Type","RMSE","ARL","SDRL","UCL","Shift 1","Shift 2")

# Set seed for reproducibility

set.seed(111)

# Training the Support Vector Regression (SVR) model using the generated Inverse Gaussian GLM profile data

SVR = svm(formula = Y ~ .,data = input_NN, scale =FALSE,type="eps-regression" ,kernel="radial")

# Define the upper control limit (UCL) for the SVR-based control chart
gen_ucl=invgaussi_identity(n=9,shift0=0,shift1=0,number_iter=10000) 
ucl_SVR=sort(predict(SVR,gen_ucl),decreasing = FALSE)[0.995*10000]

# Initialize a vector to store run lengths

arl=c()

# Set the shift values for the in-control process

s1=0
s2=0

# Evaluate the in-control performance of the SVR-based control chart

for (k in 1:300) {
  rl=0
  yhat=0
  
  # Generate profiles until the SVR prediction exceeds the UCL
  
  while (yhat<ucl_SVR) {
    rl=rl+1
    
    # Generate an Inverse Gaussian GLM profile with identity link
    
    x_NN_new=invgaussi_identity(n=9,shift0=s1,shift1=s2)
    
    # Obtain the SVR prediction for the generated profile
    
    yhat=predict(SVR,x_NN_new)
  }
  
  # Store the run length for the current simulation
  
  arl[k]=rl
  print(paste("iter=",k," ","rl=",rl))
}

# Calculate the RMSE of the trained SVR model

RMSE=rmse(SVR$fitted,input_NN[,4])

# Store the in-control performance measures of the SVR-based control chart

SVR_invgaussi_identity[1,]<-c('Eps radial',RMSE,mean(arl),sd(arl),ucl_SVR)
SVR_invgaussi_identity = rbind(SVR_invgaussi_identity,c('Eps radial',RMSE,mean(arl),sd(arl),ucl_SVR))

# Evaluate the SVR-based control chart under different shift conditions

for (y in 1:22) {
  arl=c()
  
  # Assign the shift values for the current simulation
  
  s1=as.numeric(list_s1[y])
  s2=as.numeric(list_s2[y])
  
  # Perform 300 simulation runs for the specified shift condition
  
  for (k in 1:300) {
    rl=0
    yhat=0
    
    # Generate profiles until the SVR prediction exceeds the UCL
    
    while (yhat<ucl_SVR) {
      rl=rl+1
      
      # Generate an Inverse Gaussian GLM profile with the specified shifts
      
      x_NN_new=invgaussi_identity(n=9,shift0=s1,shift1=s2)
      
      # Obtain the SVR prediction for the generated profile
      
      yhat=predict(SVR,x_NN_new)
    }
    
    # Store the run length for the current simulation
    
    arl[k]=rl
    print(paste("iter=",k," ","rl=",rl))
  }
  
  # Store the performance measures for the specified shift condition
  
  SVR_invgaussi_identity_shift[y,]<-c("Eps linear",SVR_invgaussi_identity[1,2],mean(arl),sd(arl),ucl_SVR,s1,s2)
}

# Initialize a matrix to store the performance results of the Random Forest model

RF_invgaussi_identity <- matrix(NA,50,5)
colnames(RF_invgaussi_identity) <- c("Number of trees","MSE","ARL","SDRL","UCL")

# Initialize a matrix to store the performance results under different shift conditions

RF_invgaussi_identity_shift <- matrix(NA,22,7)
colnames(RF_invgaussi_identity_shift) <- c("Number of trees","MSE","ARL","SDRL","UCL","Shift 1","Shift 2")

# Evaluate the effect of the number of trees on the Random Forest model

for (i in 1:50) {
  set.seed(111)
  
  # Train the Random Forest model with i trees using the Inverse Gaussian GLM profile data
  
  RF=randomForest(Y~ .,data=input_NN,ntree=i,importance=FALSE)
  
  # Store the number of trees
  
  RF_invgaussi_identity[i,1]=i
  
  # Store the mean squared error (MSE) of the Random Forest model
  
  RF_invgaussi_identity[i,2]=mean(RF$mse)
  
}

# Set seed for reproducibility

set.seed(111)

# Train the final Random Forest model with 45 trees

RF=randomForest(Y~ .,data=input_NN,ntree=45,importance=FALSE)

# Define the upper control limit (UCL) for the Random Forest-based control chart

gen_ucl=invgaussi_identity(n=9,shift0=0,shift1=0,number_iter=10000) 
ucl_RF=sort(predict(RF,gen_ucl),decreasing = FALSE)[0.995*10000]

# Initialize a vector to store run lengths

arl=c()

# Set the shift values for the in-control process

s1=0
s2=0

# Evaluate the in-control performance of the Random Forest-based control chart

for (k in 1:300) {
  rl=0
  yhat=0
  
  # Generate profiles until the Random Forest prediction exceeds the UCL
  
  while (yhat<ucl_RF) {
    rl=rl+1
    
    # Generate an Inverse Gaussian GLM profile with identity link
    
    x_NN_new=invgaussi_identity(n=9,shift0=s1,shift1=s2)
    
    # Obtain the Random Forest prediction for the generated profile
    
    yhat=predict(RF,x_NN_new)
  }
  
  # Store the run length for the current simulation
  
  arl[k]=rl
  print(paste("iter=",k," ","rl=",rl))
}

# Store the in-control performance measures of the Random Forest-based control chart

RF_invgaussi_identity[45 ,]<-c(45,mean(RF$mse),mean(arl),sd(arl),ucl_RF)
RF_invgaussi_identity = rbind(RF_invgaussi_identity,c(45,mean(RF$mse),mean(arl),sd(arl),ucl_RF))

# Evaluate the Random Forest-based control chart under different shift conditions

for (y in 1:22) {
  arl=c()
  
  # Assign the shift values for the current simulation
  
  s1=as.numeric(list_s1[y])
  s2=as.numeric(list_s2[y])
  
  # Perform 300 simulation runs for the specified shift condition
  
  for (k in 1:300) {
    rl=0
    yhat=0
    
    # Generate profiles until the Random Forest prediction exceeds the UCL
    
    while (yhat<ucl_RF) {
      rl=rl+1
      
      # Generate an Inverse Gaussian GLM profile with the specified shifts
      
      x_NN_new=invgaussi_identity(n=9,shift0=s1,shift1=s2)
      
      # Obtain the Random Forest prediction for the generated profile
      
      yhat=predict(RF,x_NN_new)
    }
    
    # Store the run length for the current simulation
    
    arl[k]=rl
    print(paste("iter=",k," ","rl=",rl))
  }
  
  # Store the performance measures for the specified shift condition
  
  RF_invgaussi_identity_shift[y,] <- c(45,RF_invgaussi_identity[45,2],mean(arl),sd(arl),ucl_RF,s1,s2)
}

# Initialize a matrix to store the performance results of the Neural Network model

NN_invgaussi_identity <- matrix(NA,12,6)
colnames(NN_invgaussi_identity) <- c("Nodes","Iterations","RMSE","ARL","SDRL","UCL")

# Initialize a matrix to store the performance results under different shift conditions

NN_invgaussi_identity_shift <- matrix(NA,22,8)
colnames(NN_invgaussi_identity_shift) <- c("Nodes","Iterations","MSE","ARL","SDRL","UCL","Shift 1","Shift 2")

# Set the initial maximum number of iterations for training the Neural Network

max_iter = 100

# Evaluate the effect of the number of hidden nodes on the Neural Network model

for (i in 1:12) {
  
  if (i%%5==0){
    max_iter <- max_iter + 100
  }
  set.seed(111)
  
  # Train the Neural Network model using the Inverse Gaussian GLM profile data
  
  NN=nnet(Y~.,data=input_NN,size=i, linout = TRUE, maxit = max_iter)
  
  # Calculate the root mean squared error (RMSE) of the Neural Network model
  
  RMSE=rmse(NN$fitted,input_NN[,4])
  
  # Store the number of hidden nodes
  
  NN_invgaussi_identity[i,1] <- i
  
  # Store the number of training iterations
  
  NN_invgaussi_identity[i,2] <- max_iter
  
  # Store the RMSE
  
  NN_invgaussi_identity[i,3] <- RMSE
  
}

# Set seed for reproducibility

set.seed(111)

# Train the final Neural Network model with 8 hidden nodes and 200 iterations

NN=nnet(Y~.,data=input_NN,size=8, linout = TRUE, maxit =200)

# Define the upper control limit (UCL) for the Neural Network-based control chart

gen_ucl=gamma_identity(n=9,shift0=0,shift1=0,number_iter=10000) 
ucl_NN=sort(predict(NN,gen_ucl),decreasing = FALSE)[0.995*10000]

# Set the shift values for the in-control process

s1=0
s2=0

# Initialize a vector to store run lengths

arl=c()

# Evaluate the in-control performance of the Neural Network-based control chart

for (k in 1:300) {
  rl=0
  yhat=0
  
  # Generate profiles until the Neural Network prediction exceeds the UCL
  
  while (yhat<ucl_NN) {
    rl=rl+1
    
    # Generate an Inverse Gaussian GLM profile with identity link
    
    x_NN_new=invgaussi_identity(n=9,shift0=s1,shift1=s2)
    
    # Obtain the Neural Network prediction for the generated profile
    
    yhat=predict(NN,x_NN_new)
  }
  
  # Store the run length for the current simulation
  
  arl[k]=rl
  print(paste("iter=",k," ","rl=",rl))
}

# Calculate the RMSE of the final Neural Network model

RMSE=rmse(NN$fitted.values,input_NN[,4])

# Store the in-control performance measures of the Neural Network-based control chart

NN_invgaussi_identity[8,] <- c(8,200,RMSE,mean(arl),sd(arl),ucl_NN)
NN_invgaussi_identity=rbind(NN_invgaussi_identity,c(8,200,RMSE,mean(arl),sd(arl),ucl_NN))

# Evaluate the Neural Network-based control chart under different shift conditions

for (y in 1:22) {
  arl=c()
  
  # Assign the shift values for the current simulation
  
  s1=as.numeric(list_s1[y])
  s2=as.numeric(list_s2[y])
  
  # Perform 300 simulation runs for the specified shift condition
  
  for (k in 1:300) {
    rl=0
    yhat=0
    
    # Generate profiles until the Neural Network prediction exceeds the UCL
    
    while (yhat<ucl_NN) {
      rl=rl+1
      
      # Generate an Inverse Gaussian GLM profile with the specified shifts
      
      x_NN_new=invgaussi_identity(n=9,shift0=s1,shift1=s2)
      
      # Obtain the Neural Network prediction for the generated profile
      
      yhat=predict(NN,x_NN_new)
    }
    
    # Store the run length for the current simulation
    
    arl[k]=rl
    print(paste("iter=",k," ","rl=",rl))
  }
  
  # Store the performance measures for the specified shift condition
  
  NN_invgaussi_identity_shift[y,]<- c(8,200,NN_invgaussi_identity[1,3],mean(arl),sd(arl),ucl_NN,s1,s2)
}

# Display the results of the SVR-based control chart

SVR_invgaussi_identity 

# Display the SVR results under different shift conditions

SVR_invgaussi_identity_shift

# Display the results of the Random Forest-based control chart

RF_invgaussi_identity 

# Display the Random Forest results under different shift conditions

RF_invgaussi_identity_shift

# Display the results of the Neural Network-based control chart

NN_invgaussi_identity 

# Display the Neural Network results under different shift conditions

NN_invgaussi_identity_shift