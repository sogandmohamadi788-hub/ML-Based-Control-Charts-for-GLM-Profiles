##### this is the function for generating data in case of logistic regression with two coefficients poisson_log
poiss_log=function(n=10,shift0=0,shift1=0,number_iter=1){
  # creating a sequence of number of length n with last number 1 and step 0.1
  x=as.numeric(seq(0.1,1,length.out=n))
  beta0=rep(3,n)+shift0*0.375
  beta1=rep(2,n)+shift1*0.177
  mu=exp(beta0+beta1*x)
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
    xstar=x
    # glm.fit is a function with which you can estimate GLM parameters for any distribution and with any link function
    glmmodel=glm.fit(xstar,ynew,family = poisson(link = "log"))
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
x_NN_in=poiss_log(n=10,number_iter=1200)
# adding to the generated data column of 0s (label in control data?)
x_NN_in_0=cbind(x_NN_in,0)

#out of control data
# when shift - out control data
set.seed(11)
# generating data with shift0 equal to 0.1
x_NN_out_shif0=poiss_log(n=10,shift0=0.1,shift1=0,number_iter=400)
set.seed(12)
# generating data with shift1 equal to 0.1
x_NN_out_shif1=poiss_log(n=10,shift0=0,shift1=0.1,number_iter=400)
set.seed(13)
# generating data with shift0 and shift1 equal to 0.1
x_NN_out_both=poiss_log(n=10,shift0=0.1,shift1=0.1,number_iter=400)

# binding all data together
x_NN_out=rbind(x_NN_out_shif0,x_NN_out_shif1,x_NN_out_both)
# adding to the generated data column of 1s (label out control data?)
x_NN_out_1=cbind(x_NN_out,1)

#in control and out of control combined
input_NN=rbind(x_NN_in_0,x_NN_out_1)

# change column names 
colnames(input_NN)=c("ymean","beta0","beta1","Y")
