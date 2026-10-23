library("optimx")
library("MASS")

#' Function to construct orthonormal basis that contains the unit vector mu/|mu|
#'
#' @param mu non-zero vector with length d
#'
#' @return d*d matrix that columns are the orthonormal basis, and the last column is mu/|mu|
#' 
ONB=function(mu){
  d = length(mu)
  u = matrix(0,d,d)
  norm_1 = sum(mu**2)
  if(norm_1 == 0){u[d,d] = 1}
  else{u[,d] = mu/sqrt(norm_1)}
  norm_d = sum(mu[1:2]**2)
  if(norm_d == 0){u[1,1] = 1}
  else{u[,1] = c(-mu[2], mu[1], rep(0,(d-2)))/sqrt(norm_d)}
  for(i in 3:d){
    norm_i = sum(mu[1:(i-1)]^2)
    if(norm_i ==  0){u[i-1,i-1] = 1}
    else{
      u_i = c(mu[1:i-1]*mu[i],-norm_i,rep(0,d-i))
      u[,i-1]=u_i/sqrt(sum(u_i**2))}
  }
  return(u)
}

#' Function that map parameter gamma to (lambda, theta, phi)
#'
#' @param gamma vector that d = (1+(1+8*(1+g))^0.5)/2 is an integer, where g = length(gamma)
#'
#' @return list that contains a vector of lambda, a vector of theta, and a vector of phi
#'
parameter=function(gamma){
  g=length(gamma)
  d=(1+sqrt(1+8*(1+g)))/2
  lambda=rep(NA,(d-1))
  #creating K for lambda_i=K_i*lambda_{i-1}
  K=rep(NA,(d-2)) 
  theta=rep(NA,(d-2))
  dim_phi = (d-2)*(d-3)/2
  phi=c()
  #calculate K_1 and theta_1
  K[1]=sqrt((gamma[1]^2)+(gamma[2]^2))+1 
  if(gamma[1]^2+gamma[2]^2==0){theta[1]=0}
  else if(gamma[2]>=0){theta[1] = acos(gamma[1]/sqrt(gamma[1]^2+gamma[2]^2))}
  else if(gamma[2]< 0){theta[1] = -acos(gamma[1]/sqrt(gamma[1]^2+gamma[2]^2))}
  #creating a counting function for getting the index
  counting=function(n){
    return(n*(n+1)/2)} 
  if(d>=4){
    for(i in 2:(d-2)){
      c1 = counting(i)
      c2 = counting(i+1)-1
      
      #grouping parameters
      group = gamma[c1:c2] 
      n=length(group)
      
      #K
      K[i] = sqrt(sum(group^2)) + 1
      
      #phi
      phi_group=rep(NA,(n-2))
      for(j in 1:(n-2)){
        if(sum(group[j:n]^2)==0){phi_group[j]=0}
        else{phi_group[j] = acos(group[j] / sqrt(sum(group[j:n]^2)))}  
      }
      phi=c(phi,phi_group)
      
      #theta
      if((group[(n-1)]^2+group[n]^2)==0){ theta[i]=0 }
      else if(group[n]>=0){theta[i] = acos(group[(n-1)]/sqrt(group[(n-1)]^2+group[n]^2))}
      else if(group[n] <0){theta[i] = -acos(group[(n-1)]/sqrt(group[(n-1)]^2+group[n]^2))}
    }
  }
  K_prod =prod(K^seq((d-2) ,1))
  lambda[1]=(1/K_prod)^(1/(d-1))
  for(j in 2:(d-1)){
    lambda[j]=K[(j-1)]*lambda[j-1]
  }
  return(list(lambda,theta,phi))
}


#' Function to create Rotation Matrix
#'
#' @param gamma a vector that d = (1+(1+8*(1+g))^0.5)/2 is an integer, where g = length(gamma)
#'
#' @return Rotation mapping, (d-1)*(d-1) matrix 
#'
rotation=function(gamma){
  para = parameter(gamma)
  theta = para[[2]]  
  phi= para[[3]]  
  d = length(theta)+2
  R=diag(d-1) #create a (d-1)-by-(d-1) identity matrix
  
  if(d >= 4){
    for(m in 1:(d-3)){
      Lo_R=diag(d-1)
      Lo_R[1,1] = cos(theta[d-m-1])
      Lo_R[2,2] = Lo_R[1,1]
      Lo_R[1,2] = -sin(theta[d-m-1])
      Lo_R[2,1] = -Lo_R[1,2]
      
      La_R=diag(d-1)
      for(j in 1:(d-m-2)){
        Rotation=diag(d-1) 
        Rotation[j+1,j+1]= cos(phi[1-j+(d-m-1)*(d-m-2)/2])
        Rotation[j+2,j+2]= Rotation[j+1,j+1]
        Rotation[j+1,j+2]= -sin(phi[1-j+(d-m-1)*(d-m-2)/2])
        Rotation[j+2,j+1]= -Rotation[j+1,j+2]
        La_R = La_R %*% Rotation
      }
      R=R %*% Lo_R %*% La_R 
    }
  }
  else{R=diag(d-1)}
  Lo_R=diag(d-1)
  Lo_R[1,1] = cos(theta[1])
  Lo_R[2,2] = Lo_R[1,1]
  Lo_R[1,2] = -sin(theta[1])
  Lo_R[2,1] = -Lo_R[1,2]
  R = R %*% Lo_R
  return(R)
}


#' Function to create Variance-Covariance Matrix, V, in ESAG(mu,V)
#'
#' @param mu mean direction, non-zero vector with length d
#' @lambda eigenvalues of V, vector of positive lambdas with length d-1, generated from function parameter(gamma)
#' @R Rotation mapping, (d-1) by (d-1) matrix generated from function rotation(gamma)
#' @p the power of the matrix V, that is V^p. Default p = 1, which gives V
#'
#' @return d by d matrix
#'
covariance_matrix=function(mu,lambda,R, p=1){
  d=length(mu)
  eigenvector_hat=ONB(mu)
  eigenvector=eigenvector_hat[,-d] %*% R
  P = cbind(eigenvector,eigenvector_hat[,d])
  V= P%*%diag(c(lambda^p,1))%*%t(P)
  return(V)
}


#' Function to create Variance-Covariance Matrix, V, in ESAG(mu,V)
#'
#' @param mu mean direction, non-zero vector with length d
#' @gamma reparametrized parameters in ESAG(mu,gamma)
#' @p the power of the matrix V, that is V^p. Default p = 1, which gives V
#'
#' @return d by d matrix
#'
covariance_matrix_gamma=function(mu,gamma, p=1){
  d=length(mu)
  para = parameter(gamma)
  lambda = para[[1]]
  R = rotation(gamma)
  eigenvector_hat=ONB(mu)
  eigenvector=eigenvector_hat[,-d] %*% R
  P = cbind(eigenvector,eigenvector_hat[,d])
  V= P%*%diag(c(lambda^p,1))%*%t(P)
  return(V)
}


#' Function to generate ESAG random variables
#'
#' @param n, number of ESAG random variables, numeric 
#' @param mu, mean direction, non-zero vector with length d
#' @param V, variance-covariance matrix, d*d squared matrix 
#'
#' @return n ESAG random variables, n by d matrix
#'
ESAG_generator=function(n,mu,V){
  if(n == 1){
    X=mvrnorm(1,mu,V)
    Y=X/sqrt(sum(X^2))
    return(Y)
  }
  else{
    X=mvrnorm(n,mu,V)
    X_norm=sqrt(apply(X^2,1,sum))
    Y=X/X_norm
    return(Y)
  }
}

#' Function to generate ESAG random variables
#'
#' @param n, number of ESAG random variables, numeric 
#' @param mu, mean direction, non-zero vector with length d
#' @param gamma, vector with length g = (d-2)*(d+1)/2
#'
#' @return n ESAG random variables, n by d matrix
#'
ESAG=function(n,mu,gamma){
  lambda=parameter(gamma)[[1]]
  R=rotation(gamma)
  V=covariance_matrix(mu,lambda,R)
  Y=ESAG_generator(n,mu,V)
  return(Y)
}


#' log likelihood function
#'
#' @param Bx_i, coefficients at covariates x_i, vector with length (d+2)(d-1)/2
#' @param y_i, vector with length d
#'
#' @return log-likelihood, numeric
#'
log_likelihood=function(Bx_i,y_i){  
  d=length(y_i)
  mu=Bx_i[1:d]
  gamma=Bx_i[-(1:d)]
  R=rotation(gamma)
  lambda=parameter(gamma)[[1]]
  V_inv=covariance_inv_matrix(mu,lambda,R)
  p=d-1
  yVy=t(y_i)%*%V_inv%*%y_i
  Cd = (2*pi)^(-p/2)
  a=as.vector( t(y_i)%*%mu/(yVy^(1/2)) )
  Mp=rep(0,p)
  Mp[1]=a*pnorm(a)+dnorm(a)
  Mp[2]=(1+a^2)*pnorm(a)+a*dnorm(a)
  if(p>=3){
    for(i in 3:p){
      Mp[i]=a*Mp[(i-1)]+(i-1)*Mp[(i-2)]
    }
  }
  M_p=Mp[p]
  f=log(Cd)-(d/2)*log(yVy)+(1/2)*((t(y_i)%*%mu)^2/yVy - t(mu)%*%mu)+log(M_p) 
  return(f)
}

#' Calculating the MLEs
#'
#' @param B_start, a vector of (d+2)(d-1)/2 as the starting point for the optimization algorithm. It is recommended to use 1 for parameters associated with mu and 0 for parameters associated with gamma
#' @param Y, directional data with L2 norm = 1, a n-by-d matrix where n is the sample size and d is the dimension
#' @param X, covariates, a n-by-p matrix, where p is the number of covariate. The first column of X should be a vector of 1 if intercept term is needed
#' @param HESS, TRUE or FALSE for the output of Hessian matrix
#'
#' @return a list of output from optim()
#'
MLE = function(B_start, Y, X, HESS = TRUE){
  joint_log_likelihood=function(B,y=Y,x=X){
    n = nrow(y)
    d = ncol(y)
    B_dim = (d^2+d-2)/2
    f=rep(0,n)
    for(i in 1:n){
      Bx = t(matrix(B,nrow=B_dim) %*% t(x))
      f[i]=log_likelihood(Bx_i = Bx[i,] ,y[i,])
    }
    return(sum(f))
  }
  op = optim(par=B_start, joint_log_likelihood, control=list(fnscale=-1,maxit=10000), method="BFGS",hessian = HESS)
  return(op)
}


#' Calculate residuals T1
#'
#' @param mles, a vector of (d+2)(d-1)/2 for mles. It should be in the sequences of (alpha_0,beta_0,alpha_1,beta_1,...)
#' @param Y, directional data with L2 norm = 1, a n-by-d matrix where n is the sample size and d is the dimension
#' @param X, covariates, a n-by-p matrix, where p is the number of covariate. The first column of X should be a vector of 1 if intercept term is needed
#'
#' @return a vector of T1 residuals, with length n
#'
residual_T = function(mles,Y,X){
  n = nrow(Y)
  d = ncol(Y)
  p = ncol(X)
  mle_len = length(mles)/p
  Bx = mles %*% t(X)
  T1 = rep(0,n)
  for(i in 1:n){
    mu_i = Bx[1:d,i]
    gamma_i = Bx[(d+1):mle_len,i]
    mu_i_norm = sqrt(sum(mu_i^2))
    Y_hat_i = mu_i/mu_i_norm
    r_i = (diag(1,d,d) - Y_hat_i %*% t(Y_hat_i)) %*% Y[i,]
    V_inv_i = covariance_matrix_gamma(mu=mu_i,gamma=gamma_i,p=-1)
    Q_i = t(r_i) %*% V_inv_i %*% r_i
    T1[i] = (mu_i_norm^2 + sum(parameter(gamma_i)[[1]]) + 1) * Q_i
  }
  return(T1)
}


#' Goodness of Fit algorithm for ESAG regression setting with non-parametric bootstrap
#'
#' @param mles, a vector of (d+2)(d-1)/2 for mles. It should be in the sequences of (alpha_0,beta_0,alpha_1,beta_1,...)
#' @param Y, directional data with L2 norm = 1, a n-by-d matrix where n is the sample size and d is the dimension
#' @param X, covariates, a n-by-p matrix, where p is the number of covariate. The first column of X should be a vector of 1 if intercept term is needed
#' @param num_boot, the number of rounds for bootstrapping. Set default at 300
#'
#' @return a list for statistics and pvalue
#'
Bootstrap_GoF = function(mles, Y, X, num_boot = 300){
  n = nrow(Y)
  d = ncol(Y)
  p = ncol(X)
  T1 = residual_T(mles = mles, Y = Y, X = X)
  stats = ks.test(T1,"pchisq",d-1)$p.value
  stats_boot = rep(0,num_boot)
  for(i in 1:num_boot){
    Y_b = matrix(0,n,d)
    Bx = t(mles %*% t(X))
    for(k in 1:n){
      Y_b[k,] = ESAG(n=1, mu = Bx[k,1:d], gamma = Bx[k, (d+1):ncol(Bx)])
    }
    result_boot = MLE(B_start = c(mles) , Y = Y_b, X= X)
    mles_boot = matrix(result_boot$par, ncol = ncol(X))
    T1_boot = residual_T(mles = mles_boot, Y = Y_b, X = X )
    stats_boot[i] = ks.test(T1_boot,"pchisq",d-1)$p.value
  }
  p_value = mean(stats >= stats_boot)
  return(list(test_statistics = stats, p_value = p_value))
}



#' Confidence Region evaluated at (mu_mle, gamma_mle)
#' @param mu_mles, a vector of d for the MLE of parameter mu
#' @param mu_mles, a vector of (d+1)*(d-2)/2 for the MLE of parameter gamma
#' @param sample_m, an integer, number of sampled quadratic terms that defines the region
#'
#' @return a vector of length sample_m 
#'
Confidence_Region = function(mu_mle, gamma_mle, sample_m = 10000){
  Y_b = ESAG(n = sample_m, mu = mu_mle, gamma= gamma_mle)
  V_mle_inv = covariance_matrix_gamma(mu = mu_mle, gamma= gamma_mle, p = -1)
  mu_mle_norm = sqrt(sum(mu_mle^2))
  Y_hat = mu_mle/mu_mle_norm
  q = rep(0,sample_m)
  for(l in 1:sample_m){
    q[l] = t(Y_b[l,] - Y_hat) %*% V_mle_inv %*% (Y_b[l,] - Y_hat)
  }
  return(q)
}

#' Prediction Region
#' @param Y, directional data with L2 norm = 1, a n-by-d matrix where n is the sample size and d is the dimension
#' @param X, covariates, a n-by-p matrix, where p is the number of covariate. The first column of X should be a vector of 1 if intercept term is needed
#' @param mles, a vector of (d+2)(d-1)/2 for mles. It should be in the sequences of (alpha_0,beta_0,alpha_1,beta_1,...)
#' @param index, an integer indicating the Prediction Region for the specific data point Y[index,] and X[index,]
#' @param num_boot_m, an integer, number of sampled quadratic terms that defines the region
#' @param num_boot, an integer, number of non-parametric bootstrap rounds for getting the variation of MLEs
#'
#' @return a list for 95%/90%/85%/80%/75%/70% of prediction region and a list of sampled quadratic terms used to define the prediction region.
#'
Prediction_Region = function(X, Y, mles, index, num_boot_m, num_boot){
  n = nrow(Y)
  d = ncol(Y)
  p = ncol(X)
  Bx = t(mles %*% t(X))
  Bx_index = Bx[index,]
  q_index = Confidence_Region(mu_mle = Bx_index[1:d], gamma_mle = Bx_index[(d+1):length(Bx_index)], sample_m = num_boot_m)
  q_index_boot = matrix(0,num_boot,num_boot_m)
  for(i in 1:num_boot){
    bootstrap_index = sample(seq(1,n),size = n, replace=TRUE)
    X_boot = X[bootstrap_index, ]
    Y_boot = Y[bootstrap_index, ]
    r_boot = MLE(B_start = c(mles), Y = Y_boot, X = X_boot)
    mle_boot = matrix(r_boot$par, ncol = p)
    Bx_boot = t(mle_boot %*% t(X))
    Bx_boot_index = Bx_boot[index,]
    q_index_boot[i, ] = Confidence_Region(mu_mle = Bx_boot_index[1:d], gamma_mle = Bx_boot_index[(d+1):length(Bx_boot_index)], sample_m = num_boot_m) 
  }
  Q_total = c(q_index, c(q_index_boot))
  Q_95 = quantile(Q_total,0.95)
  Q_90 = quantile(Q_total,0.90)
  Q_85 = quantile(Q_total,0.85)
  Q_80 = quantile(Q_total,0.80)
  Q_75 = quantile(Q_total,0.75)
  Q_70 = quantile(Q_total,0.70)
  mat = matrix(c(Q_95,Q_90,Q_85,Q_80,Q_75,Q_70),6,1)
  rownames(mat) = c("95% PR", "90% PR", "85% PR", "80% PR", "75% PR", "70% PR")
  return(list(Prediction_Region = mat, bootstrapped_Q = Q_total))
}

#' Calculating the MLEs under the restriction of letting all parameters associated with gamma to be zero. That is, forcing to have an isotropic ESAG model fit
#'
#' @param B_start, a vector of (d+2)(d-1)/2 as the starting point for the optimization algorithm. It is recommended to use 1 for parameters associated with mu and 0 for parameters associated with gamma
#' @param Y, directional data with L2 norm = 1, a n-by-d matrix where n is the sample size and d is the dimension
#' @param X, covariates, a n-by-p matrix, where p is the number of covariate. The first column of X should be a vector of 1 if intercept term is needed
#'
#' @return a list of output from optim()
#'
MLE_isotropy = function(B_start, Y, X){
  joint_log_likelihood=function(B,y=Y,x=X){
    n = nrow(y)
    d = ncol(y)
    p = ncol(x)
    B_dim = (d^2+d-2)/2
    B_mle = c()
    for(l in 1:p){
      B_mle = c(B_mle, B[((l-1)*d+1):((l-1)*d+4)], rep(0,(B_dim - d)))
    }
    f=rep(0,n)
    for(i in 1:n){
      Bx = t(matrix(B_mle,nrow=B_dim) %*% t(x))
      f[i]=log_likelihood(Bx_i = Bx[i,] ,y[i,])
    }
    return(sum(f))
  }
  op = optim(par=B_start, joint_log_likelihood, control=list(fnscale=-1,maxit=10000), method="BFGS",hessian = TRUE)
  return(op)
}


#' Calculating the MLEs under the restriction of letting alpha_p = 0, which is the parameter for the last column of X associated with mu
#'
#' @param B_start, a vector of (d+2)(d-1)/2 as the starting point for the optimization algorithm. It is recommended to use 1 for parameters associated with mu and 0 for parameters associated with gamma
#' @param Y, directional data with L2 norm = 1, a n-by-d matrix where n is the sample size and d is the dimension
#' @param X, covariates, a n-by-p matrix, where p is the number of covariate. The first column of X should be a vector of 1 if intercept term is needed
#'
#' @return a list of output from optim()
#'
MLE_restricted_mu = function(B_start, Y, X){
  joint_log_likelihood=function(B,y=Y,x=X){
    n = nrow(y)
    d = ncol(y)
    p = ncol(x)
    B_dim = (d^2+d-2)/2
    if(p > 1){
      B_mle = c(B[1: (B_dim*(p-1) )], rep(0,d), B[ (B_dim*(p-1)+1) : length(B)] )
    }
    if(p == 1){
      B_mle = c(rep(0,d), B[ (B_dim*(p-1)+1) : length(B)] )
    }
    f=rep(0,n)
    for(i in 1:n){
      Bx = t(matrix(B_mle,nrow=B_dim) %*% t(x))
      f[i]=log_likelihood(Bx_i = Bx[i,] ,y[i,])
    }
    return(sum(f))
  }
  op = optim(par=B_start, joint_log_likelihood, control=list(fnscale=-1,maxit=10000), method="BFGS",hessian = TRUE)
  return(op)
}


#' Calculating the MLEs under the restriction of letting beta_p = 0, which is the parameter for the last column of X associated with gamma
#'
#' @param B_start, a vector of (d+2)(d-1)/2 as the starting point for the optimization algorithm. It is recommended to use 1 for parameters associated with mu and 0 for parameters associated with gamma
#' @param Y, directional data with L2 norm = 1, a n-by-d matrix where n is the sample size and d is the dimension
#' @param X, covariates, a n-by-p matrix, where p is the number of covariate. The first column of X should be a vector of 1 if intercept term is needed
#'
#' @return a list of output from optim()
#'
MLE_restricted_gamma = function(B_start, Y, X, Hess = TRUE, optim_method = "BFGS"){
  joint_log_likelihood=function(B,y=Y,x=X){
    n = nrow(y)
    d = ncol(y)
    p = ncol(x)
    B_dim = (d^2+d-2)/2
    B_mle = c(B, rep(0, (B_dim - d) ))
    f=rep(0,n)
    for(i in 1:n){
      Bx = t(matrix(B_mle,nrow=B_dim) %*% t(x))
      f[i]=log_likelihood(Bx_i = Bx[i,] ,y[i,])
    }
    return(sum(f))
  }
  op = optim(par=B_start, joint_log_likelihood, control=list(fnscale=-1,maxit=10000), method=optim_method,hessian = Hess)
  return(op)
}

std = function(x){
  return( 1 + (x - min(x))/(max(x)-min(x)) )
}

#' Estimating the second moment evaluated at mu_i and gamma_i
#'
#' @param mu_i, a vector with dimension d, for the directional parameter in ESAG(mu,gamma)
#' @param gamma_i, a vector with dimension (d+1)(d-2)/2, for the reparametrized parameter in ESAG(mu,gamma)
#' @param mc, an integer, number of rounds of Monte Carlo 
#'
#' @return a vector with dimension d
#'
second_mome = function(mu_i, gamma_i, mc = 1000){
  Y = ESAG(n=mc, mu=mu_i, gamma = gamma_i)
  return(apply(Y**2,2,mean))
}



#' Calculate the test statistics RoC and D
#'
#' @param alpha_0, a d-by-p matrix, where d is the dimension of ESAG and p is the number of covariates. Each column of alpha_0 is the corresponding restricted MLEs associated with the column of X, i.e., the covariates. For the restricted components, set to be zero 
#' @param alpha_a, a d-by-p matrix, where d is the dimension of ESAG and p is the number of covariates. Each column of alpha_a is the corresponding full MLEs associated with the column of X, i.e., the covariates.
#' @param X, covariates, a n-by-p matrix, where p is the number of covariate. The first column of X should be a vector of 1 if intercept term is needed
#'
#' @return gives a list for the test statistics RoC and D
#'
stats_gamma = function(alpha_0,alpha_a,X){
  n = nrow(X)
  p = ncol(X)
  stat_D = rep(0,n)
  stat_RoC = rep(0,n)
  mu_0 = t(alpha_0 %*% t(X))
  mu_a = t(alpha_a %*% t(X))
  for(i in 1:n){
    mu_0_i = mu_0[i,]
    mu_a_i = mu_a[i,]
    mu_0_norm_i = sqrt(sum(mu_0_i**2))
    mu_a_norm_i = sqrt(sum(mu_a_i**2))
    stat_RoC[i] = (mu_a_norm_i/mu_0_norm_i)
    stat_D[i] = (2 - sum(mu_0_i *mu_a_i)/(mu_0_norm_i*mu_a_norm_i) ) * stat_RoC[i]
  }
  return(list(RoC = mean(stat_RoC), D = mean(stat_D)))
}

#' Calculate the test statistics M
#'
#' @param X, covariates, a n-by-p matrix, where p is the number of covariate. The first column of X should be a vector of 1 if intercept term is needed
#' @param Y, directional data with L2 norm = 1, a n-by-d matrix where n is the sample size and d is the dimension
#' @param mle_0, a (d+1)(d-2)/2-by-p matrix, where d is the dimension of ESAG and p is the number of covariates. Each column of mle_0 is the corresponding restricted MLEs (alpha,beta) associated with the column of X, i.e., the covariates. For the restricted components, set to be zero 
#' @param mc, an integer for number of rounds of Monte Carlo 
#' 
#' @return an integer for test statistics M
#'
mome_stats = function(Y,X,mle_0, mc = 1000){
  n = nrow(Y)
  d = ncol(Y)
  p = ncol(X)
  B_dim = length(mle_0)/p
  quad_diff = matrix(0,n,d)
  Bx = t(mle_0 %*% t(X))
  for(i in 1:n){
    mu_i = Bx[i,1:d]
    gamma_i = Bx[i, (d+1):B_dim]
    EY2_i = second_mome(mu_i, gamma_i, mc)
    quad_diff[i,] = Y[i,]**2 - EY2_i
  }
  statistics = apply(quad_diff,2,mean)
  return( sum(statistics**2) )
}


## mle_0 = 0 for the components that are not estimated/evaluated
## mle_0 and mle_a should be the same in terms of size

#' Hypothesis testing for isotropy
#'
#' @param mle_0, a (d+1)(d-2)/2-by-p matrix, where d is the dimension of ESAG and p is the number of covariates. Each column of mle_0 is the corresponding restricted MLEs (alpha,beta) associated with the column of X, i.e., the covariates. For the restricted components, set to be zero 
#' @param mle_a, a (d+1)(d-2)/2-by-p matrix, where d is the dimension of ESAG and p is the number of covariates. Each column of mle_a is the corresponding full MLEs (alpha,beta) associated with the column of X, i.e., the covariates. mle_0 and mle_a should have the same dimension
#' @param X, covariates, a n-by-p matrix, where p is the number of covariate. The first column of X should be a vector of 1 if intercept term is needed
#' @param Y, directional data with L2 norm = 1, a n-by-d matrix where n is the sample size and d is the dimension
#' @param num_boot, an integer for number of bootstraps rounds
#' @param mc, an integer for number of rounds of Monte Carlo 
#' 
#' @return a list for test statistics RoC, D, M and corresponding pvalues. In addition, has a vector with length num_boot for bootstrapped -2log_likelihood ratio.
#'
hypothesis_testing_isotropy = function(mle_0,mle_a, X, Y, num_boot, mc = 1000){
  n = nrow(Y)
  d = ncol(Y)
  p = ncol(X)
  mle_len = length(mle_0)/p
  stats_result = stats_gamma(alpha_0 = mle_0[1:d,], alpha_a = mle_a[1:d,], X= X)
  RoC = stats_result$RoC
  D = stats_result$D
  M = mome_stats(Y=Y,X=X,mle_0 = mle_0, mc = mc)
  B_boot = c(mle_0)
  RoC_boot = rep(0,num_boot)
  D_boot = rep(0,num_boot)
  LR_boot = rep(0,num_boot)
  M_boot = rep(0,num_boot)
  for(i in 1:num_boot){
    Bx = t(mle_0 %*% t(X))
    Y_b = matrix(0,n,d)
    for(j in 1:n){
      Y_b[j,] = ESAG(1, mu = Bx[i, 1:d], gamma = Bx[i, (d+1):mle_len])
    }
    r_0_boot = MLE_isotropy(B_start = c(mle_0[1:d,]), Y = Y_b, X = X )
    r_a_boot = MLE(B_start = c(mle_0), Y = Y_b, X = X)
    mle_0_boot = matrix(r_0_boot$par, ncol=p)
    mle_a_boot = matrix(r_a_boot$par, ncol=p)
    stats_result_boot = stats_gamma(alpha_0 = mle_0_boot, alpha_a = mle_a_boot[1:d,] ,X = X)
    RoC_boot[i] = stats_result_boot$RoC
    D_boot[i] = stats_result_boot$D
    LR_boot[i] = -2 *(r_0_boot$value - r_a_boot$value)
    make_zeros = matrix(0,nrow = (mle_len - d), ncol = p)
    mle_0_boot_rev = rbind(mle_0_boot, make_zeros)
    M_boot[i] = mome_stats(Y = Y_b,X = X,mle_0 = mle_0_boot_rev, mc = mc)
  }
  pvalue_RoC = mean(RoC < RoC_boot)
  pvalue_D = mean(D < D_boot)
  pvalue_M = mean(M < M_boot)
  return(list(RoC = RoC, D = D, M = M, pvalue_RoC = pvalue_RoC, pvalue_D = pvalue_D, pvalue_M = pvalue_M, bootstrapped_LR = LR_boot ))
}


#' Hypothesis testing for gamma dependency for X_p
#'
#' @param mle_0, a (d+1)(d-2)/2-by-p matrix, where d is the dimension of ESAG and p is the number of covariates. Each column of mle_0 is the corresponding restricted MLEs (alpha,beta) associated with the column of X, i.e., the covariates. For the restricted components, set to be zero. That is, beta_p = 0 
#' @param mle_a, a (d+1)(d-2)/2-by-p matrix, where d is the dimension of ESAG and p is the number of covariates. Each column of mle_a is the corresponding full MLEs (alpha,beta) associated with the column of X, i.e., the covariates. mle_0 and mle_a should have the same dimension
#' @param X, covariates, a n-by-p matrix, where p is the number of covariate. The first column of X should be a vector of 1 if intercept term is needed
#' @param Y, directional data with L2 norm = 1, a n-by-d matrix where n is the sample size and d is the dimension
#' @param num_boot, an integer for number of bootstraps rounds
#' @param mc, an integer for number of rounds of Monte Carlo 
#' 
#' @return a list for test statistics RoC, D, M and corresponding pvalues. In addition, has a vector with length num_boot for bootstrapped -2log_likelihood ratio.
#'
hypothesis_testing_gamma = function(mle_0,mle_a, X, Y, num_boot, mc = 1000){
  n = nrow(Y)
  d = ncol(Y)
  p = ncol(X)
  mle_len = length(mle_0)/p
  stats_result = stats_gamma(alpha_0 = mle_0[1:d,], alpha_a = mle_a[1:d,], X= X)
  RoC = stats_result$RoC
  D = stats_result$D
  M = mome_stats(Y=Y,X=X,mle_0 = mle_0, mc = mc)
  B_boot = c(mle_0)
  RoC_boot = rep(0,num_boot)
  D_boot = rep(0,num_boot)
  LR_boot = rep(0,num_boot)
  M_boot = rep(0,num_boot)
  for(i in 1:num_boot){
    Bx = t(mle_0 %*% t(X))
    Y_b = matrix(0,n,d)
    for(j in 1:n){
      Y_b[j,] = ESAG(1, mu = Bx[i, 1:d], gamma = Bx[i, (d+1):mle_len])
    }
    r_0_boot = MLE_restricted_gamma(B_start = c(mle_0)[-((length(mle_0)-d):length(mle_0))], Y = Y_b, X = X )
    r_a_boot = MLE(B_start = c(mle_0), Y = Y_b, X = X)
    num_g = mle_len - d ## make up zeros for the restricted part
    mle_0_boot = matrix(c(r_0_boot$par, rep(0,num_g)), ncol=p) ## fill zeros
    mle_a_boot = matrix(r_a_boot$par, ncol=p)
    stats_result_boot = stats_gamma(alpha_0 = mle_0_boot[1:d,], alpha_a = mle_a_boot[1:d,] ,X = X)
    RoC_boot[i] = stats_result_boot$RoC
    D_boot[i] = stats_result_boot$D
    LR_boot[i] = -2 *(r_0_boot$value - r_a_boot$value)
    M_boot[i] = mome_stats(Y = Y_b,X = X,mle_0 = mle_0_boot, mc = mc)
  }
  pvalue_RoC = mean(RoC < RoC_boot)
  pvalue_D = mean(D < D_boot)
  pvalue_M = mean(M < M_boot)
  return(list(RoC = RoC, D = D, M = M, pvalue_RoC = pvalue_RoC, pvalue_D = pvalue_D, pvalue_M = pvalue_M, bootstrapped_LR = LR_boot ))
}

#' Hypothesis testing for mu dependency for X_p
#'
#' @param mle_0, a (d+1)(d-2)/2-by-p matrix, where d is the dimension of ESAG and p is the number of covariates. Each column of mle_0 is the corresponding restricted MLEs (alpha,beta) associated with the column of X, i.e., the covariates. For the restricted components, set to be zero. That is, alpha_p = 0 
#' @param mle_a, a (d+1)(d-2)/2-by-p matrix, where d is the dimension of ESAG and p is the number of covariates. Each column of mle_a is the corresponding full MLEs (alpha,beta) associated with the column of X, i.e., the covariates. mle_0 and mle_a should have the same dimension
#' @param X, covariates, a n-by-p matrix, where p is the number of covariate. The first column of X should be a vector of 1 if intercept term is needed
#' @param Y, directional data with L2 norm = 1, a n-by-d matrix where n is the sample size and d is the dimension
#' @param num_boot, an integer for number of bootstraps rounds
#' @param mc, an integer for number of rounds of Monte Carlo 
#' 
#' @return a list for test statistics RoC, D, M and corresponding pvalues. In addition, has a vector with length num_boot for bootstrapped -2log_likelihood ratio.
#'
hypothesis_testing_mu = function(mle_0,mle_a, X, Y, num_boot, mc = 1000){
  n = nrow(Y)
  d = ncol(Y)
  p = ncol(X)
  mle_len = length(mle_0)/p
  stats_result = stats_gamma(alpha_0 = mle_0[1:d,], alpha_a = mle_a[1:d,], X = X)
  RoC = stats_result$RoC
  D = stats_result$D
  M = mome_stats(Y=Y,X=X,mle_0 = mle_0, mc = mc)
  B_boot = c(mle_0)
  RoC_boot = rep(0,num_boot)
  D_boot = rep(0,num_boot)
  LR_boot = rep(0,num_boot)
  M_boot = rep(0,num_boot)
  for(i in 1:num_boot){
    Bx = t(mle_0 %*% t(X))
    Y_b = matrix(0,n,d)
    for(j in 1:n){
      Y_b[j,] = ESAG(1, mu = Bx[i, 1:d], gamma = Bx[i, (d+1):mle_len])
    }
    r_0_boot = MLE_restricted_mu(B_start =  c(c(mle_0[,1:(p-1)]), c(mle_0[ (d+1):mle_len ,p])) , Y = Y_b, X = X )
    # r_a_boot = MLE(B_start = c(mle_0), Y = Y_b, X = X)
    # in general, seems like using '1' instead of '0' works better for the starting point regarding mu. For gamma, using 0s are good choices.
    r_a_boot = MLE(B_start =  c(c(mle_0[,1:(p-1)]), rep(1,d), c(mle_0[ (d+1):mle_len ,p])) , Y = Y_b, X = X )
    num_g = d ## make up zeros for the restricted part
    mle_0_boot = matrix( c(r_0_boot$par[1:((p-1)*mle_len)], rep(0,num_g), r_0_boot$par[((p-1)*mle_len + 1) : length(r_0_boot$par)]), ncol = p ) ## fill zeros
    mle_a_boot = matrix(r_a_boot$par, ncol=p)
    stats_result_boot = stats_gamma(alpha_0 = mle_0_boot[1:d,], alpha_a = mle_a_boot[1:d,] ,X = X)
    RoC_boot[i] = stats_result_boot$RoC
    D_boot[i] = stats_result_boot$D
    LR_boot[i] = -2 *(r_0_boot$value - r_a_boot$value)
    M_boot[i] = mome_stats(Y = Y_b,X = X,mle_0 = mle_0_boot, mc = mc)
  }
  pvalue_RoC = mean(RoC < RoC_boot)
  pvalue_D = mean(D < D_boot)
  pvalue_M = mean(M < M_boot)
  return(list(RoC = RoC, D = D, M = M, pvalue_RoC = pvalue_RoC, pvalue_D = pvalue_D, pvalue_M = pvalue_M, bootstrapped_LR = LR_boot ))
}
