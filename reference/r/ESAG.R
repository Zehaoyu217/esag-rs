
library("optimx")
library("MASS")

#' Function to construct orthonormal basis that contains the unit vector mu/|mu|
#'
#' @param mu non-zero vector with length d
#'
#' @return d*d matrix that columns are the orthonormal basis, and the last column is mu/|mu|
#'
ONB=function(mu){
  d=length(mu)
  u=matrix(NA,d,d)
  u[,1]=mu 
  u[,2]=c(-mu[2],mu[1],rep(0,(d-2)))
  
  for(i in 3:d){
    u_i = mu*c(rep(1,i-1),rep(0,d-i+1))*mu[i]
    u_i[i]=-sum(mu[1:(i-1)]^2)
    u[,i]=u_i
  }
  
  V=matrix(NA,d,d)
  std = function(x){return(x/sqrt(sum(x^2)))}
  V = apply(u,2,std)
  K = cbind(V[,2:d],V[,1])
  
  ord = which(is.na(colSums(K))) #to find out columns that are NA (indicating mu_1,...,mu_k = 0 for k>=2) 
  for(i in ord){ # replace these columns with e_j
    K[,i] = rep(0,d) 
    K[i,i] = 1
  }
  return(K)
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
  if(as.integer(d) != d){return('incorrect dimensions of the input vector')}
  lambda=rep(NA,(d-1))
  
  K=rep(NA,(d-2)) #creating K for lambda_i=K_i*lambda_{i-1}
  theta=rep(NA,(d-2))
  dim_phi = (d-2)*(d-3)/2
  phi=c()
  
  K[1]=sqrt((gamma[1]^2)+(gamma[2]^2))+1 #calclate K_1 
  theta[1]=atan2(gamma[2],gamma[1]) #calculate theta1
  
  counting=function(n){ 
    return(n*(n+1)/2)} #creating a counting function
  
  if(d>=4){
    for(i in 2:(d-2)){
      c1 = counting(i)
      c2 = counting(i+1)-1
      group = gamma[c1:c2] #grouping parameters
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
      theta[i] = atan2(group[(n)],group[(n-1)])
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
#'
#' @return d by d matrix
#'
covariance_matrix=function(mu,lambda,R){
  d=length(mu)
  eigenvector_hat=ONB(mu)
  eigenvector=eigenvector_hat[,-d] %*% R
  P = cbind(eigenvector,eigenvector_hat[,d])
  V= P%*%diag(c(lambda,1))%*%t(P)
  return(V)
}


#' Function to create Variance-Covariance Matrix, V^0.5, in ESAG(mu,V)
#'
#' @param mu mean direction, non-zero vector with length d
#' @lambda eigenvalues of V, vector of positive lambdas with length d-1, generated from function parameter(gamma)
#' @R Rotation mapping, (d-1) by (d-1) matrix generated from function rotation(gamma)
#'
#' @return d by d matrix
#'
covariance_matrix_2=function(mu,lambda,R){
  d=length(mu)
  eigenvector_hat=ONB(mu)
  eigenvector=eigenvector_hat[,-d] %*% R
  P = cbind(eigenvector,eigenvector_hat[,d])
  V= P%*%diag(c(lambda^(0.5),1))%*%t(P)
  return(V)
}

#' Function to create inverse of Variance-Covariance Matrix, V^-1, in ESAG(mu,V)
#'
#' @param mu, mean direction, non-zero vector with length d
#' @param lambda, eigenvalues of V, vector of positive lambdas with length d-1, generated from function parameter(gamma)
#' @param R, Rotation mapping, (d-1)*(d-1) matrix generated from function rotation(gamma)
#'
#' @return d by d matrix
#'
covariance_inv_matrix=function(mu,lambda,R){
  d=length(mu)
  eigenvector_hat=ONB(mu)
  eigenvector=eigenvector_hat[,-d] %*% R
  P = cbind(eigenvector,eigenvector_hat[,d])
  V= P%*%diag(c(1/lambda,1))%*%t(P)
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

#' Function to generate T1 version of residuals given mu, gamma, and the data Y
#'
#' @param mu, mean direction, non-zero vector with length d
#' @param gamma, vector with length g = (d-2)*(d+1)/2
#' @param Y, n by d matrix, n observations of d-dimension ESAG 
#'
#' @return T1 residuals, vector with length n
#'
Model_diagnostic_mu = function(mu,gamma,Y){
  n = nrow(Y)
  d = length(mu)
  chi_sq = rep(0,n)
  mu_norm = sqrt(sum(mu^2))
  lambda = parameter(gamma)[[1]]
  R = rotation(gamma)
  V_inv = covariance_inv_matrix(mu,lambda,R)
  for(i in 1:n){
    Z =  (diag(1,d,d)-(mu/mu_norm)%*%t(mu/mu_norm))%*%Y[i,]
    chi_sq[i]  = (sum(lambda)+1+mu_norm^2)*t(Z)%*% V_inv %*% Z
  }
  return(chi_sq)
}

#' Function to generate residuals, Q, given mu, gamma, and the data Y
#'
#' @param mu, mean direction, non-zero vector with length d
#' @param gamma, vector with length g = (d-2)*(d+1)/2
#' @param Y, n by d matrix, n observations of d-dimension ESAG 
#'
#' @return Q residuals, vector with length n
#'
Model_diagnostic_Q = function(mu,gamma,Y){
  n = nrow(Y)
  d = length(mu)
  chi_sq = rep(0,n)
  for(i in 1:n){
    mu_norm = sqrt(sum(mu^2))
    lambda = parameter(gamma)[[1]]
    R = rotation(gamma)
    V_inv = covariance_inv_matrix(mu,lambda,R)
    Z =  (diag(1,d,d)-(mu/mu_norm)%*%t(mu/mu_norm))%*%Y[i,]
    chi_sq[i]  = t(Z)%*% V_inv %*% Z
  }
  return(chi_sq)
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

#' Goodness of Fit test with bootstrap, non-regression version
#'
#' @param MLE, maximum likelihood estimations given the data Y, vector with length (d+2)(d-1)/2
#' @param Y, dataset with n observations, n by d matrix
#' @param B, number of bootstrap sampling, numeric
#'
#' @return p-values, numeric
#'
Bootstrap_gof = function(MLE,Y,B){  #B is # of bootstrap MC replicas
  k=nrow(Y)
  B_dim=length(MLE)
  d=sqrt(2*B_dim+(9/4))-(1/2)
  pvalue_Q = rep(0,B)
  pvalue_W = rep(0,B)
  library(optimx)
  
  # log likelihood function
  log_likelihood_boot=function(B,y){  
    B_dim=length(B)
    d=length(y)
    y = as.matrix(y,length(y),1)
    Bx = B
    mu=Bx[1:d]
    gamma=Bx[(d+1):B_dim]
    R=rotation(gamma)
    lambda=parameter(gamma)[[1]]
    V_inv=covariance_inv_matrix(mu,lambda,R)
    p=d-1
    yVy=t(y)%*%V_inv%*%y
    a=as.vector( t(y)%*%mu/(yVy^(1/2)) )
    Cd = (2*pi)^(-p/2)
    Mp=rep(0,p)
    Mp[1]=a*pnorm(a)+dnorm(a)
    Mp[2]=(1+a^2)*pnorm(a)+a*dnorm(a)
    if(p>=3){
      for(i in 3:p){
        Mp[i]=a*Mp[(i-1)]+(i-1)*Mp[(i-2)]
      } 
    }
    M_p=Mp[p]
    f=log(Cd)-(d/2)*log(yVy)+(1/2)*((t(y)%*%mu)^2/yVy - t(mu)%*%mu)+log(M_p)
    return(f)
  }
  
  #joint log likeliehood function
  joint_log_likelihood_boot=function(B,y=Y_boot){
    n=nrow(y)
    f=0
    for(i in 1:n){
      f=f+log_likelihood_boot(B,y[i,])
    }
    return(f)
  }
  
  B_boot = MLE
  for(l in 1:B){
    Y_boot=ESAG(k,mu=B_boot[1:4],gamma=B_boot[5:9])
    op = optim(par=B_boot, joint_log_likelihood_boot, control=list(fnscale=-1,maxit=10000), method="BFGS")
    MLE_boot = op$par
    residual_Q_boot = Model_diagnostic_Q(mu=MLE_boot[1:4],gamma=MLE_boot[5:9],Y=Y_boot)
    
    ## another bootstrap of bootstrap
    Y_bb = ESAG(k,mu=MLE_boot[1:4],gamma=MLE_boot[5:9])
    residual_Q_bb = Model_diagnostic_Q(mu=MLE_boot[1:4],gamma=MLE_boot[5:9],Y=Y_bb)
    
    ## get the distribution of the pvalue
    pvalue_Q[l] = ks.test(residual_Q_boot, residual_Q_bb)$p.value
  }
  
  residual_Q = Model_diagnostic_Q(mu=MLE[1:4],gamma=MLE[5:9],Y=Y)
  Y_copy = ESAG(k,mu=MLE[1:4],gamma=MLE[5:9])
  residual_Q_copy = Model_diagnostic_Q(mu=MLE[1:4],gamma=MLE[5:9],Y=Y_copy)
  p_Q = ks.test(residual_Q,residual_Q_copy)$p.value
  real_p_Q = mean(pvalue_Q < p_Q)
  return(real_p_Q )
}

#' Function to find MLEs
#'
#' @param B_start, starting point of the coefficient, ((d+2)(d-1)/2) by p matrix
#' @param Y, dataset with n observations, n*d matrix
#' @param X, covariates, n*p matrix, p is number of covariates
#' @param H, TRUE/FALSE for Hessian matrix in the output
#'
#' @return list that contains a vector of MLEs, maxmimal log-likelihood value, etc from function optim() with 'BFGS'. It can output Hessian matrix as well
#'
MLE = function(B_start=c(), Y, X=c(), H = FALSE){
  if(is.null(X)){
    X = as.matrix(rep(1,nrow(Y)))
  } else{ X = cbind(rep(1,nrow(Y)),X) }
  
  if(is.null(B_start)){
    B_dim = ((ncol(Y)^2+ncol(Y)-2)/2)
    B_start = matrix(0,B_dim,ncol(X))
    B_start[1:ncol(Y) ,] = 1
  }
  if(ncol(B_start) != (ncol(X)) ){
    return("number of covariates in B and X do not match")
  } else if( nrow(B_start) !=  ((ncol(Y)^2+ncol(Y)-2)/2) )
    {
    return("dimension of B do not match with dimension of Y")
  } else if(nrow(Y) != nrow(X)){
    return("number of observations of X and Y do not match")
  }
  else{
    B = c(B_start)
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
    start = B
    op = optim(par=start, joint_log_likelihood, control=list(fnscale=-1,maxit=10000), method="BFGS", hessian = H)
    return(op)
  }
}