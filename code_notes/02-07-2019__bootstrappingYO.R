
#==== THURSDAY, 02/07/2019 =====================================================





#=== Illustrating the law of large numbers
n <- 10000
values <- rep(0, n)
theta = 0.3

for(i in 1:n){
  x = rbinom(i, 1, prob=theta)
  values[i] = mean(x)
}

plot(values, lwd = 2, type="l", col="blue")



#=== The Central Limit Theorem: distn of the estimator will approximate a
# normal distn. (For example, x_bar ~ Normal(mu, sigma^2/n))

# Make draws from a binomial distribution
m = 10000
n = 1000
theta = 0.25
x = rbinom(m,n,prob = theta)
p = x/n
hist(p)
# The histogram should be centered around 0.25 and look approx normal.
# We're creating the pt est for p






#=== The bootstrap: resampling from the data with replacment.

# The Universe, Mother nature, the tao, etc. gives a single statistician a
# sample of data.
n = 500
data = rexp(n = 500)  # data from an exponential distn

# Now the statistican uses the Bootstrap algorithm to approxmiate the sampling
# distribution
artDat <- list()
m = 5000
values = rep(0,m)

for(i in 1:m){
  
  artDat[[i]] = sample(data,n,replace = TRUE)
  values[i] = mean(artDat[[i]])
}

hist(values)
# Nice, this looks very close to looking normal.

x = rnorm(m, mean = 1, sd = 1/sqrt(500))
plot(density(x), lwd = 2, type = "l", col = "blue")




#=== TESTING AGAIN WITH THE GAMMA DISTN
n = 5000
data = rgamma(n, shape = 0.01, rate = 1)


for(i in 1:m){
  
  artDat[[i]] = sample(data,n,replace = TRUE)
  values[i] = mean(artDat[[i]])
}

hist(values)  # As n -> infinity the histogram becomes more normal






#=== NOW WITH TIME SERIES MODELING: BOOTSTRAPPING THE EPSILON TERMS
# x_t = (phi)*(x_{t-1}) + u_t   (the u vector is our epsilons)

phi = 0.7
n = 500
x = rep(0, n)

x[1] <- rnorm(1)
u <- rnorm(n)    # these are the epsilons

# This does the bootstrapping
for(t in 2:n){
  x[t] = phi * x[t-1] + u[t]
}

plot(x, col="purple", type = "l")


# Now hand the data to a statistician
Y <- x[2:n]
X <- x[1:(n-1)]

reg <- lm(Y ~ X)
reg
# In the regression, our pt estimate of phi is the slope value 
# (I got X 0.65774)   ---- this only gives one estimate, THAT'S WHY WE USE
# BOOTSTRAPPING: MANY ESTIMATES. HELL YA




#=== MONTE CARLO (then bootstrap) for estimating phi parameter of the series

sim_AR <- function(phi_hat, y0, size){
  y <- rep(0,size)
  y[1] <- phi_hat * y0 + rnorm(size)
  u = rnorm(size)
  
  for(t in 2:size){
    y[t] <- phi_hat * y[t-1] + u[t]
  }
  
  return(y)
}

n = 50   # this is the sample size
m = 1000   # this is the number of times we're estimating phi, generating the sampling distn histgrm
phi_estmt = rep(0, m)
phi_hat = coef(reg)[[2]]   # using the same regression as above
y0 = x[1]

for(i in 1:m){
  y_sim <- sim_AR(phi_hat, y0, n)
  Y <- y_sim[2:n]
  X <- y_sim[1:(n-1)]
  phi_estmt[i] <- coef(lm(Y ~ X))[[2]]
}

hist(phi_estmt)






#==== BOOTSTRAPPING ===========================================================
# much the same as monte carlo, but we don't assume a parametric distn for the epsilon germs
# Bootstrapping is better than monte carlo because MC assumes the data-generating
# mechanism, while bootstrapping lets the "data speak for themselves".

simAR1Bootstrap <- function(phi_hat, resids, y0, size){
  y <- rep(0,size)
  u <- sample(resids, size, replace = TRUE)
  y[1] <- phi_hat * y0 + u[1]

  for(t in 2:size){
    y[t] <- phi_hat * y[t-1] + u[t]
  }
  
  return(y)
}

n = 500   # this is the sample size
m = 1000   # this is the number of times we're estimating phi, generating the sampling distn histgrm
phi_estmt = rep(0, m)
phi_hat = coef(reg)[[2]]   # using the same regression as above
resids <- reg$residuals
y0 = x[1]

for(i in 1:m){
  y_sim <- simAR1Bootstrap(phi_hat, resids, y0, n)
  Y <- y_sim[2:n]
  X <- y_sim[1:(n-1)]
  phi_estmt[i] <- coef(lm(Y ~ X))[[2]]
}

hist(phi_estmt)










