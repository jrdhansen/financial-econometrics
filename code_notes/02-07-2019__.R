
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
# x_t = (phi)*(x_{t-1}) + eps_t



























