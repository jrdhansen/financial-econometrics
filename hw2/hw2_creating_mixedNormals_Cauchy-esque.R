


drawMixedNormals <- function(ndraws){
  
  draws <- rep(0, ndraws)
  
  for(i in 1:ndraws){
    pick <- sample(c(1,2,3), ndraws, replace=TRUE)
    if(pick == 1){
      draws[i] <- rnorm(ndraws, mean=0, sd=1)
    }
    else if(pick ==2){
      draws[i] <- rnorm(ndraws, mean=0, sd=5)
    }
    else if(pick == 3){
      draws[i] <- rnorm(ndraws, mean=0, sd=10)
    }
    else{
      draws[i] <- rnorm(ndraws, mean=0, sd=1)
    }
  }
  return(draws)
}


M <- 1000
z <- rnorm(M, mean=0, sd=1)
hist(z, breaks=50)

# This increased the variation of the data (thanks to the the new random normals
# that have sd=5 and sd=10)
# THIS LOOKS LIKE A CAUCHY DISTRIBUTION (more like what real data would be,
# NOT THE NICE RANDOM NORMAL, BUT A "mixture of random normals")
x <- drawMixedNormals(M)
hist(x, breaks=50)

























