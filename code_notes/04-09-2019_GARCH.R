
'''
Brough Lecture Notes: GARCH Models in R

Finance 5330: Financial Econometrics 
Tyler J. Brough 
Last Updated: April 1, 2019 
'''


options(warn=-1)
require(bayesGARCH)
require(rugarch)
require(zoo)

ibm_path <- "C:/__JARED/_USU_Sp2019/_Brough_FinEcon/__Fin5330-master__04-02-2019/Notebooks/data/IBM-1999-2003.csv"
ibm <- read.csv(ibm_path, header=T)

head(ibm)
tail(ibm)

# zoo makes a meta-package for time series. Default package to use for time series
index <- as.Date(as.character(ibm$date), "%Y%m%d")
ret <- zoo(ibm$RET, as.Date(as.character(ibm$date), "%Y%m%d"))

# We see volatility clustering and mean-reverting of IBM returns in this plot.
plot(ret)
# Default GARCH specification is GARCH(1,1)
# the armaOrder = c(0,0) is shutting off the linear model for THE MEAN, where
# GARCH is concerned with the variance (volatility) we're just specifying 
# the model for the errors (variance) as GARCH(1,1) --> this is a DEFAULT
spec <- ugarchspec(mean.model = list(armaOrder = c(0,0), include.mean=FALSE))

# help(ugarchfit)

data <- zoo
fit <- ugarchfit(data = ret, spec = spec, solver="hybrid")

fit

plot(sigma(fit))

vals <- coef(fit)
what <- coef(fit)[1]
ahat <- coef(fit)[2]
bhat <- coef(fit)[3]

what
ahat
bhat

ret[76]

ind <- (abs(ret) >= 0.12)
ret[ind]

ii <- 1:length(ret)
c(ii[ind])

spec2 <- ugarchspec(variance.model=list(model = "gjrGARCH",
                                        garchOrder=c(1,1,1)),
                    mean.model=list(armaOrder=c(0,0), include.mean=F))

fit2 <- ugarchfit(data=ret, spec=spec2, solver="hybrid")
fit2





dem2gbp_path <- "C:/__JARED/_USU_Sp2019/_Brough_FinEcon/__Fin5330-master__04-02-2019/Notebooks/data/dem2gbp.csv"
dem2gbp <- read.csv(dem2gbp_path, header=T)
data(dem2gbp)
y <- dem2gbp$DEM2GBP[1:750]
spec1 <- ugarchspec(mean.model=list(armaOrder=c(0,0), include.mean=F), distribution="norm")
fit1 <- ugarchfit(data=y, spec=spec1)
coef(fit1)


spec2 <- ugarchspec(mean.model=list(armaOrder=c(0,0), include.mean=F), distribution="std")
fit2 <- ugarchfit(data=y, spec=spec2)
coef(fit2)
# The shape parameter is the degrees of freedom in the student-t distribution
# (the value of 5.33 we get implies thick tails).



#---------- bayesGARCH ---------------------------------------------------------
args(bayesGARCH)
# We leave the defaults, which amounts to a diffuse prior. Then when we take the
# mean of the posterior it will basically be giving us the MLE estimates 
# obtained above.
mcmc <- bayesGARCH(y, control = list(n.chain = 2, l.chain = 10000))
smpl <- formSmpl(mcmc, l.bi = 500)

# Look at the means (and/or medians) in this summary and notice that they are
# very similar to the MLE estimates for the parameters we obtain above.
summary(smpl)









