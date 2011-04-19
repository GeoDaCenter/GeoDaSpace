# Experiments w/ copula package in R

library(copula)
library(spdep)

# Generate spatially autocorrelated data for a lattice of 10, 10
lambda <- 0.5
w <- cell2nb(10, 10)
e <- runif(100)
u <- invIrM(w, rho=lambda, method="chol", feasible=TRUE) %*% e
wu <- lag.listw(nb2listw(w), u)

# Graphical devices
#mp <- moran.plot(u)
#plot(u, wu)

### Copula ###
#x <- matrix(cbind(u, wu), nrow=length(u), ncol=2)
#f <- pnorm(x)

##### Normal #####
cop <- normalCopula(0.3,dim=2,dispstr="un")
nor <- mvdc(copula=cop, 
            margins=c('norm', 'norm'),
            paramMargins=list(list(mean=0, sd=1), list(mean=0, sd=3))
            )
rn <- rmvdc(nor, 1000)

##### Clayton #####
clc <- archmCopula(family='clayton', dim=2, param=0.5)
cl <- mvdc(copula=clc,
            margins=c('norm', 'norm'),
            paramMargins=list(list(mean=0, sd=1), list(mean=0, sd=1))
            )
rc <- rmvdc(cl, 100)
#plot(rn)
#plot(rc)

#c <- contour(cl, dmvdc, xlim=c(-3, 3), ylim=c(-3, 3))
#points(rc)
#c <- contour(nor, dmvdc, xlim=c(-3, 3), ylim=c(-3, 3))

fit.clc <- archmCopula(family='clayton', dim=2, param=0.5)
fit.cl <- mvdc(copula=fit.clc,
            margins=c('norm', 'norm'),
            paramMargins=list(list(mean=2, sd=10), list(mean=2, sd=10))
            )
fitC <- fitMvdc(rc, fit.cl, start=c(0, 1, 0, 1, 0))
m1 <- fitC@estimate[1]
v1 <- fitC@estimate[2]
m2 <- fitC@estimate[3]
v2 <- fitC@estimate[4]
pa <- fitC@estimate[5]

eclc <- archmCopula(family='clayton', dim=2, param=pa)
resC <- mvdc(copula=eclc,
            margins=c('norm', 'norm'),
            paramMargins=list(list(mean=m1, sd=v1), list(mean=m2, sd=v2))
            )
c <- contour(resC, dmvdc, xlim=c(-3, 3), ylim=c(-3, 3))
points(rc)
#plot(rc)


