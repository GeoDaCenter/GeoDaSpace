# Experiments w/ copula package in R

library(copula)
library(fCopulae)
library(spdep)

#set.seed(123)

# Generate spatially autocorrelated data for a lattice of 10, 10
lambda <- 0.85
s <- 10
w <- cell2nb(s, s)
e <- runif(100)
u <- invIrM(w, rho=lambda, method="chol", feasible=TRUE) %*% e
wu <- lag.listw(nb2listw(w), u)
u <- rank(u) / (s**2+1)
wu <- rank(wu) / (s**2+1)
x <- matrix(cbind(u, wu), nrow=length(u), ncol=2)

# Generate dots from Clayton copula
cop  <- claytonCopula(0.85)
#cop  <- normalCopula(0.85)
nc <- rcopula(cop, 5)
#nc <- x

#par(mfrow = c(1, 2))

# Compute empirical copula
ec <- dempiricalCopula(nc, N=10)
#image(ec$z)
filled.contour(x=ec$x, y=ec$y, z=ec$z, color = topo.colors, nlevels=10,
    axes=FALSE, frame.plot=FALSE)

#plot(u, wu)


# Graphical devices
#mp <- moran.plot(u)
#plot(u, wu)

### Copula ###
#x <- matrix(cbind(u, wu), nrow=length(u), ncol=2)
#f <- pnorm(x)

#   ##### Normal #####
#   cop <- normalCopula(0.3,dim=2,dispstr="un")
#   nor <- mvdc(copula=cop, 
#               margins=c('norm', 'norm'),
#               paramMargins=list(list(mean=0, sd=1), list(mean=0, sd=3))
#               )
#   rn <- rmvdc(nor, 1000)

#   ##### Clayton #####
#   clc <- archmCopula(family='clayton', dim=2, param=1.95)
#   cl <- mvdc(copula=clc,
#               margins=c('norm', 'norm'),
#               paramMargins=list(list(mean=0, sd=1), list(mean=0, sd=1))
#               )
#   rc <- rmvdc(cl, 100)
#   #plot(rn)
#   #plot(rc)

#   #c <- contour(cl, dmvdc, xlim=c(-3, 3), ylim=c(-3, 3))
#   #points(rc)
#   #c <- contour(nor, dmvdc, xlim=c(-3, 3), ylim=c(-3, 3))

#   fit.clc <- archmCopula(family='clayton', dim=2, param=0.5)
#   fit.cl <- mvdc(copula=fit.clc,
#               margins=c('norm', 'norm'),
#               paramMargins=list(list(mean=2, sd=10), list(mean=2, sd=10))
#               )
#   fitC <- fitMvdc(rc, fit.cl, start=c(0, 1, 0, 1, 0))
#   m1 <- fitC@estimate[1]
#   v1 <- fitC@estimate[2]
#   m2 <- fitC@estimate[3]
#   v2 <- fitC@estimate[4]
#   pa <- fitC@estimate[5]

#   eclc <- archmCopula(family='clayton', dim=2, param=pa)
#   resC <- mvdc(copula=eclc,
#               margins=c('norm', 'norm'),
#               paramMargins=list(list(mean=m1, sd=v1), list(mean=m2, sd=v2))
#               )
#   c <- contour(resC, dmvdc, xlim=c(-3, 3), ylim=c(-3, 3))
#   points(rc)
#   #plot(rc)


