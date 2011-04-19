# Functions to generate copula plots

library(copula)
library(spdep)

sp.data <- function(s, lambda){
    # s     : size of the lattice
    w <- cell2nb(s, s)
    e <- runif(s**2)
    u <- invIrM(w, rho=lambda, method="chol", feasible=TRUE) %*% e
    w <- nb2listw(w)
    wu <- lag.listw(w, u)
    list(dat=matrix(cbind(u, wu), nrow=length(u), ncol=2), w=w)
}

cop.norm <- function(x){
    cop <- normalCopula(0.3,dim=2,dispstr="un")
    nor <- mvdc(copula=cop, 
                margins=c('norm', 'norm'),
                paramMargins=list(list(mean=0, sd=1), list(mean=0, sd=1))
                )
    print('Copula generated, estimating...')
    fitC <- fitMvdc(x, nor, start=c(0, 1, 0, 1, 0))
    print('Estimated!!!')
    m1 <- fitC@estimate[1]
    v1 <- fitC@estimate[2]
    m2 <- fitC@estimate[3]
    v2 <- fitC@estimate[4]
    pa <- fitC@estimate[5]

    enor <- normalCopula(pa,dim=2,dispstr="un")
    emvdc <- mvdc(copula=enor,
                margins=c('norm', 'norm'),
                paramMargins=list(list(mean=m1, sd=v1), list(mean=m2, sd=v2))
                )
    c <- contour(emvdc, dmvdc, 
        xlim=c(min(x[, 1]), max(x[, 1])), 
        ylim=c(min(x[, 2]), max(x[, 2])),
        lwd=0.5, col='red'
        )
    points(x, pch=20)
    fitC
}

coPlot <- function(s, lambda){
    dat <- sp.data(s, lambda)
    f <- cop.norm(dat$dat)
    n <- paste('N:', s**2)
    l <- paste('Lambda:', lambda)
    m <- paste('Moran:', round(moran.test(dat$dat[, 1], dat$w)$statistic,4))
    p <- paste('Theta:', round(f@estimate[5],4))
    tit <- paste(n, m, l, p, sep='     ')
    t <- title(tit)
}

###################################
lambdas <- cbind(0, 0.25, 0.5, 0.75)
s <- 40 
###################################
par(mfrow = c(2, 2))
for(lambda in lambdas){
    coPlot(s, lambda)
}

