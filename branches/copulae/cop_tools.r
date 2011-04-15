# Functions to generate copula plots

library(copula)
library(spdep)

sp.data <- function(s, lambda){
    # s     : size of the lattice
    w <- cell2nb(s, s)
    e <- runif(s**2)
    u <- invIrM(w, rho=lambda, method="chol", feasible=TRUE) %*% e
    wu <- lag.listw(nb2listw(w), u)
    matrix(cbind(u, wu), nrow=length(u), ncol=2)
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
        lwd=0.2
        )
    points(x, pch=20)
    fitC
}

coPlot <- function(s, lambda){
    f <- cop.norm(sp.data(s, lambda))
    n <- paste('N:', s**2)
    l <- paste('Lambda:', lambda)
    p <- paste('Theta:', f@estimate[5])
    tit <- paste(n, l, p, sep='     ')
    t <- title(tit)
}

###################################
lambdas <- cbind(0, 0.25, 0.5, 0.75)
s <- 4
###################################
par(mfrow = c(2, 2))
for(lambda in lambdas){
    coPlot(s, lambda)
}

