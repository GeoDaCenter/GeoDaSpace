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

est.cop <- function(x, copula){
    #copula     : clayton, frank, gumbel, normal
	if (copula == "normal") {
    	cop <- normalCopula(0.3,dim=2,dispstr="un")
    } else {
    	cop <- archmCopula(family=copula,param=2)
    }
    copd <- mvdc(copula=cop, margins=c('norm', 'norm'), paramMargins=list(list(mean=0, sd=1), list(mean=0, sd=1)))               
    print('Copula generated, estimating...')
    if (copula == "frank" | copula == "gumbel") {
        theta <- 2
    } else {
        theta <- 0
    }
    fitC <- fitMvdc(x, copd, start=c(0, 1, 0, 1, theta))
    print('Estimated!!!')
    m1 <- fitC@estimate[1]
    v1 <- fitC@estimate[2]
    m2 <- fitC@estimate[3]
    v2 <- fitC@estimate[4]
    pa <- fitC@estimate[5]

	if (copula == "normal") {
    	enor <- normalCopula(pa,dim=2,dispstr="un")
    } else {
    	enor <- archmCopula(family=copula,param=pa,dim=2,dispstr="un")
    }
    	
    emvdc <- mvdc(copula=enor,
        margins=c('norm', 'norm'),
        paramMargins=list(list(mean=m1, sd=v1), list(mean=m2, sd=v2))
        )
    # Graphics
    z = dmvdc(emvdc, x)
    c <- contour(emvdc, dmvdc, 
        xlim=c(min(x[, 1]), max(x[, 1])), 
        ylim=c(min(x[, 2]), max(x[, 2])),
        zlim=c(min(z), max(z)),
        lwd=1.5, col="red"
        )
    points(x, pch=20, col="black", cex=0.25)
    fitC
}

coPlot <- function(s, lambda, copula){
    dat <- sp.data(s, lambda)
    f <- est.cop(dat$dat, copula)
    n <- paste('N:', s**2)
    l <- paste('Lda:', lambda)
    p <- paste('Theta:', round(f@estimate[5],2))
    ll <- paste('LL:', round(f@loglik,1))
    tit <- paste(n, l, p, ll, sep=' ')
    t <- title(tit)
}

#   ###################################
#   lambdas <- cbind(0, 0.25, 0.5, 0.75)
#   lambdas <- cbind(0.25, 0.75)
#   s <- 10
#   copula <- "clayton" #(clayton, frank, gumbel, normal)
#   ###################################
#       par(mfrow = c(1, 2))
#       for(lambda in lambdas){
#          #dat <- sp.data(s, lambda)
#          #f <- est.cop(dat$dat, copula)
#           coPlot(s, lambda, copula)
#       }

