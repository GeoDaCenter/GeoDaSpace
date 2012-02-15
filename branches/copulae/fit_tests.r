library(copula)

myAnalysis <- function(myLoss) {
    pseudoLoss <- sapply(myLoss, rank, ties.method='random')

    it.sim <- indepTestSim(nrow(pseudoLoss), 2, print.every=0)
    indTest <- indepTest(pseudoLoss, it.sim)$global.statistic.pvalue
    gof.g <- gofCopula(gumbelCopula(1), pseudoLoss, method = "itau",
                       simulation = "mult")$pvalue
    gof.c <- gofCopula(claytonCopula(1), pseudoLoss, method = "itau",
                       simulation = "mult")$pvalue
    gof.f <- gofCopula(frankCopula(1), pseudoLoss, method = "itau",
                       simulation = "mult")$pvalue
    gof.n <- gofCopula(normalCopula(0), pseudoLoss, method = "itau",
                       simulation = "mult")$pvalue
    gof.p <- gofCopula(plackettCopula(1), pseudoLoss, method = "itau",
                       simulation = "mult")$pvalue
    gof.t <- gofCopula(tCopula(0, df = 4, df.fixed = TRUE), pseudoLoss,
                       method = "itau", simulation = "mult")$pvalue
    fit.g <- fitCopula(gumbelCopula(1), pseudoLoss, method = "itau")
    c(indep = indTest, gof.g = gof.g, gof.c = gof.c, gof.f = gof.f,
      gof.n = gof.n, gof.t = gof.t, gof.p = gof.p, est = fit.g@estimate,
      se = sqrt(fit.g@var.est))
}

set.seed(123)

# Generate spatially autocorrelated data for a lattice of 10, 10
lambda <- 0.5

e <- runif(100)
u <- invIrM(w, rho=lambda, method="chol", feasible=TRUE) %*% e
wu <- lag.listw(nb2listw(w), u)

# Get the rank
x <- list(x=u, y=wu)
rx <- sapply(x, rank, ties.method='random')

# Test of independence
   it.sim <- indepTestSim(nrow(u), 2, print.every=0)
   it <- indepTest(rx, it.sim)
   print(it)

# Goodness of fit

# gof "normal"
normalCop <- normalCopula(1)

# gof "clayton"
claytonCop <- claytonCopula(1)

# gof "gumbel"
gumbelCop <- gumbelCopula(1)

# gof "frank"
frankCop <- frankCopula(1)

# gof "plackett"
plackettCop <- plackettCopula(1)

copulas <- c(normalCop, claytonCop, gumbelCop, frankCop, plackettCop)

for(cop in copulas){
    #print(paste('Copula', cop))
    gof <- gofCopula(cop, rx, 
                     method='itau', simulation='mult')
    print(gof)
}

#ma <- myAnalysis(x)
#myReps <- t(replicate(10, myAnalysis(x)))
#round(apply(myReps, 2, summary), 3)

