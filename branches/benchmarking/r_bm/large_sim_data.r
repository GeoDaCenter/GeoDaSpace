# Script to replicate in R large_sim_data.py and benchmark
# NOTE: it only compares OLS and spatial diagnostics

library(spdep)

test.large.sp <- function(s, k){
    n <- s**2
    mes <- paste('N:', n)
    print(mes)

    ti <- proc.time()
    t0 <- proc.time()
    y <- rnorm(n)
    x <- replicate(k, rnorm(n))
    t1 <- proc.time()
    time <- t1 - t0
    print('Create data:')
    print(time[3])

    t0 <- proc.time()
    w <- cell2nb(s, s)
    w <- nb2listw(w)
    t1 <- proc.time()
    time <- t1 - t0
    print('Created weights:')
    print(time[3])

    t0 <- proc.time()
    ols <- lm(y ~ x)
    t1 <- proc.time()
    time <- t1 - t0
    print('Regression:')
    print(time[3])

    t0 <- proc.time()
    lms <- lm.LMtests(ols, w, test='all')
    t1 <- proc.time()
    time <- t1 - t0
    print('LM diagnostics:')
    print(time[3])

    t0 <- proc.time()
    moran <- lm.morantest(ols, w)
    t1 <- proc.time()
    time <- t1 - t0
    print('Moran test:')
    print(time[3])

    total <- t1 - ti
    print('Total time:')
    print(total[3])

}

k <- 10
sizes <- cbind(150, 300, 450, 600, 750, 800, 900, 1000)
sizes <- cbind(1150, 1300, 1450, 1600, 1750, 1900, 2000)
#sizes <- cbind(150)

for(size in sizes){
    mes <- paste('Evaluating size:', size**2)
    print(mes)
    sink('large-data-sim.txt', append=TRUE)
    test.large.sp(size, k)
    sink()
}

