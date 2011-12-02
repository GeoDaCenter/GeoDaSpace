# Script to replicate in R large_sim_data.py and benchmark
# NOTE: it only compares OLS and spatial diagnostics

library(spdep)
library(sphet)
source('/Users/dani/code/spdep/R/lm.moran_dani.R')

test.large.olsSPd <- function(s, k){
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
    print('MORAN')
    print(moran)
    t1 <- proc.time()
    time <- t1 - t0
    print('Moran test:')
    print(time[3])

    total <- t1 - ti
    print('Total time:')
    print(total[3])

}

test.large.GMSWLS <- function(s, k){
    n <- s**2
    mes <- paste('N:', n)
    print(mes)
    model <- 'Model: GMSWLS'
    print(model)

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
    ols <- GMerrorsar(y ~ x, data=list(), w)
    t1 <- proc.time()
    time <- t1 - t0
    print('Regression:')
    print(time[3])

    total <- t1 - ti
    print('Total time:')
    print(total[3])

}

test.large.STSLS  <- function(s, k){
    n <- s**2
    mes <- paste('N:', n)
    print(mes)
    model <- 'Model: STSLS'
    print(model)

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
    ols <- stsls(y ~ x, data=list(), w)
    t1 <- proc.time()
    time <- t1 - t0
    print('Regression:')
    print(time[3])

    total <- t1 - ti
    print('Total time:')
    print(total[3])

}

test.large.sp.models  <- function(s, k){
    n <- s**2
    mes <- paste('N:', n)
    print(mes)
    model <- 'Model: STSLS'
    print(model)

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
    ols <- GMerrorsar(y ~ x, data=list(), w)
    t1 <- proc.time()
    time <- t1 - t0
    print('GMSWLS:')
    print(time[3])

    t0 <- proc.time()
    ols <- stsls(y ~ x, data=list(), w)
    t1 <- proc.time()
    time <- t1 - t0
    print('STSLS:')
    print(time[3])

    total <- t1 - ti
    print('Total time:')
    print(total[3])

}

test.large.spHet_error.models  <- function(s, k){
    n <- s**2
    mes <- paste('N:', n)
    print(mes)
    model <- 'Model: Het'
    print(model)

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
    ols <- gstslshet(y ~ x, data=list(), w, sarar=FALSE)
    t1 <- proc.time()
    time <- t1 - t0
    print('SWLS_Het:')
    print(time[3])

    total <- t1 - ti
    print('Total time:')
    print(total[3])

}

test.large.spHet_sarar.models  <- function(s, k){
    n <- s**2
    mes <- paste('N:', n)
    print(mes)
    model <- 'Model: Het'
    print(model)

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
    ols <- gstslshet(y ~ x, data=list(), w)
    t1 <- proc.time()
    time <- t1 - t0
    print('GSTSLS_Het_lag:')
    print(time[3])

    total <- t1 - ti
    print('Total time:')
    print(total[3])

}

test.small.all  <- function(s, k){
    n <- s**2
    mes <- paste('N:', n)
    print(mes)
    model <- 'Model: Het'
    print(model)

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

    t0 <- proc.time()
    ols <- GMerrorsar(y ~ x, data=list(), w)
    t1 <- proc.time()
    time <- t1 - t0
    print('GMSWLS:')
    print(time[3])

    t0 <- proc.time()
    ols <- stsls(y ~ x, data=list(), w)
    t1 <- proc.time()
    time <- t1 - t0
    print('STSLS:')
    print(time[3])

    t0 <- proc.time()
    ols <- gstslshet(y ~ x, data=list(), w, sarar=FALSE)
    t1 <- proc.time()
    time <- t1 - t0
    print('SWLS_Het:')
    print(time[3])

    t0 <- proc.time()
    ols <- gstslshet(y ~ x, data=list(), w)
    t1 <- proc.time()
    time <- t1 - t0
    print('GSTSLS_Het_lag:')
    print(time[3])

    total <- t1 - ti
    print('Total time:')
    print(total[3])

}

k <- 10
sizes <- cbind(150, 300, 450, 600, 750, 800, 850, 900, 1000, 1250, 1500, 1750, 2000, 2500)
#sizes <- cbind(1150, 1300, 1450, 1600, 1750, 1900, 2000)
#sizes <- cbind(15)
#sizes <- cbind(30, 50, 70, 100)
#sizes <- cbind(10)

#for(size in sizes){
   #mes <- paste('Evaluating size:', size**2)
   #print(mes)

   #sink('/Users/dani/Desktop/ols_r.log', append=TRUE)
   #test.large.olsSPd(size, k)
   #sink()

   #sink('/Users/dani/Dropbox/aagLogs/gmswls_r.log', append=TRUE)
   #test.large.GMSWLS(size, k)
   #sink()

   #sink('/Users/dani/Dropbox/aagLogs/stsls_r.log', append=TRUE)
   #test.large.STSLS(size, k)
   #sink()

   #sink('/Users/dani/Dropbox/aagLogs/spHet_error_r.log', append=TRUE)
   #test.large.spHet_error.models(size, k)
   #sink()

   #sink('/Users/dani/Dropbox/aagLogs/spHet_sarar_r.log', append=TRUE)
   #test.large.spHet_sarar.models(size, k)
   #sink()

   #sink('/home/dani/Dropbox/aagLogs/smAll_r.log', append=TRUE)
   #test.large.spHet_sarar.models(size, k)
   #sink()

}

