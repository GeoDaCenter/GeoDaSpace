# Script to fit different copulas to Lee's data

library(foreign)
library(spdep)
source('cop_tools.r')

biPlot <- function(x, mnttl){
    par(mfrow = c(1, 2))
    cops <- cbind('normal', 'clayton')
    #cops <- cbind('clayton')
    for(copula in cops){
        print(paste('Estimating', copula))
        c <- est.cop(x, copula)
        abline(v=mean(x[,1]), cex=0.5)
        abline(h=mean(x[,2]), cex=0.5)
        plot(lm(x[, 1] ~ x[, 2]))
        tit <- paste(copula, 'Theta:', c@estimate[5])
        title(tit)
    }
    title(main=mnttl)
}

w <- read.gal('/Volumes/GeoDa/Workspace/CopulaData/county_geoda2.gal')
#w <- read.gal('/home/dani/AAA/LargeData/CopulaData/county_geoda2.gal')
w <- nb2listw(w)

dbf.link <- '/Volumes/GeoDa/Workspace/CopulaData/county_geoda2.dbf'
#dbf.link <- '/home/dani/AAA/LargeData/CopulaData/county_geoda2.dbf'

dbf <- read.dbf(dbf.link)
vars <- cbind('CSCREEE', 'CSCREEL', 'SSCREEE', 'SSCREEL')
vars <- cbind('CSCREEE')

for(var in vars){
        v <- dbf[, var]
        #wvar <- lag.listw(w, v)
        #x <- matrix(cbind(var, wvar), nrow=length(v))
        dat <- sp.data(10, 0.5)
        w <- dat$w
        x <- dat$dat

        png.link <- paste('~/Desktop/copulaPlots', var, '.png', sep='')
        png(png.link, width=960, height=480)
        biPlot(x, var)
        dev.off()

        vs <- (x[, 1] - mean(x[, 1])) / sqrt(var(x[, 1]))
        wvs <- lag.listw(w, vs)
        x <- matrix(cbind(vs, wvs), nrow=length(vs))
        png.link <- paste('~/Desktop/copulaPlots', var, '_std.png', sep='')
        png(png.link, width=960, height=480)
        biPlot(x, var)
        dev.off()
}

#   png('~/Desktop/copulaPlots.png', width=960, height=960)
#   par(mfrow = c(2, 2))
#   for(var in vars){
#       v <- dbf[, var]
#       wv <- lag.listw(w, v)
#       v <- (v - mean(v)) / sqrt(var(v))
#       # Standardize
#       wv <- (wv - mean(wv)) / sqrt(var(wv))
#       x <- matrix(cbind(v, wv), nrow=length(v))
#       plot(x, xlab='y', ylab='wy', pch=20)
#       mi <- moran.test(v, w)$statistic
#       lines(lowess(v, wv), col='red')
#       tit <- paste(var, mi)
#       title(tit)
#       abline(v=0)
#       abline(h=0)
#   }
#   dev.off()
