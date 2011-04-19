# Script to fit different copulas to Lee's data

library(foreign)
library(spdep)

w <- read.gal('/Volumes/GeoDa/Workspace/CopulaData/county_geoda2.gal')
w <- nb2listw(w)

dbf.link <- '/Volumes/GeoDa/Workspace/CopulaData/county_geoda2.dbf'

dbf <- read.dbf(dbf.link)
vars <- cbind('CSCREEE', 'CSCREEL', 'SSCREEE', 'SSCREEL')

png('~/Desktop/copulaPlots.png', width=960, height=960)
par(mfrow = c(2, 2))
for(var in vars){
    v <- dbf[, var]
    wv <- lag.listw(w, v)
    v <- (v - mean(v)) / sqrt(var(v))
    # Standardize
    wv <- (wv - mean(wv)) / sqrt(var(wv))
    x <- matrix(cbind(v, wv), nrow=length(v))
    plot(x, xlab='y', ylab='wy', pch=20)
    mi <- moran.test(v, w)$statistic
    tit <- paste(var, mi)
    title(tit)
    abline(v=0)
    abline(h=0)
}
dev.off()
