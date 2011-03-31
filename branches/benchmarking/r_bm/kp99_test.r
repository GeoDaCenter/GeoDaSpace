#Script to test results of spError.GSLS against spdep's 

library(foreign)
library(spdep)

dbf <- read.dbf('../../../trunk/econometrics/examples/columbus.dbf')
y <- dbf$HOVAL
x <- cbind(dbf$INC, dbf$CRIME)
w <- read.gal('../../../trunk/econometrics/examples/columbus.GAL')
w <- nb2listw(w)

model <- GMerrorsar(y ~ INC + CRIME, data=dbf, w, return_LL=TRUE)
