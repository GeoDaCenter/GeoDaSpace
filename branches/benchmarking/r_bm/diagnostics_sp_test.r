#Script to test diagnostics_sp module against R

library(foreign)
library(spdep)

dbf <- read.dbf('../../../trunk/econometrics/examples/columbus.dbf')
y <- dbf$CRIME
x <- cbind(dbf$INC, dbf$HOVAL)
w <- read.gal('../../../trunk/econometrics/examples/columbus.GAL')
w <- nb2listw(w)

#model <- lm(y ~ x, data=dbf)
model <- stsls(y ~ x, data=dbf, w)

#moran <- lm.morantest(model, w)
#print(moran)

#lm <- lm.LMtests(model, w, test='SARMA')
#print(lm)
