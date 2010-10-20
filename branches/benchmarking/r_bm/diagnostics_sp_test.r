#Script to test diagnostics_sp module against R

library(foreign)
library(spdep)

dbf <- read.dbf('../../../trunk/econometrics/examples/columbus.dbf')
y <- dbf$HOVAL
x <- cbind(dbf$INC, dbf$CRIME)
w <- read.gal('../../../trunk/econometrics/examples/columbus.GAL')
w <- nb2listw(w)

model <- lm(y ~ x, data=dbf)

moran <- lm.morantest(model, w)
print(moran)

