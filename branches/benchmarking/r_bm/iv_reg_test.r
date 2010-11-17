# Script to test different IV estimation classes (TSLS, STSLS...)

library(foreign)
library(spdep)

dbf <- read.dbf('../../../trunk/econometrics/examples/columbus.dbf')
y <- dbf$CRIME
x <- cbind(dbf$INC, dbf$HOVAL)
w <- read.gal('../../../trunk/econometrics/examples/columbus.GAL')
w <- nb2listw(w)

model <- stsls(y ~ x, data=list(), w)
print(model)
