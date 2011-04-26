# Script to replicate columbus regression to check STATA
library(spdep)
library(foreign)

w <- read.gal('../../../trunk/econometrics/examples/columbus.gal')
w <- nb2listw(w)
dbf <- read.dbf('../../../trunk/econometrics/examples/columbus.dbf')

# GM Error
#model <- GMerrorsar(HOVAL ~ INC + CRIME, data=dbf, w, return_LL=TRUE)

# S2SLS
model <- stsls(HOVAL ~ INC + CRIME, data=dbf, w)

summary(model)
