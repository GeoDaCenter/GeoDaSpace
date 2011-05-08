# Script to replicate columbus regression to check STATA
library(spdep)
library(sphet)
library(sem)
library(car)
library(foreign)

##################### COLUMBUS #####################

w <- read.gal('../../../trunk/econometrics/examples/columbus.gal')
w <- nb2listw(w)
dbf <- read.dbf('../../../trunk/econometrics/examples/columbus.dbf')

##### OLS
#model <- lm(HOVAL ~ INC + CRIME, data=dbf)
# White corrention
#white.vars <- hccm(model)

##### STSLS
#model <- tsls(HOVAL ~ INC + CRIME, ~ INC + DISCBD, data=dbf)

##### GM Error
#model <- GMerrorsar(HOVAL ~ INC + CRIME, data=dbf, w, return_LL=TRUE)

##### S2SLS
#model <- stsls(HOVAL ~ INC + CRIME, data=dbf, w)
# White
#model <- stsls(HOVAL ~ INC + CRIME, data=dbf, w, robust=TRUE)

##### SpHet Error
model <- gstslshet(HOVAL ~ INC + CRIME, data=dbf, w, sarar=FALSE, inverse=FALSE)

##### SpHet SARAR
#model <- gstslshet(HOVAL ~ INC + CRIME, data=dbf, w, sarar=TRUE)


##################### NAT #####################

w <- read.gal('../../../trunk/econometrics/examples/NAT.gwt')
w <- nb2listw(w)
dbf <- read.dbf('../../../trunk/econometrics/examples/NAT.dbf')

# STSLS-HAC
stslshac(HR90 ~ RD90 + DV90, data=dbf, w)

print(summary(model))
