# Script to replicate NAT regression to check against other software

library(spdep)
#library(sphet)

library(sem)
#library(car)
library(foreign)

setwd("~/code/spreg/branches/benchmarking/r_bm")

source("~/code/sphet/R/dist_functions.R")
source("~/code/sphet/R/distance.R")
source("~/code/sphet/R/gs2slshet_bak.R")
source("~/code/sphet/R/gwt2dist.R")
source("~/code/sphet/R/kernelsfun.R")
source("~/code/sphet/R/listw2dgCMatrix.R")
source("~/code/sphet/R/Omega.R")
source("~/code/sphet/R/s2slshac.R")
source("~/code/sphet/R/summary.sphet.R")
source("~/code/sphet/R/twostagels.R")
source("~/code/sphet/R/utilities.R")


w <- read.gal('../../../trunk/econometrics/examples/NAT_queen.gal',
    override.id=TRUE)
w <- nb2listw(w)
dbf <- read.dbf('../../../trunk/econometrics/examples/NAT.dbf')

# STSLS-HAC
#model <- stslshac(HR90 ~ RD90 + DV90, data=dbf, w)
model <- gstslshet(HR90 ~ MA90 + DV90, data=dbf, w)
#model <- lm(HR90 ~ MA90 + DV90, data=dbf)

print(summary(model))
