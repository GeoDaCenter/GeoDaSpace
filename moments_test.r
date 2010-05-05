#Computation of Moments (G & g) for spHetErr in R using 'sphet'

#Point to data folder
setwd('~/repos/spreg/trunk/examples')

#Install dependencies. 
library(spdep)
library(sphet)
source('~/repos/R/sphet/R/utilities.R')

#Sends all the output from the script to a file in the working directory. 
sink("~/repos/spreg/branches/r_sphet/timing_momentsR.txt")
print('sphet: spatial error model')
print('Note: time in seconds')

#Small dataset (n=100), 1st order rook.
data<-read.csv("n100_stdnorm_vars6.csv")
u<-data$varA
gal<-read.gal("w_rook_n100_order1_k4.gal",override.id=TRUE)
listw<-nb2listw(gal)
t0<-proc.time()
m<-Ggfastfast(listw=listw, u, length(data$varA))
t1<-proc.time()
time<-t1-t0
print('N: 100. Geog: First order rook')
print(time[[3]])

#Small dataset (n=100), 2nd order rook. Note that this uses data imported above.
gal<-read.gal("w_rook_n100_order2_k10.gal",override.id=TRUE)
listw<-nb2listw(gal)
t0<-proc.time()
m<-Ggfastfast(listw=listw, u, length(data$varA))
t1<-proc.time()
time<-t1-t0
print('N: 100. Geog: Second order rook')
print(time[[3]])

##########################################################
# Nick: from here it still runs the whole model, change it.
##########################################################

#Small dataset (n=100), 3rd order rook.
gal<-read.gal("w_rook_n100_order3_k19.gal",override.id=TRUE)
listw<-nb2listw(gal)
t0<-proc.time()
res<-gstslshet(varA~varB + varC,data=data,listw=listw,sarar=FALSE)
t1<-proc.time()
time<-t1-t0
print('N: 100. Geog: Third order rook')
print(time[[3]])

#Medium dataset (n=10000), 1st order rook. Import new data here.
#data<-read.csv("n10000_stdnorm_vars6.csv")
#gal<-read.gal("w_rook_n10000_order1_k4.gal",override.id=TRUE)
#listw<-nb2listw(gal)
#t0<-proc.time()
#res<-gstslshet(varA~varB + varC,data=data,listw=listw,sarar=FALSE)
#t1<-proc.time()
#time<-t1-t0
#print('N: 10000. Geog: First order rook')
#print(time[[3]])

#Medium dataset (n=10000), 2nd order rook.
#gal<-read.gal("w_rook_n10000_order2_k12.gal",override.id=TRUE)
#listw<-nb2listw(gal)
#t0<-proc.time()
#res<-gstslshet(varA~varB + varC,data=data,listw=listw,sarar=FALSE)
#t1<-proc.time()
#time<-t1-t0
#print('N: 10000. Geog: Second order rook')
#print(time[[3]])

#Medium dataset (n=10000), 3rd order rook.
#gal<-read.gal("w_rook_n10000_order3_k23.gal",override.id=TRUE)
#listw<-nb2listw(gal)
#t0<-proc.time()
#res<-gstslshet(varA~varB + varC,data=data,listw=listw,sarar=FALSE)
#t1<-proc.time()
#time<-t1-t0
#print('N: 10000. Geog: Third order rook')
#print(time[[3]])

#Large dataset (n=1000000), 1st order rook. Import new data here.
#data<-read.csv("n1000000_stdnorm_vars6.csv")
#gal<-read.gal("w_rook_n10000_order1_k4.gal",override.id=TRUE)
#listw<-nb2listw(gal)
#t0<-proc.time()
#res<-gstslshet(varA~varB + varC,data=data,listw=listw,sarar=FALSE)
#t1<-proc.time()
#time<-t1-t0
#print(time[[3]])

#Turns off the writing of results. 
sink()

