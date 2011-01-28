# Script to get a plot of both spdep and spreg performance in one
library(ggplot2)

r <- 'rLog.csv'
r <- read.csv(r)

spreg <- 'spregLog.csv'
spreg <- read.csv(spreg)

p <- ggplot(r, aes(x=n, y=creDa))
p <- p + geom_area(aes(x=n, y=creDa))
#   p <- p + geom_area(data=csv, aes(x=V3, fill=as.factor(1),alpha=0.5))
#   p <- p + geom_area(data=csv, aes(x=V4, fill=as.factor(2),alpha=0.5))
p <- p + scale_fill_discrete(
   legend=T,
   name="Distribución N. publicaciones",
   breaks=as.factor(c(0,1,2)), 
   labels=c('Pre-tesis','Tesis-6','>6 años post-tesis')
)

ggsave(p,file="ggplot.png",width=7,height=5)
