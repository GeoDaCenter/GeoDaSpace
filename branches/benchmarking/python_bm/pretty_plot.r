# Script to get a plot of both spdep and spreg performance in one
library(ggplot2)

r <- 'rLog.csv'
r <- read.csv(r)

spreg <- 'spregLog.csv'
spreg <- read.csv(spreg)

p <- ggplot(r)
p <- p + geom_area(aes(x=n, y=reg, fill=as.factor(1), alpha=0.4))
p <- p + geom_area(aes(x=n, y=lm, fill=as.factor(2), colour=as.factor(2), alpha=0.4))
p <- p + geom_area(aes(x=n, y=moran, fill=as.factor(3), colour=as.factor(3), alpha=0.4))
p <- p + scale_fill_discrete(
    #palette=1,
    legend=T,
    name="Performance",
    breaks=as.factor(c(1,2,3)), 
    labels=c('Regression','LM Diagnostics', 'Residual Moran')
)
p <- p + scale_alpha(legend=F)
p <- p + scale_colour(legend=T)

p <- p + opts(panel.grid.major = theme_blank())
#p <- p + opts(panel.grid.minor = theme_blank())
#p <- p + opts(panel.background = theme_blank())
p <- p + labs(x='N', y='Time')
p <- p + opts(panel.border = theme_blank())



ggsave(p,file="ggplot.png",width=7,height=5)

print(p)
