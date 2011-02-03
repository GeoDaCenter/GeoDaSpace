# Script to get a plot of both spdep and spreg performance in one
library(ggplot2)

spreg <- 'spregLog.csv'
spreg <- 'large_sw_rplot.csv'
spreg <- read.csv(spreg)

p <- ggplot(spreg)
p <- p + geom_line(aes(x=n, y=lm, colour=as.factor(0), alpha=0.4))
p <- p + geom_line(aes(x=n, y=moran, colour=as.factor(1), alpha=0.4))
p <- p + geom_line(aes(x=n, y=reg, colour=as.factor(2)))
p <- p + scale_colour_discrete(
    #palette=1,
    legend=T,
    name="Performance",
    breaks=as.factor(c(0,1,2)), 
    labels=c('LM Diagnostics','Residual Moran', 'Regression')
)
p <- p + scale_alpha(legend=F)
p <- p + opts(title = 'Simulation using lat2SW')

p <- p + opts(panel.grid.major = theme_blank())
#p <- p + opts(panel.grid.minor = theme_blank())
p <- p + opts(panel.background = theme_blank())
p <- p + labs(x='N', y='Time (seconds)')
p <- p + opts(panel.border = theme_blank())


ggsave(p,file="sim_lat2SW.png",width=7,height=5)

print(p)
