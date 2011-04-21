/* Run Columbus to check results */
clear
cd "/Users/dani/repos/spreg/trunk/econometrics/examples/"

insheet using columbus.csv
spmat import w using "columbus.gal", geoda replace 

/* GM error */
spivreg hovaln96 incn96 crimen96, el(w) id(polyidn160)

/* IV lag */
spivreg hovaln96 incn96 crimen96, dl(w) id(polyidn160)

/* GM error Het */
spivreg hovaln96 incn96 crimen96, el(w) id(polyidn160) het

/* IV lag Het */
spivreg hovaln96 incn96 crimen96, dl(w) id(polyidn160) het

cd "/Users/dani/repos/spreg/branches/benchmarking/stata_bm/"
