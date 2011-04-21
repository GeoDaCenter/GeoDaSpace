clear

cd "/Users/dani/repos/spreg/branches/benchmarking/stata_bm/"

/* Estimate spatial models in STATA */

drawnorm setter, n(100)

/* Parameters */
generate k = 1
generate s = 50
generate n = s^2

/* y */
drawnorm y, n(100) m(0) sd(1)
/* X */
drawnorm x0, n(100) m(0) sd(1)
drawnorm x1, n(100) m(0) sd(1)
drawnorm x2, n(100) m(0) sd(1)
drawnorm x3, n(100) m(0) sd(1)
drawnorm x4, n(100) m(0) sd(1)
drawnorm x5, n(100) m(0) sd(1)
drawnorm x6, n(100) m(0) sd(1)
drawnorm x7, n(100) m(0) sd(1)
drawnorm x8, n(100) m(0) sd(1)
drawnorm x9, n(100) m(0) sd(1)

/* weights */
spmat import w using "rook100.gal", geoda replace

