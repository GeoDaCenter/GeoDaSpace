/* Run Columbus to check results */
clear
clear matrix
clear mata
set memory 1g
/*cd "/Users/dani/repos/spreg/trunk/econometrics/examples/"*/
cd "/home/dani/repos/spreg/trunk/econometrics/examples/"

/*insheet using columbus.csv */
/*spmat import w using "columbus.gal", geoda replace */
/*shp2dta using columbus, database(columbus) coordinates(columbusxy) genid(col_id) gencentroids(c) */
use columbus 
/*Queen*/
spmat contiguity w using columbusxy, norm(row) id(id) replace

/*
reg HOVAL INC CRIME
*/

/* GM error (equivalent commands) Does not match*/
spivreg HOVAL INC CRIME, el(w) id(POLYID)
spreg gs2sls HOVAL INC CRIME, el(w) id(POLYID)

/*
/* IV lag (equivalent commands) Matches R */
spivreg HOVAL INC CRIME, dl(w) id(POLYID)
spreg gs2sls HOVAL INC CRIME, dl(w) id(POLYID)

/* GM error Het */
spivreg HOVAL INC CRIME, el(w) id(polyidn160) het

/* IV lag Het */
spivreg HOVAL INC CRIME, dl(w) id(polyidn160) het
*/

/*cd "/Users/dani/repos/spreg/branches/benchmarking/stata_bm/"*/
cd "/home/dani/repos/spreg/branches/benchmarking/stata_bm/"
