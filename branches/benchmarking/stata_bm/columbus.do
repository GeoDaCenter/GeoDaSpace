/* Run Columbus to check results */
clear
clear matrix
clear mata
set memory 1g
cd "/Users/dani/code/spreg/trunk/econometrics/examples/"
/*cd "/home/dani/repos/spreg/trunk/econometrics/examples/"*/
/*cd "C:\Users\Pedro\Documents\Academico\GeodaCenter\python\SVN\spreg\trunk\econometrics\examples"*/

/*insheet using columbus.csv */
/*Opt A*/
/*spmat import w using "columbus.gal", geoda replace */
/*Opt B*/
shp2dta using columbus, database(columbus) coordinates(columbusxy) genid(col_id) gencentroids(c) replace

use columbus 
/*Queen*/
spmat contiguity w using columbusxy, norm(row) id(col_id) replace

/* OLS
reg HOVAL INC CRIME
reg HOVAL INC CRIME, robust
*/

/* TSLS
ivreg  HOVAL (CRIME =  DISCBD) INC
ivreg  HOVAL (CRIME =  DISCBD) INC, robust
*/

/* GM error (equivalent commands) */
/*spivreg HOVAL INC CRIME, el(w) dl(w) id(col_id)
spivreg HOVAL INC (CRIME =  DISCBD), el(w) dl(w) id(col_id)
spivreg HOVAL INC CRIME, el(w) id(col_id)
spivreg HOVAL INC (CRIME =  DISCBD), el(w) id(col_id) het*/
/*spreg gs2sls HOVAL INC CRIME, el(w) id(POLYID)*/

reg HOVAL INC CRIME

*/
/* IV lag (equivalent commands) Matches R */
/*spivreg HOVAL INC CRIME, dl(w) id(POLYID)
spreg gs2sls HOVAL INC CRIME, dl(w) id(POLYID)
spivreg HOVAL INC (CRIME =  DISCBD), dl(w) id(POLYID)
*/

/*
/* GM error Het */
spivreg HOVAL INC CRIME, el(w) id(POLYID) het
spivreg HOVAL INC (CRIME =  DISCBD), el(w) id(POLYID) het

/* IV lag Het */
spivreg HOVAL INC CRIME, dl(w) id(POLYID) het

/* IV combo Het */
spivreg HOVAL INC CRIME, el(w) dl(w) id(POLYID) het
spivreg HOVAL INC (CRIME =  DISCBD), el(w) dl(w) id(POLYID) het
*/


    /* VC matrix Omega */
    matrix list e(V)
    /* Initial estimation of lambda */
    display e(rho_2sls)
    /* Initial estimation of betas */
    matrix list e(delta_2sls)


cd "/Users/dani/code/spreg/branches/benchmarking/stata_bm/"
/*cd "/home/dani/repos/spreg/branches/benchmarking/stata_bm/"*/
