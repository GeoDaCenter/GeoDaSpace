/* Run Columbus to check results */
clear
clear matrix
clear mata
set memory 1g
cd "/Users/dani/code/spreg/trunk/econometrics/examples/"
/*cd "/home/dani/code/spreg/trunk/econometrics/examples/"*/

/*insheet using columbus.csv */
/*Opt A*/
/*spmat import w using "columbus.gal", geoda replace */

/*Opt B*/
shp2dta using NAT, database(nat) coordinates(natxy) genid(ids) gencentroids(nat_c) replace

use nat
/*Queen*/
spmat contiguity w using natxy, norm(row) id(ids) replace

/* GM error Het */
/*spivreg HR90 MA90 DV90, el(w) id(ids) het*/
spivreg HR90 MA90 DV90, el(w) id(ids) het
    /* VC matrix Omega */
    matrix list e(V)
    /* Initial estimation of lambda */
    display e(rho_2sls)
    /* Initial estimation of betas */
    matrix list e(delta_2sls)

