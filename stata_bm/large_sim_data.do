
/* Estimate spatial models in STATA */

/* Import a .gal file */
spmat import w using "/Users/dani/repos/spreg/branches/benchmarking/stata_bm/rook100.gal", geoda replace
spmat export w using "/Users/dani/repos/spreg/branches/benchmarking/stata_bm/rook100.spmat"
spmat use w using "/Users/dani/repos/spreg/branches/benchmarking/stata_bm/rook100.spmat", replace 

/*normalize(spe) replace*/

