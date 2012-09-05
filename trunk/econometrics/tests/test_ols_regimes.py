import unittest
import numpy as np
import pysal
from econometrics.ols_regimes import OLS_Regimes

PEGP = pysal.examples.get_path

class TestOLS_regimes(unittest.TestCase):
    def setUp(self):
        db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
        self.y_var = 'CRIME'
        self.y = np.array([db.by_col(self.y_var)]).reshape(49,1)
        self.x_var = ['INC','HOVAL']
        self.x = np.array([db.by_col(name) for name in self.x_var]).T
        self.r_var = 'NSA'
        self.regimes = db.by_col(self.r_var)
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_OLS(self):
        start_suppress = np.get_printoptions()['suppress']
        np.set_printoptions(suppress=True)    
        ols = OLS_Regimes(self.y, self.x, self.regimes, w=self.w, constant_regi='many', nonspat_diag=False, spat_diag=True, name_y=self.y_var, name_x=self.x_var, name_ds='columbus', name_regimes=self.r_var, name_w='columbus.gal')        
        #np.testing.assert_array_almost_equal(ols.aic, 408.73548964604873 ,7)
        np.testing.assert_array_almost_equal(ols.ar2,0.50761700679873101 ,7)
        np.testing.assert_array_almost_equal(ols.betas,np.array([[ 68.78670869],\
                [ -1.9864167 ],[ -0.10887962],[ 67.73579559],[ -1.36937552],[ -0.31792362]])) 
        np.testing.assert_array_almost_equal(ols.lm_error, \
            (5.92970357,  0.01488775),7)
        np.testing.assert_array_almost_equal(ols.lm_lag, \
            (8.78315751,  0.00304024), 7)
        np.testing.assert_array_almost_equal(ols.lm_sarma, \
                (8.89955982,  0.01168114), 7)
        np.testing.assert_array_almost_equal(ols.mean_y, \
            35.1288238979591,7)
        #bp = np.array([2, 5.7667905131212587, 0.05594449410070558])
        #ols_bp = np.array([ols.breusch_pagan['df'], ols.breusch_pagan['bp'], ols.breusch_pagan['pvalue']])
        #np.testing.assert_array_almost_equal(bp, ols_bp, 7)
        #np.testing.assert_array_almost_equal(ols.f_stat, \
        #    (12.358198885356581, 5.0636903313953024e-05), 7)
        #jb = np.array([2, 39.706155069114878, 2.387360356860208e-09])
        #ols_jb = np.array([ols.jarque_bera['df'], ols.jarque_bera['jb'], ols.jarque_bera['pvalue']])
        #np.testing.assert_array_almost_equal(ols_jb,jb, 7)
        #white = np.array([5, 2.90606708, 0.71446484])
        #ols_white = np.array([ols.white['df'], ols.white['wh'], ols.white['pvalue']])
        #np.testing.assert_array_almost_equal(ols_white,white, 7)
        #kb = {'df': 2, 'kb': 2.2700383871478675, 'pvalue': 0.32141595215434604}
        #for key in kb:
        #    self.assertAlmostEqual(ols.koenker_bassett[key],  kb[key], 7)
        #np.testing.assert_array_almost_equal(ols.logll, -201.3677448230244 ,7)
        #np.testing.assert_array_almost_equal(ols.moran_res[0], \
        #    0.20373540938,7)
        #np.testing.assert_array_almost_equal(ols.moran_res[1], \
        #    2.59180452208,7)
        #np.testing.assert_array_almost_equal(ols.moran_res[2], \
        #    0.00954740031251,7)
        #np.testing.assert_array_almost_equal(ols.mulColli, \
        #    12.537554873824675 ,7)
        #np.testing.assert_array_almost_equal(ols.schwarz, \
        #    414.41095054038061,7 )
        #np.testing.assert_array_almost_equal(ols.sig2ML, \
        #    217.28602192257551,7 )
        np.testing.assert_equal(ols.k, 6)
        np.testing.assert_equal(ols.kf, 0)
        np.testing.assert_equal(ols.kr, 3)
        np.testing.assert_equal(ols.n, 49)
        np.testing.assert_equal(ols.nr, 2)
        np.testing.assert_equal(ols.name_ds,  'columbus')
        np.testing.assert_equal(ols.name_gwk,  None)
        np.testing.assert_equal(ols.name_w,  'columbus.gal')
        np.testing.assert_equal(ols.name_x,  ['0.0_CONSTANT', '0.0_INC', '0.0_HOVAL', '1.0_CONSTANT', '1.0_INC', '1.0_HOVAL'])
        np.testing.assert_equal(ols.name_y,  'CRIME')
        np.testing.assert_array_almost_equal(ols.predy[3], np.array([
            51.05003696]),7)
        np.testing.assert_array_almost_equal(ols.r2, \
                0.55890690192386316 ,7)
        np.testing.assert_array_almost_equal(ols.rlm_error, \
                (0.11640231,  0.73296972),7)
        np.testing.assert_array_almost_equal(ols.rlm_lag, \
            (2.96985625,  0.08482939), 7)
        np.testing.assert_equal(ols.robust,  'unadjusted')
        np.testing.assert_array_almost_equal(ols.sig2, \
            137.84897351821013,7 )
        np.testing.assert_array_almost_equal(ols.sig2n, \
                120.96950737312316, 7)
        np.testing.assert_array_almost_equal(ols.t_stat[2][0], \
                -0.43342216706091791,7)
        np.testing.assert_array_almost_equal(ols.t_stat[2][1], \
                0.66687472578594531,7)
        np.set_printoptions(suppress=start_suppress)        

if __name__ == '__main__':
    unittest.main()
