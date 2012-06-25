import unittest
import numpy as np
import pysal
import econometrics as EC

PEGP = pysal.examples.get_path

class TestBaseOLS(unittest.TestCase):
    def setUp(self):
        db = pysal.open(PEGP('columbus.dbf'),'r')
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.w = pysal.weights.rook_from_shapefile(PEGP("columbus.shp"))

    def test_ols(self):
        self.X = np.hstack((np.ones(self.y.shape),self.X))
        ols = EC.ols.BaseOLS(self.y,self.X)
        np.testing.assert_array_almost_equal(ols.betas, np.array([[
            46.42818268], [  0.62898397], [ -0.48488854]]))
        vm = np.array([[  1.74022453e+02,  -6.52060364e+00,  -2.15109867e+00],
           [ -6.52060364e+00,   2.87200008e-01,   6.80956787e-02],
           [ -2.15109867e+00,   6.80956787e-02,   3.33693910e-02]])
        np.testing.assert_array_almost_equal(ols.vm, vm,6)

    def test_OLS(self):
        ols = EC.OLS(self.y, self.X, self.w, spat_diag=True, moran=True, \
                name_y='home value', name_x=['income','crime'], \
                name_ds='columbus')
        
        np.testing.assert_array_almost_equal(ols.aic, \
                408.73548964604873 ,7)
        np.testing.assert_array_almost_equal(ols.ar2, \
                0.32123239427957662 ,7)
        np.testing.assert_array_almost_equal(ols.betas, \
                np.array([[ 46.42818268], [  0.62898397], \
                    [ -0.48488854]]), 7) 
        bp = np.array([2, 5.7667905131212587, 0.05594449410070558])
        ols_bp = np.array([ols.breusch_pagan['df'], ols.breusch_pagan['bp'], ols.breusch_pagan['pvalue']])
        np.testing.assert_array_almost_equal(bp, ols_bp, 7)
        np.testing.assert_array_almost_equal(ols.f_stat, \
            (12.358198885356581, 5.0636903313953024e-05), 7)
        jb = np.array([2, 39.706155069114878, 2.387360356860208e-09])
        ols_jb = np.array([ols.jarque_bera['df'], ols.jarque_bera['jb'], ols.jarque_bera['pvalue']])
        np.testing.assert_array_almost_equal(ols_jb,jb, 7)
        np.testing.assert_equal(ols.k,  3)
        kb = {'df': 2, 'kb': 2.2700383871478675, 'pvalue': 0.32141595215434604}
        for key in kb:
            self.assertAlmostEqual(ols.koenker_bassett[key],  kb[key], 7)
        np.testing.assert_array_almost_equal(ols.lm_error, \
            (4.1508117035117893, 0.041614570655392716),7)
        np.testing.assert_array_almost_equal(ols.lm_lag, \
            (0.98279980617162233, 0.32150855529063727), 7)
        np.testing.assert_array_almost_equal(ols.lm_sarma, \
                (4.3222725729143736, 0.11519415308749938), 7)
        np.testing.assert_array_almost_equal(ols.logll, \
                -201.3677448230244 ,7)
        np.testing.assert_array_almost_equal(ols.mean_y, \
            38.436224469387746,7)
        np.testing.assert_array_almost_equal(ols.moran_res[0], \
            0.20373540938,7)
        np.testing.assert_array_almost_equal(ols.moran_res[1], \
            2.59180452208,7)
        np.testing.assert_array_almost_equal(ols.moran_res[2], \
            0.00954740031251,7)
        np.testing.assert_array_almost_equal(ols.mulColli, \
            12.537554873824675 ,7)
        np.testing.assert_equal(ols.n,  49)
        np.testing.assert_equal(ols.name_ds,  'columbus')
        np.testing.assert_equal(ols.name_gwk,  None)
        np.testing.assert_equal(ols.name_w,  'unknown')
        np.testing.assert_equal(ols.name_x,  ['CONSTANT', 'income', 'crime'])
        np.testing.assert_equal(ols.name_y,  'home value')
        np.testing.assert_array_almost_equal(ols.predy[3], np.array([
            33.53969014]),7)
        np.testing.assert_array_almost_equal(ols.r2, \
                0.34951437785126105 ,7)
        np.testing.assert_array_almost_equal(ols.rlm_error, \
                (3.3394727667427513, 0.067636278225568919),7)
        np.testing.assert_array_almost_equal(ols.rlm_lag, \
            (0.17146086940258459, 0.67881673703455414), 7)
        np.testing.assert_equal(ols.robust,  'unadjusted')
        np.testing.assert_array_almost_equal(ols.schwarz, \
            414.41095054038061,7 )
        np.testing.assert_array_almost_equal(ols.sig2, \
            231.4568494392652,7 )
        np.testing.assert_array_almost_equal(ols.sig2ML, \
            217.28602192257551,7 )
        np.testing.assert_array_almost_equal(ols.sig2n, \
                217.28602192257551, 7)
 
        np.testing.assert_array_almost_equal(ols.t_stat[2][0], \
                -2.65440864272,7)
        np.testing.assert_array_almost_equal(ols.t_stat[2][1], \
                0.0108745049098,7)
    def test_OLS_Regimes(self):
        regimes = [1] * self.w.n
        regimes[:int(self.w.n/2.)] = [0] * int(self.w.n/2.)
        ols = EC.ols.OLS_Regimes(self.y, self.X, regimes, w=self.w, spat_diag=True, moran=True, \
                name_y='home value', name_x=['income','crime'], \
                name_ds='columbus', nonspat_diag=False)
        
        #np.testing.assert_array_almost_equal(ols.aic, \
        #        408.73548964604873 ,7)
        #np.testing.assert_array_almost_equal(ols.ar2, \
        #        0.32123239427957662 ,7)
        np.testing.assert_array_almost_equal(ols.betas, \
                np.array([[ 62.00971868, \
                           0.54875832, \
                          -0.73304867, \
                          13.06125529, \
                           1.5544147 , \
                          -0.08041612]]).T, 7)
        #bp = np.array([ 2, 5.7667905131212587, 0.05594449410070558])
        #ols_bp = np.array([ols.breusch_pagan['df'], ols.breusch_pagan['bp'], ols.breusch_pagan['pvalue']])
        #np.testing.assert_array_almost_equal(bp, ols_bp, 7)
        #np.testing.assert_array_almost_equal(ols.f_stat, \
        #    (12.358198885356581, 5.0636903313953024e-05), 7)
        #jb = np.array([2, 39.706155069114878, 2.387360356860208e-09])
        #ols_jb = np.array([ols.jarque_bera['df'], ols.jarque_bera['jb'], ols.jarque_bera['pvalue']])
        #np.testing.assert_array_almost_equal(ols_jb,jb, 7)
        np.testing.assert_equal(ols.k,  6)
        kb = {'df': 2, 'kb': 2.2700383871478675, 'pvalue': 0.32141595215434604}
        #for key in kb:
        #    self.assertAlmostEqual(ols.koenker_bassett[key],  kb[key], 7)
        np.testing.assert_array_almost_equal(ols.lm_error, \
            (0.48910653,  0.48432613),7)
        np.testing.assert_array_almost_equal(ols.lm_lag, \
            (0.04972876,  0.82353592), 7)
        np.testing.assert_array_almost_equal(ols.lm_sarma, \
                (0.54831396,  0.76021273), 7)
        #np.testing.assert_array_almost_equal(ols.logll, \
        #        -201.3677448230244 ,7)
        np.testing.assert_array_almost_equal(ols.mean_y, \
            38.436224469387746,7)
        np.testing.assert_array_almost_equal(ols.moran_res[0], \
            0.06993615168844197,7)
        np.testing.assert_array_almost_equal(ols.moran_res[1], \
            1.5221890693566746,7)
        np.testing.assert_array_almost_equal(ols.moran_res[2], \
            0.12796171317640753,7)
        #np.testing.assert_array_almost_equal(ols.mulColli, \
        #    12.537554873824675 ,7)
        np.testing.assert_equal(ols.n,  49)
        np.testing.assert_equal(ols.name_ds,  'columbus')
        np.testing.assert_equal(ols.name_gwk,  None)
        np.testing.assert_equal(ols.name_w,  'unknown')
        np.testing.assert_equal(ols.name_x,  ['0_-_CONSTANT', '0_-_income',
            '0_-_crime', '1_-_CONSTANT', '1_-_income', '1_-_crime'])
        np.testing.assert_equal(ols.name_y,  'home value')
        np.testing.assert_array_almost_equal(ols.predy[3], np.array([
            40.72470539]),7)
        np.testing.assert_array_almost_equal(ols.r2, \
                0.4880852959523868 ,7)
        np.testing.assert_array_almost_equal(ols.rlm_error, \
                (0.4985852 ,  0.48012244),7)
        np.testing.assert_array_almost_equal(ols.rlm_lag, \
            (0.05920743,  0.80775305), 7)
        np.testing.assert_equal(ols.robust,  'unadjusted')
        #np.testing.assert_array_almost_equal(ols.schwarz, \
        #    414.41095054038061,7 )
        np.testing.assert_array_almost_equal(ols.sig2, \
            194.858482437219,7 )
        #np.testing.assert_array_almost_equal(ols.sig2ML, \
        #    217.28602192257551,7 )
        np.testing.assert_array_almost_equal(ols.sig2n, \
                170.9982600979677, 7)
 
        np.testing.assert_array_almost_equal(ols.t_stat[2][0], \
                -3.2474266949819044,7)
        np.testing.assert_array_almost_equal(ols.t_stat[2][1], \
                0.0022611435827165284,7)

if __name__ == '__main__':
    unittest.main()
