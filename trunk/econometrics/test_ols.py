import unittest
import numpy as np
import pysal
import ols as OLS

class Test_OLS(unittest.TestCase):
    def setUp(self):
        db=pysal.open("examples/columbus.dbf","r")
        y = np.array(db.by_col("CRIME"))
        y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("HOVAL"))
        X = np.array(X).T
        self.y = y
        self.X = X

    """ All methods names that begin with 'test' will be executed as a test case """
    def test_OLS_dev(self):
        ols = OLS.OLS_dev(self.X,self.y)
        # test the typical usage
        x = np.hstack((np.ones(self.y.shape),self.X))
        np.testing.assert_array_equal(ols.x, x)
        np.testing.assert_array_equal(ols.y, self.y)
        betas = np.array([[ 68.6189611 ], [ -1.59731083], [ -0.27393148]])
        np.testing.assert_array_almost_equal(ols.betas, betas, decimal=8)
        xtx = np.array([[  4.90000000e+01,   7.04371999e+02,   1.88337500e+03],
                        [  7.04371999e+02,   1.16866734e+04,   2.96004418e+04],
                        [  1.88337500e+03,   2.96004418e+04,   8.87576189e+04]])
        np.testing.assert_array_almost_equal(ols.xtx, xtx, decimal=4)
        xtxi = np.array([[  1.71498009e-01,  -7.20680548e-03,  -1.23561717e-03],
                         [ -7.20680548e-03,   8.53813203e-04,  -1.31821143e-04],
                         [ -1.23561717e-03,  -1.31821143e-04,   8.14475943e-05]])
        np.testing.assert_array_almost_equal(ols.xtxi, xtxi, decimal=8)
        u_sub = np.array([[  0.34654188], [ -3.694799  ], [ -5.28739398],
                       [-19.98551514], [  6.44754899]])
        np.testing.assert_array_almost_equal(ols.u[0:5], u_sub, decimal=8)
        predy_sub = np.array([[ 15.37943812], [ 22.496553  ], [ 35.91417498],
                          [ 52.37327514], [ 44.28396101]])
        np.testing.assert_array_almost_equal(ols.predy[0:5], predy_sub, decimal=8)
        self.assertEquals(ols.n, 49)
        self.assertEquals(ols.k, 3)
        self.assertAlmostEquals(ols.utu, 6014.892735784364, places=10)
        self.assertAlmostEquals(ols.sig2n, 122.75291297519111, places=10)
        self.assertAlmostEquals(ols.sig2n_k, 130.75853773444268, places=10)
        self.assertAlmostEquals(ols.sig2, 130.75853773444268, places=10)
        vm = np.array([[  2.24248289e+01,  -9.42351346e-01,  -1.61567494e-01],
                       [ -9.42351346e-01,   1.11643366e-01,  -1.72367399e-02],
                       [ -1.61567494e-01,  -1.72367399e-02,   1.06499683e-02]])
        np.testing.assert_array_almost_equal(ols.vm, vm, decimal=6)
        self.assertAlmostEquals(ols.mean_y, 35.128823897959187, places=10)
        self.assertAlmostEquals(ols.std_y, 16.560476353674986, places=10)

    def test_OLS(self):
        ols = OLS.OLS(self.X, self.y)
        x = np.hstack((np.ones(self.y.shape), self.X))
        # make sure OLS matches OLS_dev
        np.testing.assert_array_equal(ols.x, x)
        np.testing.assert_array_equal(ols.y, self.y)
        betas = np.array([[ 68.6189611 ], [ -1.59731083], [ -0.27393148]])
        np.testing.assert_array_almost_equal(ols.betas, betas, decimal=8)
        xtx = np.array([[  4.90000000e+01,   7.04371999e+02,   1.88337500e+03],
                        [  7.04371999e+02,   1.16866734e+04,   2.96004418e+04],
                        [  1.88337500e+03,   2.96004418e+04,   8.87576189e+04]])
        np.testing.assert_array_almost_equal(ols.xtx, xtx, decimal=4)
        xtxi = np.array([[  1.71498009e-01,  -7.20680548e-03,  -1.23561717e-03],
                         [ -7.20680548e-03,   8.53813203e-04,  -1.31821143e-04],
                         [ -1.23561717e-03,  -1.31821143e-04,   8.14475943e-05]])
        np.testing.assert_array_almost_equal(ols.xtxi, xtxi, decimal=8)
        u_sub = np.array([[  0.34654188], [ -3.694799  ], [ -5.28739398],
                       [-19.98551514], [  6.44754899]])
        np.testing.assert_array_almost_equal(ols.u[0:5], u_sub, decimal=8)
        predy_sub = np.array([[ 15.37943812], [ 22.496553  ], [ 35.91417498],
                          [ 52.37327514], [ 44.28396101]])
        np.testing.assert_array_almost_equal(ols.predy[0:5], predy_sub, decimal=8)
        self.assertEquals(ols.n, 49)
        self.assertEquals(ols.k, 3)
        self.assertAlmostEquals(ols.utu, 6014.892735784364, places=10)
        self.assertAlmostEquals(ols.sig2n, 122.75291297519111, places=10)
        self.assertAlmostEquals(ols.sig2n_k, 130.75853773444268, places=10)
        self.assertAlmostEquals(ols.sig2, 130.75853773444268, places=10)
        self.assertAlmostEquals(ols.sigML, 130.75853773444268, places=10)
        vm = np.array([[  2.24248289e+01,  -9.42351346e-01,  -1.61567494e-01],
                       [ -9.42351346e-01,   1.11643366e-01,  -1.72367399e-02],
                       [ -1.61567494e-01,  -1.72367399e-02,   1.06499683e-02]])
        np.testing.assert_array_almost_equal(ols.vm, vm, decimal=6)
        self.assertAlmostEquals(ols.mean_y, 35.128823897959187, places=10)
        self.assertAlmostEquals(ols.std_y, 16.560476353674986, places=10)
        # start testing specific attributes for the OLS class
        self.assertEquals(ols.name_ds, 'unknown')
        self.assertEquals(ols.name_x, ['CONSTANT', 'var_1', 'var_2'])
        self.assertEquals(ols.name_y, 'dep_var')
        ols = OLS.OLS(self.X, self.y, name_ds='Columbus',
                      name_x=['inc','hoval'], name_y='crime')
        self.assertEquals(ols.name_ds, 'Columbus')
        self.assertEquals(ols.name_x, ['CONSTANT', 'inc', 'hoval'])
        self.assertEquals(ols.name_y, 'crime')
        self.assertAlmostEquals(ols.r2, 0.55240404083742334, places=10)
        self.assertAlmostEquals(ols.ar2, 0.5329433469607896, places=10)
        self.assertAlmostEquals(ols.sig2, 130.75853773444268, places=10)
        self.assertAlmostEquals(ols.Fstat[0], 28.385629224694853, places=10)
        self.assertAlmostEquals(ols.Fstat[1], 9.3407471005108332e-09, places=10)
        self.assertAlmostEquals(ols.logll, -187.3772388121491, places=10)
        self.assertAlmostEquals(ols.aic, 380.7544776242982, places=10)
        self.assertAlmostEquals(ols.sc, 386.42993851863008, places=10)
        std_err = np.array([ 4.73548613,  0.33413076,  0.10319868])
        np.testing.assert_array_almost_equal(ols.std_err, std_err, decimal=8)
        t_stat = [(14.490373143689094, 9.2108899889173982e-19),
                  (-4.7804961912965762, 1.8289595070843232e-05),
                  (-2.6544086427176916, 0.010874504909754612)]
        np.testing.assert_array_almost_equal(ols.Tstat, t_stat, decimal=8)
        self.assertAlmostEquals(ols.mulColli, 6.5418277514438046, places=10)
        jb = {'df': 2, 'jb': 1.835752520075947, 'pvalue': 0.39936629124876566} 
        self.assertEquals(ols.JB, jb)
        bp = {'bp': 10.012849713093686, 'df': 2, 'pvalue': 0.0066947954259665692} 
        self.assertEquals(ols.BP, bp)
        kb = {'df': 2, 'kb': 7.2165644721877449, 'pvalue': 0.027098355486469869} 
        self.assertEquals(ols.KB, kb)
        white = {'df': 5, 'pvalue': 0.0012792228173925788, 'wh': 19.946008239903257} 
        self.assertEquals(ols.white, white)
        # test not using constant
        ols = OLS.OLS(self.X,self.y, constant=False)
        betas = np.array([[ 1.28624161], [ 0.22045774]])
        np.testing.assert_array_almost_equal(ols.betas, betas, decimal=8)



suite = unittest.TestLoader().loadTestsFromTestCase(Test_OLS)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
