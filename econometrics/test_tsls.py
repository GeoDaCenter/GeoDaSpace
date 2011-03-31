import unittest
import numpy as np
import pysal
import twosls as TSLS

class Test_TSLS(unittest.TestCase):
    """ setUp is called before each test function execution """
    def setUp(self):
        db=pysal.open("examples/columbus.dbf","r")
        y = np.array(db.by_col("CRIME"))
        y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X = np.array(X).T
        yd = []
        yd.append(db.by_col("HOVAL"))
        yd = np.array(yd).T
        q = []
        q.append(db.by_col("DISCBD"))
        q = np.array(q).T
        self.y = y
        self.X = X
        self.yd = yd
        self.q = q

    """ All method names that begin with 'test' will be executed as a test case """
    def test_BaseTSLS(self):
        tsls = TSLS.BaseTSLS(self.y, self.X, self.yd, self.q)
        # test the typical usage
        x = np.hstack((np.ones(self.y.shape),self.X))
        np.testing.assert_array_equal(tsls.x, x)
        z = np.hstack((x, self.yd))   
        np.testing.assert_array_equal(tsls.z, z)
        h = np.hstack((x, self.q))   
        np.testing.assert_array_equal(tsls.h, h)
        np.testing.assert_array_equal(tsls.q, self.q)
        np.testing.assert_array_equal(tsls.yend, self.yd)
        np.testing.assert_array_equal(tsls.y, self.y)

        betas = np.array([[ 88.46579584], [  0.5200379 ], [ -1.58216593]])
        np.testing.assert_array_almost_equal(tsls.betas, betas, decimal=8)
        u_sub = np.array([[ 44.41547467], [-10.19309584], [-24.44666724],
                        [ -5.87833504], [ -6.83994851]])
        np.testing.assert_array_almost_equal(tsls.u[0:5], u_sub, decimal=8)
        predy_sub = np.array([[-28.68949467], [ 28.99484984], [ 55.07344824],
                    [ 38.26609504], [ 57.57145851]])
        np.testing.assert_array_almost_equal(tsls.predy[0:5], predy_sub, decimal=8)
        self.assertEquals(tsls.n, 49)
        self.assertEquals(tsls.k, 3)
        self.assertAlmostEquals(tsls.utu, 27028.127012242025, places=10)
        self.assertAlmostEquals(tsls.sig2n, 551.59442882126586, places=10)
        self.assertAlmostEquals(tsls.sig2n_k, 587.56797852700049, places=10)
        self.assertAlmostEquals(tsls.sig2, 551.59442882126586, places=10)
        vm = np.array([[ 229.05640809,   10.36945783,   -9.54463414],
                        [  10.36945783,    2.0013142 ,   -1.01826408],
                        [  -9.54463414,   -1.01826408,    0.62914915]])
        np.testing.assert_array_almost_equal(tsls.vm, vm, decimal=6)
        self.assertAlmostEquals(tsls.mean_y, 35.128823897959187, places=10)
        self.assertAlmostEquals(tsls.std_y, 16.732092091229699, places=10)

    def test_TSLS(self):
        tsls = TSLS.TSLS(self.y, self.X, self.yd, self.q)
        # test that TSLS matches Base_TSLS
        x = np.hstack((np.ones(self.y.shape),self.X))
        np.testing.assert_array_equal(tsls.x, x)
        z = np.hstack((x, self.yd))   
        np.testing.assert_array_equal(tsls.z, z)
        h = np.hstack((x, self.q))   
        np.testing.assert_array_equal(tsls.h, h)
        np.testing.assert_array_equal(tsls.q, self.q)
        np.testing.assert_array_equal(tsls.yend, self.yd)
        np.testing.assert_array_equal(tsls.y, self.y)

        betas = np.array([[ 88.46579584], [  0.5200379 ], [ -1.58216593]])
        np.testing.assert_array_almost_equal(tsls.betas, betas, decimal=8)
        u_sub = np.array([[ 44.41547467], [-10.19309584], [-24.44666724],
                        [ -5.87833504], [ -6.83994851]])
        np.testing.assert_array_almost_equal(tsls.u[0:5], u_sub, decimal=8)
        predy_sub = np.array([[-28.68949467], [ 28.99484984], [ 55.07344824],
                    [ 38.26609504], [ 57.57145851]])
        np.testing.assert_array_almost_equal(tsls.predy[0:5], predy_sub, decimal=8)
        self.assertEquals(tsls.n, 49)
        self.assertEquals(tsls.k, 3)
        self.assertAlmostEquals(tsls.utu, 27028.127012242025, places=10)
        self.assertAlmostEquals(tsls.sig2n, 551.59442882126586, places=10)
        self.assertAlmostEquals(tsls.sig2n_k, 587.56797852700049, places=10)
        self.assertAlmostEquals(tsls.sig2, 551.59442882126586, places=10)
        vm = np.array([[ 229.05640809,   10.36945783,   -9.54463414],
                        [  10.36945783,    2.0013142 ,   -1.01826408],
                        [  -9.54463414,   -1.01826408,    0.62914915]])
        np.testing.assert_array_almost_equal(tsls.vm, vm, decimal=6)
        self.assertAlmostEquals(tsls.mean_y, 35.128823897959187, places=10)
        self.assertAlmostEquals(tsls.std_y, 16.732092091229699, places=10)

        # start testing specific attributes for the TSLS class
        std_err = np.array([ 15.13460961,   1.41467812,   0.79318923])
        np.testing.assert_array_almost_equal(tsls.std_err, std_err, decimal=8)
        z_stat = [(5.8452644704592283, 5.05764077974161e-09),
                  (0.36760156683559597, 0.71317034634659371),
                  (-1.9946891307831864, 0.046076795581400903)]
        np.testing.assert_array_almost_equal(tsls.z_stat, z_stat, decimal=8)
        self.assertAlmostEquals(tsls.sig2, 551.59442882126586, places=10)
        ##########################################################    
        # currently does not include non-spatial model diagnostics
        ##########################################################  

        # test generic variable names
        self.assertEquals(tsls.name_ds, 'unknown')
        self.assertEquals(tsls.name_x, ['CONSTANT', 'var_1'])
        self.assertEquals(tsls.name_y, 'dep_var')
        self.assertEquals(tsls.name_yend, ['endogenous_1'])
        self.assertEquals(tsls.name_z, ['CONSTANT', 'var_1', 'endogenous_1'])
        self.assertEquals(tsls.name_q, ['instrument_1'])
        self.assertEquals(tsls.name_h, ['CONSTANT', 'var_1', 'instrument_1'])
        # test variable names
        tsls = TSLS.TSLS(self.y, self.X, self.yd, self.q, name_ds='Columbus',
                      name_x=['inc'], name_yend=['hoval'], name_y='crime',
                      name_q=['discbd'])
        self.assertEquals(tsls.name_ds, 'Columbus')
        self.assertEquals(tsls.name_x, ['CONSTANT', 'inc'])
        self.assertEquals(tsls.name_y, 'crime')
        self.assertEquals(tsls.name_yend, ['hoval'])
        self.assertEquals(tsls.name_z, ['CONSTANT', 'inc', 'hoval'])
        self.assertEquals(tsls.name_q, ['discbd'])
        self.assertEquals(tsls.name_h, ['CONSTANT', 'inc', 'discbd'])
        # test not using constant
        tsls = TSLS.TSLS(self.y, self.X, self.yd, self.q, constant=False)
        betas = np.array([[ 2.61999431], [-0.30612671]])
        np.testing.assert_array_almost_equal(tsls.betas, betas, decimal=8)
        # test robust results
        ###################################################    
        # currently don't have any software to test against
        ###################################################    

        # test spatial diagnostics
        w = pysal.open('examples/columbus.gal', 'r').read()
        w.transform = 'r'
        tsls = TSLS.TSLS(self.y, self.X, self.yd, self.q, w=w)
        ###################################################    
        # waiting on finalized AK Test
        ###################################################    
        


suite = unittest.TestLoader().loadTestsFromTestCase(Test_TSLS)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
