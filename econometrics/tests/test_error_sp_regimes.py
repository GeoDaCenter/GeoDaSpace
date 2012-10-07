import unittest
import pysal
import numpy as np
from econometrics import error_sp_regimes as SP

class TestGM_Error_Regimes(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("CRIME"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("HOVAL"))
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        self.w = pysal.queen_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'
        self.r_var = 'NSA'
        self.regimes = db.by_col(self.r_var)
        
    def test_model(self):
        reg = SP.GM_Error_Regimes(self.y, self.X, self.regimes, self.w)
        betas = np.array([[ 63.3443073 ],
       [ -0.15468   ],
       [ -1.52186509],
       [ 61.40071412],
       [ -0.33550084],
       [ -0.85076108],
       [  0.38671608]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([-2.06177251])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        predy = np.array([ 17.78775251])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        n = 49
        self.assertAlmostEqual(reg.n,n,6)
        k = 6
        self.assertAlmostEqual(reg.k,k,6)
        y = np.array([ 15.72598])
        np.testing.assert_array_almost_equal(reg.y[0],y,6)
        x = np.array([[  0.      ,   0.      ,   0.      ,   1.      ,  80.467003,  19.531   ]])
        np.testing.assert_array_almost_equal(reg.x[0].toarray(),x,6)
        e = np.array([ 1.40747232])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e,6)
        my = 35.128823897959187
        self.assertAlmostEqual(reg.mean_y,my)
        sy = 16.732092091229699
        self.assertAlmostEqual(reg.std_y,sy)
        vm = np.array([ 50.55875289,  -0.14444487,  -2.05735489,   0.        ,
         0.        ,   0.        ])
        np.testing.assert_array_almost_equal(reg.vm[0],vm,6)
        sig2 = 102.13050615267227
        self.assertAlmostEqual(reg.sig2,sig2,5)
        pr2 = 0.5525102200608539
        self.assertAlmostEqual(reg.pr2,pr2)
        std_err = np.array([ 7.11046784,  0.21879293,  0.58477864,  7.50596504,  0.10800686,
        0.57365981])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,6)
        chow_r = np.array([[ 0.03533785,  0.85088948],
       [ 0.54918491,  0.45865093],
       [ 0.67115641,  0.41264872]])
        np.testing.assert_array_almost_equal(reg.chow.regi,chow_r,6)
        chow_j = 0.81985446000130979
        self.assertAlmostEqual(reg.chow.joint[0],chow_j)

class TestGM_Endog_Error_Regimes(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("CRIME"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        yd = []
        yd.append(db.by_col("HOVAL"))
        self.yd = np.array(yd).T
        q = []
        q.append(db.by_col("DISCBD"))
        self.q = np.array(q).T
        self.w = pysal.queen_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'
        self.r_var = 'NSA'
        self.regimes = db.by_col(self.r_var)

    def test_model(self):
        reg = SP.GM_Endog_Error_Regimes(self.y, self.X, self.yd, self.q, self.regimes, self.w)
        betas = np.array([[ 77.48384119],
       [  4.52986158],
       [ 78.9320719 ],
       [  0.42186212],
       [ -3.23823614],
       [ -1.14757678],
       [  0.20222208]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([ 20.89658342])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        e = np.array([ 25.21818196])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e,6)
        predy = np.array([-5.17060342])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        n = 49
        self.assertAlmostEqual(reg.n,n)
        k = 6
        self.assertAlmostEqual(reg.k,k)
        y = np.array([ 15.72598])
        np.testing.assert_array_almost_equal(reg.y[0],y,6)
        x = np.array([[  0.   ,   0.   ,   1.   ,  19.531]])
        np.testing.assert_array_almost_equal(reg.x[0].toarray(),x,6)
        yend = np.array([[  0.      ,  80.467003]])
        np.testing.assert_array_almost_equal(reg.yend[0].toarray(),yend,6)
        z = np.array([[  0.      ,   0.      ,   1.      ,  19.531   ,   0.      ,
         80.467003]])
        np.testing.assert_array_almost_equal(reg.z[0].toarray(),z,6)
        my = 35.128823897959187
        self.assertAlmostEqual(reg.mean_y,my)
        sy = 16.732092091229699
        self.assertAlmostEqual(reg.std_y,sy)
        vm = np.array([ 390.88230862,   52.2591731 ,    0.        ,    0.        ,
        -32.64271168,    0.        ])
        np.testing.assert_array_almost_equal(reg.vm[0],vm,5)
        pr2 = 0.19624004251854663
        self.assertAlmostEqual(reg.pr2,pr2)
        sig2 = 649.4001905308783
        self.assertAlmostEqual(reg.sig2,sig2,5)
        std_err = np.array([ 19.77074375,   6.07666754,  24.32254318,   2.17776787,
         2.97078325,   0.94392357])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,6)
        chow_r = np.array([[ 0.0021348 ,  0.96314775],
       [ 0.40499741,  0.5245196 ],
       [ 0.4498365 ,  0.50241261]])
        np.testing.assert_array_almost_equal(reg.chow.regi,chow_r,6)
        chow_j = 1.2885586296152094
        self.assertAlmostEqual(reg.chow.joint[0],chow_j)

class TestGM_Combo_Regimes(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("CRIME"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        yd = []
        yd.append(db.by_col("HOVAL"))
        self.yd = np.array(yd).T
        q = []
        q.append(db.by_col("DISCBD"))
        self.q = np.array(q).T
        self.w = pysal.queen_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'
        self.r_var = 'NSA'
        self.regimes = db.by_col(self.r_var)

    def test_model(self):
        reg = SP.GM_Combo_Regimes(self.y, self.X, self.regimes, self.yd, self.q, w=self.w)
        predy_e = np.array([ 18.82774339])
        np.testing.assert_array_almost_equal(reg.predy_e[0],predy_e,6)
        betas = np.array([[ 36.44798052],
       [ -0.7974482 ],
       [ 30.53782661],
       [ -0.72602806],
       [ -0.30953121],
       [ -0.21736652],
       [  0.64801059],
       [ -0.16601265]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([ 0.84393304])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        e_filtered = np.array([ 0.4040027])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e_filtered,5)
        predy = np.array([ 14.88204696])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        n = 49
        self.assertAlmostEqual(reg.n,n)
        k = 7
        self.assertAlmostEqual(reg.k,k)
        y = np.array([ 15.72598])
        np.testing.assert_array_almost_equal(reg.y[0],y,6)
        x = np.array([[  0.   ,   0.   ,   1.   ,  19.531]])
        np.testing.assert_array_almost_equal(reg.x[0].toarray(),x,6)
        yend = np.array([[  0.       ,  80.467003 ,  24.7142675]])
        np.testing.assert_array_almost_equal(reg.yend[0].toarray(),yend,6)
        z = np.array([[  0.       ,   0.       ,   1.       ,  19.531    ,   0.       ,
         80.467003 ,  24.7142675]])
        np.testing.assert_array_almost_equal(reg.z[0].toarray(),z,6)
        my = 35.128823897959187
        self.assertAlmostEqual(reg.mean_y,my)
        sy = 16.732092091229699
        self.assertAlmostEqual(reg.std_y,sy)
        vm = np.array([ 109.23549239,   -0.19754121,   84.29574673,   -1.99317178,
         -1.60123994,   -0.1252719 ,   -1.3930344 ])
        np.testing.assert_array_almost_equal(reg.vm[0],vm,4)
        sig2 = 94.98610921110007
        self.assertAlmostEqual(reg.sig2,sig2,4)
        pr2 = 0.6493586702255537
        self.assertAlmostEqual(reg.pr2,pr2)
        pr2_e = 0.5255332447240576
        self.assertAlmostEqual(reg.pr2_e,pr2_e)
        std_err = np.array([ 10.45157846,   0.93942923,  11.38484969,   0.60774708,
         0.44461334,   0.15871227,   0.15738141])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,5)
        chow_r = np.array([[ 0.49716076,  0.48075032],
       [ 0.00405377,  0.94923363],
       [ 0.03866684,  0.84411016]])
        np.testing.assert_array_almost_equal(reg.chow.regi,chow_r,6)
        chow_j = 0.64531386285872072
        self.assertAlmostEqual(reg.chow.joint[0],chow_j)

if __name__ == '__main__':
    unittest.main()
