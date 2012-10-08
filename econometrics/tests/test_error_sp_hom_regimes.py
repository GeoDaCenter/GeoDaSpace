import unittest
import pysal
import numpy as np
from econometrics import error_sp_hom_regimes as SP

class TestGM_Error_Hom_Regimes(unittest.TestCase):
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
        reg = SP.GM_Error_Hom_Regimes(self.y, self.X, self.regimes, self.w)
        betas = np.array([[ 62.95986466],
       [ -0.15660795],
       [ -1.49054832],
       [ 60.98577615],
       [ -0.3358993 ],
       [ -0.82129289],
       [  0.54033921]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([-2.19031456])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        predy = np.array([ 17.91629456])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        n = 49
        self.assertAlmostEqual(reg.n,n,6)
        k = 6
        self.assertAlmostEqual(reg.k,k,6)
        y = np.array([ 15.72598])
        np.testing.assert_array_almost_equal(reg.y[0],y,6)
        x = np.array([[  0.      ,   0.      ,   0.      ,   1.      ,  80.467003,  19.531   ]])
        np.testing.assert_array_almost_equal(reg.x[0].toarray(),x,6)
        e = np.array([ 2.72131648])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e,6)
        my = 35.128823897959187
        self.assertAlmostEqual(reg.mean_y,my)
        sy = 16.732092091229699
        self.assertAlmostEqual(reg.std_y,sy)
        vm = np.array([ 49.16245801,  -0.12493165,  -1.89294614,   5.71968257,
        -0.0571525 ,   0.05745855,   0.        ])
        np.testing.assert_array_almost_equal(reg.vm[0],vm,6)
        sig2 = 96.96108341267626
        self.assertAlmostEqual(reg.sig2,sig2,5)
        pr2 = 0.5515791216023577
        self.assertAlmostEqual(reg.pr2,pr2)
        std_err = np.array([ 7.01159454,  0.20701411,  0.56905515,  7.90537942,  0.10268949,
        0.56660879,  0.15659504])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,6)
        chow_r = np.array([[ 0.03888544,  0.84367579],
       [ 0.61613446,  0.43248738],
       [ 0.72632441,  0.39407719]])
        np.testing.assert_array_almost_equal(reg.chow.regi,chow_r,6)
        chow_j = 0.92133276766189676
        self.assertAlmostEqual(reg.chow.joint[0],chow_j)

class TestGM_Endog_Error_Hom_Regimes(unittest.TestCase):
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
        reg = SP.GM_Endog_Error_Hom_Regimes(self.y, self.X, self.yd, self.q, self.regimes, self.w)
        betas = np.array([[ 77.26679984],
       [  4.45992905],
       [ 78.59534391],
       [  0.41432319],
       [ -3.20196286],
       [ -1.13672283],
       [  0.22178164]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([ 20.50716917])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        e = np.array([ 25.22635318])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e,6)
        predy = np.array([-4.78118917])
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
        vm = np.array([ 403.76852704,   69.06920553,   19.8388512 ,    3.62501395,
        -40.30472224,   -1.6601927 ,   -1.64319352])
        np.testing.assert_array_almost_equal(reg.vm[0],vm,5)
        pr2 = 0.19776512679498906
        self.assertAlmostEqual(reg.pr2,pr2)
        sig2 = 644.23810259214
        self.assertAlmostEqual(reg.sig2,sig2,5)
        std_err = np.array([ 20.09399231,   7.03617703,  23.64968032,   2.176846  ,
         3.40352278,   0.92377997,   0.24462006])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,6)
        chow_r = np.array([[ 0.00191145,  0.96512749],
       [ 0.31031517,  0.57748685],
       [ 0.34994619,  0.55414359]])
        np.testing.assert_array_almost_equal(reg.chow.regi,chow_r,6)
        chow_j = 1.248410480025556
        self.assertAlmostEqual(reg.chow.joint[0],chow_j)

class TestGM_Combo_Hom_Regimes(unittest.TestCase):
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
        reg = SP.GM_Combo_Hom_Regimes(self.y, self.X, self.regimes, self.yd, self.q, w=self.w)
        betas = np.array([[ 36.93726782],
       [ -0.829475  ],
       [ 30.86675168],
       [ -0.72375344],
       [ -0.30190094],
       [ -0.22132895],
       [  0.64190215],
       [ -0.07314671]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([ 0.94039246])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        e_filtered = np.array([ 0.74211331])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e_filtered,5)
        predy_e = np.array([ 18.68732105])
        np.testing.assert_array_almost_equal(reg.predy_e[0],predy_e,6)
        predy = np.array([ 14.78558754])
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
        vm = np.array([ 111.54419614,   -0.23476709,   83.37295278,   -1.74452409,
         -1.60256796,   -0.13151396,   -1.43857915,    2.19420848])
        np.testing.assert_array_almost_equal(reg.vm[0],vm,4)
        sig2 = 95.57694234438294
        self.assertAlmostEqual(reg.sig2,sig2,4)
        pr2 = 0.6504148883591536
        self.assertAlmostEqual(reg.pr2,pr2)
        pr2_e = 0.5271368969923579
        self.assertAlmostEqual(reg.pr2_e,pr2_e)
        std_err = np.array([ 10.56144858,   0.93986958,  11.52977369,   0.61000358,
         0.44419535,   0.16191882,   0.1630835 ,   0.41107528])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,5)
        chow_r = np.array([[ 0.47406771,  0.49112176],
       [ 0.00879838,  0.92526827],
       [ 0.02943577,  0.86377672]])
        np.testing.assert_array_almost_equal(reg.chow.regi,chow_r,6)
        chow_j = 0.59098559257602923
        self.assertAlmostEqual(reg.chow.joint[0],chow_j)

if __name__ == '__main__':
    unittest.main()
