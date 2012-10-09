import unittest
import pysal
import numpy as np
from econometrics import error_sp_het_regimes as SP

class TestGM_Error_Het_Regimes(unittest.TestCase):
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
        reg = SP.GM_Error_Het_Regimes(self.y, self.X, self.regimes, self.w)
        betas = np.array([[ 62.95986466],
       [ -0.15660795],
       [ -1.49054832],
       [ 60.98577615],
       [ -0.3358993 ],
       [ -0.82129289],
       [  0.54662719]])
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
        e = np.array([ 2.77847355])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e,6)
        my = 35.128823897959187
        self.assertAlmostEqual(reg.mean_y,my)
        sy = 16.732092091229699
        self.assertAlmostEqual(reg.std_y,sy)
        vm = np.array([  3.86154100e+01,  -2.51553730e-01,  -8.20138673e-01,
         1.71714184e+00,  -1.94929113e-02,   1.23118051e-01,
         0.00000000e+00])
        np.testing.assert_array_almost_equal(reg.vm[0],vm,6)
        pr2 = 0.5515791216043385
        self.assertAlmostEqual(reg.pr2,pr2)
        std_err = np.array([ 6.21412987,  0.15340022,  0.44060473,  7.6032169 ,  0.19353719,
        0.73621596,  0.13968272])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,6)
        chow_r = np.array([[ 0.04190799,  0.83779526],
       [ 0.5736724 ,  0.44880328],
       [ 0.62498575,  0.42920056]])
        np.testing.assert_array_almost_equal(reg.chow.regi,chow_r,6)
        chow_j = 0.72341901308525713
        self.assertAlmostEqual(reg.chow.joint[0],chow_j)

class TestGM_Endog_Error_Het_Regimes(unittest.TestCase):
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
        reg = SP.GM_Endog_Error_Het_Regimes(self.y, self.X, self.yd, self.q, self.regimes, self.w)
        betas = np.array([[ 77.26679984],
       [  4.45992905],
       [ 78.59534391],
       [  0.41432319],
       [ -3.20196286],
       [ -1.13672283],
       [  0.2174965 ]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([ 20.50716917])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        e = np.array([ 25.13517175])
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
        vm = np.array([ 509.66122149,  150.5845341 ,    9.64413821,    5.54782831,
        -80.95846045,   -2.25308524,   -3.2045214 ])
        np.testing.assert_array_almost_equal(reg.vm[0],vm,5)
        pr2 = 0.19776512679331681
        self.assertAlmostEqual(reg.pr2,pr2)
        std_err = np.array([ 22.57567765,  11.34616946,  17.43881791,   1.30953812,
         5.4830829 ,   0.74634612,   0.29973079])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,6)
        chow_r = np.array([[ 0.0022216 ,  0.96240654],
       [ 0.13127347,  0.7171153 ],
       [ 0.14367307,  0.70465645]])
        np.testing.assert_array_almost_equal(reg.chow.regi,chow_r,6)
        chow_j = 1.2329971019087163
        self.assertAlmostEqual(reg.chow.joint[0],chow_j)

class TestGM_Combo_Het_Regimes(unittest.TestCase):
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
        reg = SP.GM_Combo_Het_Regimes(self.y, self.X, self.regimes, self.yd, self.q, w=self.w)
        betas = np.array([[  3.69372678e+01],
       [ -8.29474998e-01],
       [  3.08667517e+01],
       [ -7.23753444e-01],
       [ -3.01900940e-01],
       [ -2.21328949e-01],
       [  6.41902155e-01],
       [ -2.45714919e-02]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([ 0.94039246])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        e_filtered = np.array([ 0.8737864])
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
        vm = np.array([ 71.26851365,  -0.58278032,  50.53169815,  -0.74561147,
        -0.79510274,  -0.10823496,  -0.98141395,   1.16575965])
        np.testing.assert_array_almost_equal(reg.vm[0],vm,4)
        pr2 = 0.6504148883602958
        self.assertAlmostEqual(reg.pr2,pr2)
        pr2_e = 0.527136896994038
        self.assertAlmostEqual(reg.pr2_e,pr2_e)
        std_err = np.array([ 8.44206809,  0.72363219,  9.85790968,  0.77218082,  0.34084146,
        0.21752916,  0.14371614,  0.39226478])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,5)
        chow_r = np.array([[ 0.54688708,  0.45959243],
       [ 0.01035136,  0.91896175],
       [ 0.03981108,  0.84185042]])
        np.testing.assert_array_almost_equal(reg.chow.regi,chow_r,6)
        chow_j = 0.78070369988354349
        self.assertAlmostEqual(reg.chow.joint[0],chow_j)

if __name__ == '__main__':
    unittest.main()
