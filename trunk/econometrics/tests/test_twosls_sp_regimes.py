import unittest
import numpy as np
import pysal
from econometrics.twosls_sp_regimes import GM_Lag_Regimes

class TestGMLag_Regimes(unittest.TestCase):
    def setUp(self):
        self.w = pysal.queen_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'
        self.db = pysal.open(pysal.examples.get_path("columbus.dbf"), 'r')
        y = np.array(self.db.by_col("CRIME"))
        self.y = np.reshape(y, (49,1))
        self.r_var = 'NSA'
        self.regimes = self.db.by_col(self.r_var)

    def test___init__(self):
        #Matches SpaceStat
        X = []
        X.append(self.db.by_col("INC"))
        X.append(self.db.by_col("HOVAL"))
        self.X = np.array(X).T
        reg = GM_Lag_Regimes(self.y, self.X, self.regimes, w=self.w, sig2n_k=True) 
        betas = np.array([[ 45.14892906],
       [ -1.42593383],
       [ -0.11501037],
       [ 40.99023016],
       [ -0.81498302],
       [ -0.28391409],
       [  0.4736163 ]])
        np.testing.assert_array_almost_equal(reg.betas, betas, 7)
        e_5 = np.array([[ -1.47960519],
       [ -7.93748769],
       [ -5.88561835],
       [-13.37941105],
       [  5.2524303 ]])
        np.testing.assert_array_almost_equal(reg.e_pred[0:5], e_5, 7)
        h_0 = np.array([[  0.       ,   0.       ,   0.       ,   1.       ,  19.531    ,
         80.467003 ,   0.       ,   0.       ,  18.594    ,  35.4585005]])
        np.testing.assert_array_almost_equal(reg.h[0]*np.eye(10), h_0)
        self.assertEqual(reg.k, 7)
        self.assertEqual(reg.kstar, 1)
        self.assertAlmostEqual(reg.mean_y, 35.128823897959187, 7)
        self.assertEqual(reg.n, 49)
        self.assertAlmostEqual(reg.pr2, 0.6572182131915739, 7)
        self.assertAlmostEqual(reg.pr2_e, 0.5779687278635434, 7)
        pfora1a2 = np.array([ -2.15017629,  -0.30169328,  -0.07603704, -22.06541809,
         0.45738058,   0.02805828,   0.39073923]) 
        np.testing.assert_array_almost_equal(reg.pfora1a2[0], pfora1a2, 7)
        predy_5 = np.array([[ 13.93216104],
       [ 23.46424269],
       [ 34.43510955],
       [ 44.32473878],
       [ 44.39117516]])
        np.testing.assert_array_almost_equal(reg.predy[0:5], predy_5, 7)
        predy_e_5 = np.array([[ 17.20558519],
       [ 26.73924169],
       [ 36.51239935],
       [ 45.76717105],
       [ 45.4790797 ]])
        np.testing.assert_array_almost_equal(reg.predy_e[0:5], predy_e_5, 7)
        q_5 = np.array([[  0.       ,   0.       ,  18.594    ,  35.4585005]])
        np.testing.assert_array_almost_equal(reg.q[0]*np.eye(4), q_5)
        self.assertEqual(reg.robust, 'unadjusted')
        self.assertAlmostEqual(reg.sig2n_k, 109.76462904625834, 7)
        self.assertAlmostEqual(reg.sig2n, 94.08396775393571, 7)
        self.assertAlmostEqual(reg.sig2, 109.76462904625834, 7)
        self.assertAlmostEqual(reg.std_y, 16.732092091229699, 7)
        u_5 = np.array([[  1.79381896],
       [ -4.66248869],
       [ -3.80832855],
       [-11.93697878],
       [  6.34033484]])
        np.testing.assert_array_almost_equal(reg.u[0:5], u_5, 7)
        self.assertAlmostEqual(reg.utu, 4610.11441994285, 7)
        varb = np.array([  1.23841820e+00,  -3.65620114e-02,  -1.21919663e-03,
         1.00057547e+00,  -2.07403182e-02,  -1.27232693e-03,
        -1.77184084e-02])
        np.testing.assert_array_almost_equal(reg.varb[0], varb, 7)
        vm = np.array([  1.35934514e+02,  -4.01321561e+00,  -1.33824666e-01,
         1.09827796e+02,  -2.27655334e+00,  -1.39656494e-01,
        -1.94485452e+00])
        np.testing.assert_array_almost_equal(reg.vm[0], vm, 6)
        x_0 = np.array([[  0.      ,   0.      ,   0.      ,   1.      ,  19.531   ,
         80.467003]])
        np.testing.assert_array_almost_equal(reg.x[0]*np.eye(6), x_0, 7)
        y_5 = np.array([[ 15.72598 ],
       [ 18.801754],
       [ 30.626781],
       [ 32.38776 ],
       [ 50.73151 ]])
        np.testing.assert_array_almost_equal(reg.y[0:5], y_5, 7)
        yend_5 = np.array([[ 24.7142675 ],
       [ 26.24684033],
       [ 29.411751  ],
       [ 34.64647575],
       [ 40.4653275 ]])
        np.testing.assert_array_almost_equal(reg.yend[0:5]*np.array([[1]]), yend_5, 7)
        z_0 = np.array([[  0.       ,   0.       ,   0.       ,   1.       ,  19.531    ,
         80.467003 ,  24.7142675]]) 
        np.testing.assert_array_almost_equal(reg.z[0]*np.eye(7), z_0, 7)
        zthhthi = np.array([  1.00000000e+00,  -2.35922393e-16,   5.55111512e-17,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
        -4.44089210e-16,   2.22044605e-16,   0.00000000e+00,
         0.00000000e+00])
        np.testing.assert_array_almost_equal(reg.zthhthi[0], zthhthi, 7)
        chow_regi = np.array([[ 0.19692667,  0.65721307],
       [ 0.5666492 ,  0.45159351],
       [ 0.45282066,  0.5009985 ]])
        np.testing.assert_array_almost_equal(reg.chow.regi, chow_regi, 7)
        self.assertAlmostEqual(reg.chow.joint[0], 0.82409867601863462, 7)
    
    def test_init_discbd(self):
        #Matches SpaceStat.
        X = np.array(self.db.by_col("INC"))
        X = np.reshape(X, (49,1))
        yd = np.array(self.db.by_col("HOVAL"))
        yd = np.reshape(yd, (49,1))
        q = np.array(self.db.by_col("DISCBD"))
        q = np.reshape(q, (49,1))
        reg = GM_Lag_Regimes(self.y, X, self.regimes, yend=yd, q=q, lag_q=False, w=self.w, sig2n_k=True) 
        tbetas = np.array([[ 42.7266306 ],
       [ -0.15552345],
       [ 37.70545276],
       [ -0.5341577 ],
       [ -0.68305796],
       [ -0.37106077],
       [  0.55809516]])
        np.testing.assert_array_almost_equal(tbetas, reg.betas)
        vm = np.array([ 270.62979422,    3.62539081,  327.89638627,    6.24949355,
         -5.25333106,   -6.01743515,   -4.19290074])
        np.testing.assert_array_almost_equal(reg.vm[0], vm, 6)
        chow_regi = np.array([[ 0.13130991,  0.71707772],
       [ 0.04740966,  0.82763357],
       [ 0.15474413,  0.6940423 ]])
        np.testing.assert_array_almost_equal(reg.chow.regi, chow_regi, 7)
        self.assertAlmostEqual(reg.chow.joint[0], 0.31248100032096549, 7)
    
    def test_lag_q(self):
        X = np.array(self.db.by_col("INC"))
        X = np.reshape(X, (49,1))
        yd = np.array(self.db.by_col("HOVAL"))
        yd = np.reshape(yd, (49,1))
        q = np.array(self.db.by_col("DISCBD"))
        q = np.reshape(q, (49,1))
        reg = GM_Lag_Regimes(self.y, X, self.regimes, yend=yd, q=q, w=self.w, sig2n_k=True) 
        tbetas = np.array([[ 37.87698329],
       [ -0.89426982],
       [ 31.4714777 ],
       [ -0.71640525],
       [ -0.28494432],
       [ -0.2294271 ],
       [  0.62996544]])
        np.testing.assert_array_almost_equal(tbetas, reg.betas)
        vm = np.array([ 128.25714554,   -0.38975354,   95.7271044 ,   -1.8429218 ,
         -1.75331978,   -0.18240338,   -1.67767464])
        np.testing.assert_array_almost_equal(reg.vm[0], vm, 6)
        chow_regi = np.array([[ 0.43494049,  0.50957463],
       [ 0.02089281,  0.88507135],
       [ 0.01180501,  0.91347943]])
        np.testing.assert_array_almost_equal(reg.chow.regi, chow_regi, 7)
        self.assertAlmostEqual(reg.chow.joint[0], 0.54288190938307757, 7)
    
    def test_all_regi(self):
        X = np.array(self.db.by_col("INC"))
        X = np.reshape(X, (49,1))
        yd = np.array(self.db.by_col("HOVAL"))
        yd = np.reshape(yd, (49,1))
        q = np.array(self.db.by_col("DISCBD"))
        q = np.reshape(q, (49,1))
        reg = GM_Lag_Regimes(self.y, X, self.regimes, yend=yd, q=q, w=self.w,regime_lag=True, regime_error = False) 
        tbetas = np.array([[ 42.35827477],
       [ -0.09472413],
       [ 32.24228762],
       [ -0.12304063],
       [ -0.68794223],
       [  0.54482537],
       [ -0.46840307],
       [  0.67108156]])
        np.testing.assert_array_almost_equal(tbetas, reg.betas)
        vm = np.array([ 239.95511019,    5.44860771,    0.        ,    0.        ,
         -5.79921118,   -3.55347672,    0.        ,    0.        ])
        np.testing.assert_array_almost_equal(reg.vm[0], vm, 6)
        chow_regi = np.array([[  1.58777344e-01,   6.90284689e-01],
       [  2.90183773e-04,   9.86408868e-01],
       [  7.55292928e-02,   7.83449974e-01],
       [  1.04465790e-01,   7.46534936e-01]])
        np.testing.assert_array_almost_equal(reg.chow.regi, chow_regi, 7)
        self.assertAlmostEqual(reg.chow.joint[0], 0.38839928684452918, 7)
    
    def test_all_regi_sig2(self):
        X = np.array(self.db.by_col("INC"))
        X = np.reshape(X, (49,1))
        yd = np.array(self.db.by_col("HOVAL"))
        yd = np.reshape(yd, (49,1))
        q = np.array(self.db.by_col("DISCBD"))
        q = np.reshape(q, (49,1))
        reg = GM_Lag_Regimes(self.y, X, self.regimes, yend=yd, q=q, w=self.w,regime_lag=True, regime_error = True) 
        tbetas = np.array([[ 42.35827477],
       [ -0.09472413],
       [ -0.68794223],
       [  0.54482537],
       [ 32.24228762],
       [ -0.12304063],
       [ -0.46840307],
       [  0.67108156]])
        np.testing.assert_array_almost_equal(tbetas, reg.betas)
        vm = np.array([ 200.92894859,    4.56244927,   -4.85603079,   -2.9755413 ,
          0.        ,    0.        ,    0.        ,    0.        ])
        np.testing.assert_array_almost_equal(reg.vm[0], vm, 6)
        chow_regi = np.array([[  1.51825373e-01,   6.96797034e-01],
       [  3.20105698e-04,   9.85725412e-01],
       [  8.58836996e-02,   7.69476896e-01],
       [  1.01357290e-01,   7.50206873e-01]])
        np.testing.assert_array_almost_equal(reg.chow.regi, chow_regi, 7)
        self.assertAlmostEqual(reg.chow.joint[0], 0.38417230022512161, 7)

    def test_fixed_const(self):
        X = np.array(self.db.by_col("INC"))
        X = np.reshape(X, (49,1))
        yd = np.array(self.db.by_col("HOVAL"))
        yd = np.reshape(yd, (49,1))
        q = np.array(self.db.by_col("DISCBD"))
        q = np.reshape(q, (49,1))
        reg = GM_Lag_Regimes(self.y, X, self.regimes, yend=yd, q=q, w=self.w, constant_regi='one') 
        tbetas = np.array([[ -0.37658823],
       [ -0.9666079 ],
       [ 35.5445944 ],
       [ -0.45793559],
       [ -0.24216904],
       [  0.62500602]])
        np.testing.assert_array_almost_equal(tbetas, reg.betas)
        vm = np.array([ 1.4183697 , -0.05975784, -0.27161863, -0.62517245,  0.02266177,
        0.00312976])
        np.testing.assert_array_almost_equal(reg.vm[0], vm, 6)
        chow_regi = np.array([[  1.85767047e-01,   6.66463269e-01],
       [  1.19445012e+01,   5.48089036e-04]])
        np.testing.assert_array_almost_equal(reg.chow.regi, chow_regi, 7)
        self.assertAlmostEqual(reg.chow.joint[0], 12.017256217621382, 7)

    def test_names(self):
        y_var = 'CRIME'
        x_var = ['INC']
        x = np.array([self.db.by_col(name) for name in x_var]).T
        yd_var = ['HOVAL']
        yd = np.array([self.db.by_col(name) for name in yd_var]).T
        q_var = ['DISCBD']
        q = np.array([self.db.by_col(name) for name in q_var]).T
        r_var = 'NSA'
        reg = GM_Lag_Regimes(self.y, x, self.regimes, yend=yd, q=q, w=self.w, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='columbus', name_w='columbus.gal')
        betas = np.array([[ 37.87698329],
       [ -0.89426982],
       [ 31.4714777 ],
       [ -0.71640525],
       [ -0.28494432],
       [ -0.2294271 ],
       [  0.62996544]])
        np.testing.assert_array_almost_equal(reg.betas, betas, 7)
        vm = np.array([ 109.93469618,   -0.33407447,   82.05180377,   -1.57964725,
         -1.50284553,   -0.15634575,   -1.43800683])
        np.testing.assert_array_almost_equal(reg.vm[0], vm, 6)
        chow_regi = np.array([[ 0.50743058,  0.47625326],
       [ 0.02437494,  0.87593468],
       [ 0.01377251,  0.9065777 ]])
        np.testing.assert_array_almost_equal(reg.chow.regi, chow_regi, 7)
        self.assertAlmostEqual(reg.chow.joint[0], 0.63336222761359162, 7)
        self.assertListEqual(reg.name_x, ['0.0_CONSTANT', '0.0_INC', '1.0_CONSTANT', '1.0_INC'])
        self.assertListEqual(reg.name_yend, ['0.0_HOVAL', '1.0_HOVAL', 'Global_W_CRIME'])
        self.assertListEqual(reg.name_q, ['0.0_DISCBD', '0.0_W_INC', '0.0_W_DISCBD', '1.0_DISCBD', '1.0_W_INC', '1.0_W_DISCBD'])
        self.assertEqual(reg.name_y, y_var)

if __name__ == '__main__':
    unittest.main()
