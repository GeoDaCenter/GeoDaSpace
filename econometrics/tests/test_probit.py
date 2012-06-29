import unittest
import pysal
import numpy as np
from econometrics import probit as PB

class TestBaseProbit(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        y = np.reshape(y, (49,1))
        self.y = (y>35).astype(float)
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.X = np.hstack((np.ones(self.y.shape),self.X))
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = PB.BaseProbit(self.y, self.X, w=self.w)
        betas = np.array([[-1.22299406], [ 0.1389295 ], [-0.02278668]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        predy = np.array([ 0.87120278])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        n = 49
        self.assertAlmostEqual(reg.n,n,6)
        k = 3
        self.assertAlmostEqual(reg.k,k,6)
        y = np.array([ 1.])
        np.testing.assert_array_almost_equal(reg.y[0],y,6)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,6)
        vm = np.array([[  1.77441832e+00,  -7.76306260e-02,  -1.91072154e-02], [ -7.76306260e-02,   4.09217850e-03,   6.23776512e-04], [ -1.91072154e-02,   6.23776512e-04,   3.00537923e-04]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)
        xmean = np.array([[  1.        ], [ 14.37493876], [ 35.1288239 ]])
        np.testing.assert_array_almost_equal(reg.xmean,xmean,6)        
        predpc = 71.428571428571431
        self.assertAlmostEqual(reg.predpc,predpc,5)
        logl = -23.410944683602331
        self.assertAlmostEqual(reg.logl,logl,5)
        scale = 0.26537273623801372
        self.assertAlmostEqual(reg.scale,scale,5)
        slopes = np.array([[ 0.0368681 ], [-0.00604696]])
        np.testing.assert_array_almost_equal(reg.slopes,slopes,6)
        slopes_vm = np.array([[  2.89828674e-04,   4.38969540e-05], [  4.38969540e-05,   2.11305873e-05]])
        np.testing.assert_array_almost_equal(reg.slopes_vm,slopes_vm,6)
        LR = 20.922745937778785
        self.assertAlmostEqual(reg.LR[0],LR,5)
        Pinkse_error = 6.2758465862873303
        self.assertAlmostEqual(reg.Pinkse_error[0],Pinkse_error,5)
        KP_error = 2.6273304744650208
        self.assertAlmostEqual(reg.KP_error[0],KP_error,5)
        PS_error = 4.6568643359267234
        self.assertAlmostEqual(reg.PS_error[0],PS_error,5)

class TestProbit(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        y = np.reshape(y, (49,1))
        self.y = (y>35).astype(float)
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = PB.Probit(self.y, self.X, w=self.w)
        betas = np.array([[-1.22299406], [ 0.1389295 ], [-0.02278668]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        predy = np.array([ 0.87120278])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        n = 49
        self.assertAlmostEqual(reg.n,n,6)
        k = 3
        self.assertAlmostEqual(reg.k,k,6)
        y = np.array([ 1.])
        np.testing.assert_array_almost_equal(reg.y[0],y,6)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,6)
        vm = np.array([[  1.77441832e+00,  -7.76306260e-02,  -1.91072154e-02], [ -7.76306260e-02,   4.09217850e-03,   6.23776512e-04], [ -1.91072154e-02,   6.23776512e-04,   3.00537923e-04]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)
        xmean = np.array([[  1.        ], [ 14.37493876], [ 35.1288239 ]])
        np.testing.assert_array_almost_equal(reg.xmean,xmean,6)        
        predpc = 71.428571428571431
        self.assertAlmostEqual(reg.predpc,predpc,5)
        logl = -23.410944683602331
        self.assertAlmostEqual(reg.logl,logl,5)
        scale = 0.26537273623801372
        self.assertAlmostEqual(reg.scale,scale,5)
        slopes = np.array([[ 0.0368681 ], [-0.00604696]])
        np.testing.assert_array_almost_equal(reg.slopes,slopes,6)
        slopes_vm = np.array([[  2.89828674e-04,   4.38969540e-05], [  4.38969540e-05,   2.11305873e-05]])
        np.testing.assert_array_almost_equal(reg.slopes_vm,slopes_vm,6)
        LR = 20.922745937778785
        self.assertAlmostEqual(reg.LR[0],LR,5)
        Pinkse_error = 6.2758465862873303
        self.assertAlmostEqual(reg.Pinkse_error[0],Pinkse_error,5)
        KP_error = 2.6273304744650208
        self.assertAlmostEqual(reg.KP_error[0],KP_error,5)
        PS_error = 4.6568643359267234
        self.assertAlmostEqual(reg.PS_error[0],PS_error,5)
        
if __name__ == '__main__':
    unittest.main()
