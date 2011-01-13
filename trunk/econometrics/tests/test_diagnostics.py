import unittest
import numpy as np
import pysal
import diagnostics
from ols import BaseOLS as OLS
#from pysal.spreg import diagnostics
#from pysal.spreg.ols import BaseOLS as OLS 

class Fstat_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_f_stat(self):
        pass


class Tstat_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_t_stat(self):
        pass


class R2_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_r2(self):
        pass


class Ar2_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_ar2(self):
        pass


class SeBetas_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_se_betas(self):
        pass


class LogLikelihood_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_log_likelihood(self):
        pass


class Akaike_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_akaike(self):
        pass


class Schwarz_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_schwarz(self):
        pass


class ConditionIndex_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_condition_index(self):
        pass


class JarqueBera_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_jarque_bera(self):
        pass


class BreuschPagan_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_breusch_pagan(self):
        pass


class White_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_white(self):
        pass


class KoenkerBassett_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_koenker_bassett(self):
        pass


class Vif_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_vif(self):
        pass


class ConstantCheck_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_constant_check(self):
        pass


suite = unittest.TestSuite()
test_classes = [Fstat_Tester, Tstat_Tester, R2_Tester, Ar2_Tester,
                SeBetas_Tester, LogLikelihood_Tester, Akaike_Tester,
                Schwarz_Tester, ConditionIndex_Tester, JarqueBera_Tester,
                BreuschPagan_Tester, White_Tester, KoenkerBassett_Tester,
                Vif_Tester, ConstantCheck_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)

