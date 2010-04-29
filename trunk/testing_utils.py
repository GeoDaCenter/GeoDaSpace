"""
Utilities for use during the testing phase of GeoDaSpace.
"""


import numpy as np
import csv as CSV
import pysal


class Test_Data:
    """Class to generate test data.

    Parameters
    ----------
    n               : integer
                      Number of observations, where the options are 100,
                      10000, and 1000000
    variables       : integer
                      Number of variables. One is used as the dependent
                      variable, and the remainders are independent variables.
    k               : string
                      Sting indicating the average number of neighbors in the
                      weights matrix, where the options are 'small', 'medium',
                      'large'

    Attributes
    ----------
    y               : array
                      nxNone array of observations on the dependent variable
    x               : array
                      nxk array of observations on the independent variables
    y_name          : string
                      Name of the dependent variable
    x_names         : list
                      k element list containing the names of the independent
                      variables
    w               : W object
                      PySAL weights object

    Examples
    --------

    >>> reg_inputs = Test_Data(100, 5, 'medium')
    >>> reg_inputs.y_name
    'varA'
    >>> reg_inputs.w.n
    100
    >>>

    """
    def __init__(self, n=100, variables=6, k='small'):
        np.random.seed(10)
        self.y = np.random.randn(n,)
        self.x = np.random.randn(n, variables-1)
        self.y_name = 'varA'
        letters = map(chr, range(66, 91))
        self.x_names = ['var'+i for i in letters[:variables-1]]
        
        if n==100:
            if k=='small':
                w_file = 'examples/w_rook_n100_order1_k4.gal'
            if k=='medium':
                w_file = 'examples/w_rook_n100_order2_k10.gal'
            if k=='large':
                w_file = 'examples/w_rook_n100_order3_k19.gal'
        if n==10000:
            if k=='small':
                w_file = 'examples/w_rook_n10000_order1_k4.gal'
            if k=='medium':
                w_file = 'examples/w_rook_n10000_order2_k12.gal'
            if k=='large':
                w_file = 'examples/w_rook_n10000_order3_k23.gal'
        if n==1000000:
            if k=='small':
                w_file = 'examples/w_rook_n1000000_order1_k4.gal'
            if k=='medium':
                w_file = None
            if k=='large':
                w_file = None
        if w_file:
            self.w = pysal.open(w_file).read()
        


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()




