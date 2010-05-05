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
    folder          : string
                      path to the folder of the trunk in your computer.
                      Default to '' so it assumes you are on the trunk

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
    def __init__(self, n=100, variables=6, k='small', folder=''):
        np.random.seed(10)
        self.y = np.random.randn(n,)
        self.x = np.random.randn(n, variables-1)
        self.y_name = 'varA'
        letters = map(chr, range(66, 91))
        self.x_names = ['var'+i for i in letters[:variables-1]]
        
        if n==100:
            if k=='small':
                w_file = folder + 'examples/w_rook_n100_order1_k4.gal'
            if k=='medium':
                w_file = folder + 'examples/w_rook_n100_order2_k10.gal'
            if k=='large':
                w_file = folder + 'examples/w_rook_n100_order3_k19.gal'
        if n==10000:
            if k=='small':
                w_file = folder + 'examples/w_rook_n10000_order1_k4.gal'
            if k=='medium':
                w_file = folder + 'examples/w_rook_n10000_order2_k12.gal'
            if k=='large':
                w_file = folder + 'examples/w_rook_n10000_order3_k23.gal'
        if n==1000000:
            if k=='small':
                w_file = folder + 'examples/w_rook_n1000000_order1_k4.gal'
            if k=='medium':
                w_file = None
            if k=='large':
                w_file = None
        if w_file:
            self.w = pysal.open(w_file).read()

def different(a, b):
    """
    Gives back 0 if the elements of two arrays of the same shape are equal
    (rel_err<=0.000001), 1 otherwise.
    ...

    Parameters
    ----------

    a           : array
                  First array to be compared
    b           : array
                  Second array to be compared

    Returns
    -------

    Implicit    : boolean
                  0 if they are the same, 1 otherwise

    """
    flag = 0
    for i,j in zip(a.flat, b.flat):
        if rel_err(a, b)>0.000001:
            flag = 1
            break
    return flag
        
def rel_err(a, b):
    """
    Relative error between two scalars. Expression:

    ..math::

        re = \dfrac{|a - b|}{a}

    NOTE: in case 'a' is 0. and 'b' is not, the denominator is set to 'b'
    ...

    Parameters
    ----------

    a           : float, integer
                  First element to be compared
    b           : float, integer
                  Second element to be compared

    Returns
    -------

    Implicit    : float
                  Relative error of the difference
    """
    a, b = map(float, [a, b])
    if a==b:
        rel_err = 0.
    elif a!=0.:
        rel_err = np.abs(a - b) / np.abs(a)
    else:
        rel_err = np.abs(a - b) / np.abs(b)
    return rel_err

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()




