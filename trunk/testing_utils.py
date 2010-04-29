"""
Utilities for use during the testing phase of GeoDaSpace.
"""


import numpy as np
import csv as CSV
import pysal


class Test_Data:
    """Class to read in test data.

    Parameters
    ----------
    data_file       : string
                      Name of the CSV file containing the data to be used.
    weights_file    : string
                      Name of the GAL file to be used

    Attributes
    ----------
    y               : array
                      nx1 array of observations on the dependent variable
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

    >>> data_file = 'examples/n100_stdnorm_vars6.csv'
    >>> w_file = 'examples/w_rook_n100_order1_k4.gal'
    >>> reg_inputs = Test_Data(data_file, w_file)
    >>> reg_inputs.y_name
    'varA'
    >>> reg_inputs.w.n
    100
    >>>

    """

    def __init__(self, data_file, weights_file):
        data = pysal.open(data_file, 'r')
        self.y = np.array(data[:,0])
        self.y_name = data.header[0]
        self.x = np.array(data[:,1:])
        self.x_names = data.header[1:]
        data.close()
        self.w = pysal.open(weights_file).read()
        





def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()




