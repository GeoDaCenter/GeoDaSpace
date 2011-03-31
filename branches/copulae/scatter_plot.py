'''
Module to generate scatter plots
'''

import pysal as ps
import numpy as np
import matplotlib.pyplot as plt
from workbench import get_sar_vector_random

def scatter_pts(w, rho=None, y=None, standardize=True):
    '''
    Build y and wy for a scatter plot
    ...

    Parameters
    ----------
    w               : pysal.W
                      Weights object
    rho             : float
                      Spatial autocorrelation coefficient
    y               : array
                      nx1 array with the variable of interest
    standardize     : boolean
                      If True (default) data are standardized following this
                      scheme:

                        .. math::
                            y* = (y - \bar{y}) / \sigma_{y}

    Returns
    -------
    y, wy   : tuple
              Tuple of arrays with the original variable and its spatial lag

    '''
    w.transform = 'r'
    if rho and y:
        raise Exception, "'scatter' only allows either 'rho' or 'y', but not both. Please provide only one of them"
    if rho:
        y = get_sar_vector_random(w=w, rho=rho)
    y = (y - np.mean(y)) / np.std(y)
    wy = ps.lag_spatial(w, y)
    return y, wy

def scatter_plot(y, wy, show=False, standardize=True):
    '''
    Builds scatter plot graph with option to plot it
    ...

    Parameters
    ----------
    y               : array
                      nx1 vector with variable of interest
    wy              : array
                      nx1 vector with spatial lag of variable of interest
    show            : boolean
                      If True triggers 'plt.show()' to display the graph
    standardize     : boolean
                      If True (default) data are standardized following this
                      scheme:

                        .. math::
                            y* = (y - \bar{y}) / \sigma_{y}

    Returns
    -------
    p               : plot
                      Matplotlibt object of graph
    '''
    mi = ps.esda.moran.Moran(y, w)
    if standardize:
        y = (y - np.mean(y)) / np.std(y)
        wy = (wy - np.mean(wy)) / np.std(wy)
    scatter = plt.scatter(y, wy)
    vbar = plt.axvline(0, color='black')
    hbar = plt.axhline(0, color='black')
    title = plt.title("Moran's I: %.4f"%mi.I)
    if show:
        plt.show()
    return scatter


if __name__ == '__main__':

    w = ps.lat2W(5, 5)
    y, wy = scatter_pts(w, rho=0.5)
    scatter = scatter_plot(y, wy, show=True, standardize=False)

