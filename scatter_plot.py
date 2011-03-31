'''
Module to generate scatter plots
'''

import pysal as ps
import numpy as np
import matplotlib.pyplot as plt
from workbench import get_sar_vector_random

class Scatter_plot:
    def __init__(self, w, rho=None, y=None, standardize=True):
        self.y, self.wy = scatter_pts(w, rho=rho, y=None) 
        self.scatter = scatter_plot(self.y, self.wy, standardize=False, rho=rho)

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

def scatter_plot(y, wy, show=False, standardize=True, rho=None):
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
    scat = plt.figure()
    sub = plt.subplot(111)
    sub.scatter(y, wy)
    vbar = sub.axvline(0, color='black')
    hbar = sub.axhline(0, color='black')
    if rho:
        title = plt.suptitle("Population $\\rho$: %.2f"%rho)
    subtitle = plt.title("Moran's I: %.4f"%mi.I)

    b, a = np.polyfit(y, wy, 1)
    x = np.linspace(np.min(y), np.max(y), 2)
    fit_line = sub.plot(x, a + b*x, color='red')
    if show:
        plt.show()
    return scat

if __name__ == '__main__':

    w = ps.lat2W(25, 25)
    rho = 0.5
    scat = Scatter_plot(w, rho=0.5, standardize=False)
    plt.show()

