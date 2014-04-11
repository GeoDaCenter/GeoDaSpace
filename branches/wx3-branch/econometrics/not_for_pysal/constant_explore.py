import pysal, time
from pysal.common import *
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def constant_check1(array):
    """
    Checks to see numpy array includes a constant.

    Parameters
    ----------
    array           : array
                      an array of variables to be inspected 

    Returns
    -------
    constant        : boolean
                      true signifies the presence of a constant

    Example
    -------

    """

    n,k = array.shape
    constant = False
    for j in range(k):
        variable = array[:,j]
        variable = variable.ravel()
        test = set(variable)
        #test = list(test)
        if len(test) == 1:
            constant = True
            break
    return constant

def constant_check2(array):
    """
    Checks to see numpy array includes a constant.

    Parameters
    ----------
    array           : array
                      an array of variables to be inspected 

    Returns
    -------
    constant        : boolean
                      true signifies the presence of a constant

    Example
    -------

    """

    n,k = array.shape
    constant = False
    for j in range(k):
        variable = array[:,j]
        varmin = variable.min()
        varmax = variable.max()
        if varmin == varmax:
            constant = True
            break
    return constant


# write all results to a pdf file
pp = PdfPages('constant_explore.pdf')
#fig = plt.figure()

# intialize results lists
check1time = []
check2time = []

intervals = range(10,11010,1000)

for i in intervals:
    # create empty example matrix
    example = np.empty((i,i))
    # check for constant using first method and record elapsed time
    t0 = time.time()
    constant_check1(example)
    t1 = time.time()
    elapsed = t1-t0
    check1time.append(elapsed)
    # check for constant using second method and record elapsed time
    t0 = time.time()
    constant_check2(example)
    t1 = time.time()
    elapsed = t1-t0
    check2time.append(elapsed)

# plot the results
plt.plot(intervals, check1time, label='Set Method')
plt.plot(intervals, check2time, label='Max/Min Method')
plt.xlabel(r'$n$')
plt.ylabel('Seconds')
plt.xlim(0,10500)
plt.legend(loc=2)
pp.savefig()
pp.close()



