'''
Plot performance in computing STSLS and GMSWLS
'''

from log_plots import load_log_py, load_log_r
import matplotlib.pylab as plt
import numpy as np
import os
from mpl_toolkits.axes_grid.axislines import Subplot

if os.uname()[0] == 'Darwin':
    print 'Running MacOSX'
    comp = '/Users/'
elif os.uname()[0] == 'Linux':
    print 'Running Linux'
    comp = '/home/'

def convert(x, lims):
    return x / (lims[1] - lims[0])

py_link = comp + 'dani/Dropbox/aagLogs/gmswls_py.log'
model_py, n_py, k_py, creDa_py, creWe_py, ols_py, lm_py, moran_py, gmswls_py, swls_het_py, stsls_het_py, stsls_py, total_py = load_log_py(py_link)
r_link = comp + 'dani/Dropbox/aagLogs/gmswls_r.log'
model_r, n_r, k_r, creDa_r, creWe_r, ols_r, lm_r, moran_r, gmswls_r, swls_het_r, stsls_het_r, stsls_r, total_r = load_log_r(r_link)

reg_fig = plt.figure(1)
#reg_sub = plt.subplot(111)
reg_sub = Subplot(reg_fig, 111)
reg_fig.add_subplot(reg_sub)

reg_sub.axis["right"].set_visible(False)
reg_sub.axis["top"].set_visible(False)
#plt.plot(-1, -1, color='white')

plt.plot(n_r[:len(gmswls_r)], gmswls_r, label='R', color='red', lw=2)

ymin, ymax = plt.ylim()
plt.ylim(-(ymax-ymin)*0.15, ymax)

plt.text(n_r[len(gmswls_r) - 1], gmswls_r[-1], 'GM Error')
plt.axvline(x = n_r[len(gmswls_r) - 1], ymax = 
        convert(gmswls_r[-1]-reg_sub.get_ylim()[0], reg_sub.get_ylim()), 
        c='black', lw=0.5, ls='--')

plt.plot(n_py[:-1], gmswls_py, label='Spreg', color='blue', lw=2)
plt.text(n_py[len(gmswls_py) - 1], gmswls_py[-1], 'GM Error',\
        verticalalignment='center')
plt.axvline(x = n_py[len(gmswls_py) - 1], ymax = 
        convert(gmswls_py[-1]-reg_sub.get_ylim()[0], reg_sub.get_ylim()), 
        c='black', lw=0.5, ls='--')

plt.legend(loc=2, frameon=False)

py_link = comp + 'dani/Dropbox/aagLogs/stsls_py.log'
model_py, n_py, k_py, creDa_py, creWe_py, ols_py, lm_py, moran_py, gmswls_py, swls_het_py, stsls_het_py, stsls_py, total_py = load_log_py(py_link)
r_link = comp + 'dani/Dropbox/aagLogs/stsls_r.log'
model_r, n_r, k_r, creDa_r, creWe_r, ols_r, lm_r, moran_r, gmswls_r, swls_het_r, stsls_het_r, stsls_r, total_r = load_log_r(r_link)

plt.plot(n_r[:len(stsls_r)], stsls_r, color='red', lw=2)
plt.text(n_py[len(stsls_r)], stsls_r[-1], 'GM Lag', verticalalignment='bottom')
plt.axvline(x = n_r[len(stsls_r) - 1], ymax = 
        convert(stsls_r[-1]-reg_sub.get_ylim()[0], reg_sub.get_ylim()), 
        c='black', lw=0.5, ls='--')

plt.plot(n_py[:len(stsls_py)], stsls_py, color='blue', lw=2)
plt.text(n_py[len(stsls_py) - 1], stsls_py[-1], 'GM Lag', verticalalignment='center')

xmin, xmax = plt.xlim()
plt.xlim(-(xmax-xmin)*0.15, xmax)

xt = reg_sub.get_xticks()
reg_sub.set_xticks(xt[1:])
plt.suptitle("Computation time: Spatial Models", weight='bold')
plt.xlabel('N')
plt.ylabel('Seconds')
plt.savefig(comp + 'dani/Dropbox/aagGraphs/sp_error_lag.png')

#plt.show()


