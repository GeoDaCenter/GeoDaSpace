'''
Plot performance in creating lattice (queen) weights
'''

from log_plots import load_log_py, load_log_r
import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.axes_grid.axislines import Subplot
import os

py_link = 'logs/ols_py.log'
model_py, n_py, k_py, creDa_py, creWe_py, ols_py, lm_py, moran_py, gmswls_py, swls_het_py, stsls_het_py, stsls_py, total_py = load_log_py(py_link)
r_link = 'logs/ols_r.log'
model_r, n_r, k_r, creDa_r, creWe_r, ols_r, lm_r, moran_r, gmswls_r, swls_het_r, stsls_het_r, stsls_r, total_r = load_log_r(r_link)


reg_fig = plt.figure(1)
#reg_sub = plt.subplot(111)
reg_sub = Subplot(reg_fig, 111)
reg_fig.add_subplot(reg_sub)
reg_sub.axis["right"].set_visible(False)
reg_sub.axis["top"].set_visible(False)

plt.plot(n_r[:len(creWe_r)], creWe_r, label='R', lw=2, c='red')
plt.plot(n_py[:len(creWe_py)], creWe_py, label='Spreg', lw=2, c='blue')

xmin, xmax = plt.xlim()
plt.xlim(-(xmax-xmin)*0.15, xmax)
ymin, ymax = plt.ylim()
plt.ylim(-(ymax-ymin)*0.15, ymax)

locs, labels = plt.yticks()
labs = np.arange(6)*1000 / 60
lab = ['']
lab.extend(map(str, map(round, labs)))
plt.yticks(locs, tuple(lab))

plt.legend(loc=2, frameon=False)
plt.suptitle('Computation time to create lattice (queen) weights', weight='bold')
plt.xlabel('N')
plt.ylabel('Minutes')
plt.savefig('logs/weights.png')

#plt.show()

