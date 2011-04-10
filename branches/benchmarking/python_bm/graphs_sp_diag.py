'''
Plot performance in computing Spatial Diagnostics
'''

from log_plots import load_log_py, load_log_r, load_ak
import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.axes_grid.axislines import Subplot
import os

py_link = 'logs/ols_py.log'
model_py, n_py, k_py, creDa_py, creWe_py, ols_py, lm_py, moran_py, gmswls_py, swls_het_py, stsls_het_py, stsls_py, total_py = load_log_py(py_link)
r_link = 'logs/ols_r.log'
model_r, n_r, k_r, creDa_r, creWe_r, ols_r, lm_r, moran_r, gmswls_r, swls_het_r, stsls_het_r, stsls_r, total_r = load_log_r(r_link)

ak_link = 'logs/ak_py.log'
n, ak = load_ak(ak_link)


reg_fig = plt.figure(1)
#reg_sub = plt.subplot(111)
reg_sub = Subplot(reg_fig, 111)
reg_fig.add_subplot(reg_sub)

reg_sub.axis["right"].set_visible(False)
reg_sub.axis["top"].set_visible(False)

plt.plot(-1, -1, color='white')

plt.plot(n_r[:len(moran_r)], moran_r, label='R', color='red', lw=2)
plt.text(n_r[len(moran_r) - 1], moran_r[-1], 'Moran')

plt.plot(n_py[:-1], moran_py, label='Spreg', color='blue', lw=2)
plt.text(n_py[len(moran_py) - 1], moran_py[-1], 'Moran', verticalalignment='top')

plt.legend(loc=2, frameon=False)

plt.plot(n[:-1], ak, label='Spreg', color='blue', lw=2)
plt.text(n[len(ak) - 1], ak[-1], 'AK', verticalalignment='top')


plt.plot(n_r[:len(lm_r)], lm_r, color='red', lw=2)
plt.text(n_py[len(lm_r)], lm_r[-1], 'LM', verticalalignment='bottom')

plt.plot(n_py[:len(lm_py)], lm_py, color='blue', lw=2)
plt.text(n_py[len(lm_py) - 1], lm_py[-1], 'LM', verticalalignment='bottom')

xmin, xmax = plt.xlim()
plt.xlim(-(xmax-xmin)*0.15, xmax)
ymin, ymax = plt.ylim()
plt.ylim(-(ymax-ymin)*0.15, ymax)

locs, labels = plt.yticks()
labs = np.arange(6)*1000 / 60
lab = ['']
lab.extend(map(str, map(round, labs)))
plt.yticks(locs, tuple(lab))

plt.suptitle("Computation time: Spatial Diagnostics", weight='bold')
plt.xlabel('N')
plt.ylabel('Minutes')
plt.savefig('logs/sp_diag.png')

#plt.show()

