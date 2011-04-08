'''
Plot performance in running OLS
'''

from log_plots import load_log_py, load_log_r
import matplotlib.pylab as plt
import matplotlib.cm as cm
import numpy as np
import os

def lin2circ(lapses, total):
    return [i * (2 * np.pi) / total for i in lapses]

if os.uname()[0] == 'Darwin':
    print 'Running MacOSX'
    comp = '/Users/'
elif os.uname()[0] == 'Linux':
    print 'Running Linux'
    comp = '/home/'

py_link = comp + 'dani/Dropbox/aagLogs/ols_py.log'
model_py, n_py, k_py, creDa_py, creWe_py, ols_py, lm_py, moran_py, gmswls_py, stsls_py, total_py = load_log_py(py_link)
r_link = comp + 'dani/Dropbox/aagLogs/ols_r.log'
model_r, n_r, k_r, creDa_r, creWe_r, ols_r, lm_r, moran_r, gmswls_r, stsls_r, total_r = load_log_r(r_link)

pyr = [(ols_py, n_py[:len(ols_py)]), (ols_r, n_r[:len(ols_r)])]

# Plotting
fig = plt.figure(1, figsize=(8,8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

times = [np.sum(ols_py), np.sum(ols_r)] # py, R
longest = times.index(np.max(times))

lapses, heights = pyr[longest]
suma = np.sum(lapses)
lapses0 = lin2circ(lapses, suma)
lefts = [np.sum(lapses0[:i]) for i in range(len(lapses0))]
bars = ax.bar(lefts, heights, lapses0)
for bar in bars:
    bar.set_alpha(0.25)

lapses = pyr[np.abs(longest - 1)][0]
lapses = lin2circ(lapses, suma)
bars = ax.bar(lefts[:len(lapses)], heights[:len(lapses)], lapses)
for bar in bars:
    bar.set_facecolor('red')

'''
reg_fig = plt.figure(2)
reg_sub = plt.subplot(111)
plt.plot(n_r[:len(ols_r)], ols_r, label='R')
plt.plot(n_py[:len(ols_py)], ols_py, label='Spreg')
plt.legend(loc=2)
plt.title('Computation time for OLS', weight='bold')
plt.xlabel('N')
plt.ylabel('Seconds')
plt.savefig(comp + 'dani/Dropbox/aagGraphs/ols.png')
'''

plt.show()


