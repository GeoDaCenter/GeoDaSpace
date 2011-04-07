'''
Plot performance in running spatial Het error models
'''

from log_plots import load_log_py, load_log_r
import matplotlib.pylab as plt

py_link = 'logs/spHet_models.log'
model, n, k, creDa, creWe, gmswls_py, stsls_py, total = load_log_py(py_link)
r_link = '../r_bm/logs/spHet_models.log'
model, n, k, creDa, creWe, gmswls_r, stsls_r, total = load_log_r(r_link)

gmswls_fig = plt.figure(1)
gmswls_sub = plt.subplot(111)
plt.plot(n[:len(gmswls_r)], gmswls_r, label='R')
plt.plot(n[:len(gmswls_py)], gmswls_py, label='Spreg')
plt.legend(loc=2)
plt.title('Computation time for Spatial Het Error - GMSWLS', weight='bold')
plt.xlabel('N')
plt.ylabel('Seconds')
plt.savefig('/home/dani/Desktop/gmswls.png')

stsls_fig = plt.figure(2)
stsls_sub = plt.subplot(111)
plt.plot(n, stsls_r, label='R')
plt.plot(n, stsls_py, label='Spreg')
plt.title('Computation time for Spatial Lag - STSLS', weight='bold')
plt.xlabel('N')
plt.ylabel('Seconds')
plt.legend(loc=2)
plt.savefig('/home/dani/Desktop/stsls.png')


#plt.show()

