'''
Plot performance in running spatial error models (GMSWLS)
'''

from log_plots import load_log_py, load_log_r
import matplotlib.pylab as plt

py_link = '/Users/dani/Dropbox/aagLogs/gmswls_py.log'
model, n, k, creDa, creWe, gmswls_py, stsls_py, total = load_log_py(py_link)
r_link = '/Users/dani/Dropbox/aagLogs/gmswls_r.log'
model, n, k, creDa, creWe, gmswls_r, stsls_r, total = load_log_r(r_link)

# Spatial Error
gmswls_fig = plt.figure(1)
gmswls_sub = plt.subplot(111)
plt.plot(n, gmswls_r, label='R')
plt.plot(n, gmswls_py, label='Spreg')
plt.legend(loc=2)
plt.title('Computation time for Spatial Error - GMSWLS', weight='bold')
plt.xlabel('N')
plt.ylabel('Seconds')
plt.savefig('/Users/dani/Desktop/gmswls.png')

# Spatial Lag
py_link = '/Users/dani/Dropbox/aagLogs/stsls_py.log'
model, n, k, creDa, creWe, gmswls_py, stsls_py, total = load_log_py(py_link)
r_link = '/Users/dani/Dropbox/aagLogs/stsls_r.log'
model, n, k, creDa, creWe, gmswls_r, stsls_r, total = load_log_r(r_link)

stsls_fig = plt.figure(2)
stsls_sub = plt.subplot(111)
plt.plot(n[:len(stsls_r)], stsls_r, label='R')
plt.plot(n[:len(stsls_py)], stsls_py, label='Spreg')
plt.title('Computation time for Spatial Lag - STSLS', weight='bold')
plt.xlabel('N')
plt.ylabel('Seconds')
plt.legend(loc=2)
plt.savefig('/Users/dani/Desktop/stsls.png')


#plt.show()

