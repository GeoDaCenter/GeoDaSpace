"""
Read the logs from benchmarking and plot them
"""

import pylab as pl

def plot_all(log_file, title='SPREG benchmarking', pic=None):
    log = open(log_file)
    n = []
    k = []
    creDa = []
    creWe = []
    reg = []
    lm = []
    moran = []
    total = []
    for line in log:
        line = line.strip('\n').split()
        if line[0] == 'n:':
            n.append(line[1])
        if line[0] == 'k:':
            k.append(line[1])
        if line[0] == 'Create':
            creDa.append(line[2])
        if line[0] == 'Created':
            creWe.append(line[2])
        if line[0] == 'Regression:':
            reg.append(line[1])
        if line[0] == 'LM':
            lm.append(line[2])
        if line[0] == 'Moran':
            moran.append(line[2])
        if line[0] == 'Total':
            total.append(line[3])
    log.close()
    pl.figure()
    pl.plot(n, creDa, label='Data Generation')
    pl.plot(n, creWe, label='Weights Creating')
    pl.plot(n, reg, label='Regression')
    pl.plot(n, lm, label='LM diagnostics Creating')
    pl.plot(n, moran, label='Moran on Residuals')
    pl.plot(n, total, label='Total time')
    pl.title(title)
    pl.legend(loc=2)
    pl.xlabel('N')
    pl.ylabel('Seconds')
    if pic:
        pl.savefig(pic)
    else:
        pl.show()
    return 'Done'

#plot_all('log_sp.txt', title='SPREG benchmarking\n(only sp Diagnostics)', \
#        pic='log_sp.png')

plot_all('log_all.txt', title='SPREG benchmarking\n(non-sp & sp Diagnostics)', \
        pic='log_all.png')

