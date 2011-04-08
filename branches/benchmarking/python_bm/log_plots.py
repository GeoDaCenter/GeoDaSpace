"""
Read the logs from benchmarking and plot them
"""

import pylab as pl

def load_log_py(log_file):
    log = open(log_file)
    n = []
    k = []
    creDa = []
    creWe = []
    ols = []
    lm = []
    moran = []
    gmswls = []
    stsls = []
    total = []
    model = 0
    for line in log:
        line = line.strip('\n').split()
        if len(line) > 0:
            if line[0] == 'Model:':
                model = line[-1]
            if line[0] == 'n:':
                n.append(line[1])
            if line[0] == 'k:':
                k.append(line[1])
            if line[0] == 'Create':
                creDa.append(line[2])
            if line[0] == 'Created':
                creWe.append(line[2])
            if line[0] == 'Regression:':
                ols.append(line[1])
            if line[0] == 'LM':
                lm.append(line[2])
            if line[0] == 'Moran':
                moran.append(line[2])
            if line[0] == 'GMSWLS:':
                gmswls.append(float(line[1]))
            if line[1] == 'STSLSk:':
                stsls.append(float(line[-1]))
            if line[0] == 'Total':
                total.append(line[3])
    log.close()
    els =  [model], n, k, creDa, creWe, ols, lm, moran, gmswls, stsls, total
    els = [map(float, i) for i in els]
    return els

def load_log_r(log_file):
    log = open(log_file)
    lines = log.readlines()
    log.close()
    n = []
    k = []
    creDa = []
    creWe = []
    total = []
    gmswls = []
    stsls = []
    lm = []
    ols = []
    moran = []
    model = 0
    for i in range(len(lines)):
        line = lines[i].strip('\n').strip('[1] ').strip('"').split(' ')
        if line[0] == 'Model:':
            model = line[-1]
        if line[0] == 'N:':
            n.append(float(line[1]))
        if line[0] == 'k:':
            k.append(line[1])
        if line[0] == 'Create':
            creDa.append(float(lines[i+2]))
        if line[0] == 'Created':
            creWe.append(float(lines[i+2]))
        if line[0] == 'Regression:':
            ols.append(float(lines[i+2]))
        if line[0] == 'Moran':
            moran.append(float(lines[i+2]))
        if line[0] == 'LM':
            lm.append(float(lines[i+2]))
        if line[0] == 'GMSWLS:':
            gmswls.append(float(lines[i+2]))
        if line[0] == 'STSLS:':
            stsls.append(float(lines[i+2]))
        if line[0] == 'Total':
            total.append(float(lines[i+2]))
    els = [model], n, k, creDa, creWe, ols, lm, moran, gmswls, stsls, total
    elsf = [model]
    elsf.extend([map(float, i) for i in els[1:]])
    return elsf


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

def write_all(log_file, outfile):
    log = open(log_file)
    n = []
    k = []
    creDa = []
    creWe = []
    reg = []
    lm = []
    moran = []
    total = []
    pack = [n, creDa, creWe, reg, lm, moran, total]
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
    fo = open(outfile, 'w')
    fo.write('n,creDa,creWe,reg,lm,moran,total\n')
    for i in range(len(n)):
        line = '%s,%s,%s,%s,%s,%s,%s\n'%tuple(map(str, [j[i] for j in pack]))
        fo.write(line)
    fo.close()
    return pack

def plot_r(log_file, title='R spdep benchmarking', pic=None):
    log = open(log_file)
    n = []
    creDa = []
    creWe = []
    reg = []
    lm = []
    moran = []
    total = []
    for line in log:
        line = line.strip('\n').split()
        if len(line) > 1:
            if line[1] == '"N:':
                n.append(float(line[2].strip('"')))
            if line[1] == '"Create':
                elapsed = log.next()
                t = log.next().strip('\n').split()
                creDa.append(float(t[0].strip('"')) / 60)
            if line[1] == '"Created':
                elapsed = log.next()
                t = log.next().strip('\n').split()
                creWe.append(float(t[0].strip('"')) / 60)
            if line[1] == '"Regression:"':
                elapsed = log.next()
                t = log.next().strip('\n').split()
                reg.append(float(t[0].strip('"')) / 60)
            if line[1] == '"LM':
                elapsed = log.next()
                t = log.next().strip('\n').split()
                lm.append(float(t[0].strip('"')) / 60)
            if line[1] == '"Moran':
                elapsed = log.next()
                t = log.next().strip('\n').split()
                moran.append(float(t[0].strip('"')) / 60)
            if line[1] == '"Total':
                elapsed = log.next()
                t = log.next().strip('\n').split()
                total.append(float(t[0].strip('"')) / 60)
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
    pl.ylabel('Minutes')
    if pic:
        pl.savefig(pic)
    else:
        pl.show()
    return [n, creDa, creWe, reg, lm, moran, total]

def write_r(log_file, outfile):
    log = open(log_file)
    n = []
    creDa = []
    creWe = []
    reg = []
    lm = []
    moran = []
    total = []
    pack = [n, creDa, creWe, reg, lm, moran, total]
    for line in log:
        line = line.strip('\n').split()
        if len(line) > 1:
            if line[1] == '"N:':
                n.append(float(line[2].strip('"')))
            if line[1] == '"Create':
                elapsed = log.next()
                t = log.next().strip('\n').split()
                creDa.append(float(t[0].strip('"')) / 60)
            if line[1] == '"Created':
                elapsed = log.next()
                t = log.next().strip('\n').split()
                creWe.append(float(t[0].strip('"')) / 60)
            if line[1] == '"Regression:"':
                elapsed = log.next()
                t = log.next().strip('\n').split()
                reg.append(float(t[0].strip('"')) / 60)
            if line[1] == '"LM':
                elapsed = log.next()
                t = log.next().strip('\n').split()
                lm.append(float(t[0].strip('"')) / 60)
            if line[1] == '"Moran':
                elapsed = log.next()
                t = log.next().strip('\n').split()
                moran.append(float(t[0].strip('"')) / 60)
            if line[1] == '"Total':
                elapsed = log.next()
                t = log.next().strip('\n').split()
                total.append(float(t[0].strip('"')) / 60)
    fo = open(outfile, 'w')
    fo.write('n,creDa,creWe,reg,lm,moran,total\n')
    for i in range(len(n)):
        line = '%s,%s,%s,%s,%s,%s,%s\n'%tuple(map(str, [j[i] for j in pack]))
        fo.write(line)
    fo.close()
    return pack

if __name__ == '__main__':

    #obj = write_all('log_sp.txt', outfile='spregLog.csv')
    #obj = write_r('large-data-simR.txt', outfile='rLog.csv')

    #obj = plot_r('large-data-simR.txt', pic='log_sp_r.png')

    #plot_all('log_sp.txt', title='SPREG benchmarking\n(only sp Diagnostics)', \
    #        pic='log_sp.png')

    #plot_all('log_all.txt', title='SPREG benchmarking\n(non-sp & sp Diagnostics)', \
    #       pic='log_all.png')

    #plot_all('large_sw.txt', title='SPREG lat2SW')
    write_all('large_sw.txt', 'large_sw_rplot.csv')
