'''
Script to launch python threads with each size to clear up memory and do
"cleaner" tests
'''

import os

folder = '~/Desktop/benchmarking'
os.system('mkdir %s'%folder)

sizes = [150, 300, 450, 600, 750, 800, 850, 900, 1000, 1250, 1500, 1750, 2000, 2500]
#sizes = [100]

methods = ['ols', 'lag', 'error_hom', 'error_het']
#methods = ['error_hom']

dist = '/Library/Frameworks/EPD64.framework/Versions/7.0/bin/python2.7'
dist = 'python'

for method in methods:
    for side in sizes:
        print('\n\tRUNNING side = %i'%side)
        os.system('%s large_sim_data.py %s %i'%(dist, method, side))
        break

