'''
Script to launch python threads with each size to clear up memory and do
"cleaner" tests
'''

import os

folder = '~/Desktop/benchmarking'
os.system('mkdir %s'%folder)

sizes = [150, 300, 450, 600, 750, 800, 850, 900, 1000, 1250, 1500, 1750, 2000, 2500]
#sizes = [100]

methods = {'ols': 'test.large.olsSPd',\
        'lag': 'test.large.STSLS', \
        'error_het': 'test.large.spHet_error.models'}
#methods = ['error_hom']


for method in methods:
    for side in sizes:
        rs = '''
        source('large_sim_data.r')

        sink('', append=TRUE) !!!!!!!!!!!!
        %s(%i, 10)
        sink()

        '''%(methods[method], side)
        print('\n\tRUNNING side = %i'%side)
        os.system('R CMD BATCH model.r'
        break

