'''
Build comparison table in .tex
'''

from log_plots import load_log_py, load_log_r
import matplotlib.pylab as plt
import numpy as np
import os
from mpl_toolkits.axes_grid.axislines import Subplot

def get_row(method):
    title, py, r = method
    line = '\\textbf{%s} '%title
    for i, j in zip(py, r):
        line += ' & %.4f & %.4f '%(i, j)
    line += ' \\\\ \n'
    return line

def get_top(ns):
    n = len(ns)
    tex = '''
    \\begin{tabular}{l | %s}
        '''%(' '.join([' c c |']*n))
    col = ''
    for i in ns:
        col += " & \\multicolumn{2}{| c |}{\\textbf{n = %i}}"%int(i)
    col += '\\\\ \n'
    col += ''.join([' & Spreg & R ']*n)
    col += '\\\\ \n \\hline \\\\ \n'
    tex += col
    return tex

py_link = 'logs/smAll_py.log'
model_py, n_py, k_py, creDa_py, creWe_py, ols_py, lm_py, moran_py, gmswls_py, swls_het_py, stsls_het_py, stsls_py, total_py = load_log_py(py_link)
r_link = 'logs/smAll_r.log'
model_r, n_r, k_r, creDa_r, creWe_r, ols_r, lm_r, moran_r, gmswls_r, swls_het_r, stsls_het_r, stsls_r, total_r = load_log_r(r_link)

methods = [('OLS', ols_py, ols_r), ('Moran', moran_py, moran_r), ('LM Tests',
    lm_py, lm_r), ('GM Error', gmswls_py,
    gmswls_r), ('GM Lag', stsls_py, stsls_r), ('Sp Het Error', swls_het_py, swls_het_r),
    ('Sp Het Combo', stsls_het_py, stsls_het_r)] 

tex = """
\\documentclass{article}
%\\usepackage{multicolumn}
\\begin{document}
\\begin{table}
    \\centering
"""
tex += get_top(n_r)

for method in methods:
    tex += get_row(method)

tex += """
\\end{tabular}
\\caption{Computation time (seconds)}
\\end{table}
\\end{document}
"""

print tex

fo = open('logs/table.tex', 'w')
fo.write(tex)
fo.close()

