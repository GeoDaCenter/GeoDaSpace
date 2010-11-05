"""
Package-wide tests for GeoDaSpace

Currently allows for testing of docstring examples from individual modules.
Prior to commiting any changes to the trunk, said changes should be checked
against the rest of the local copy of the trunk by running::

    python tests.py

If all tests pass, changes have not broken the current trunk and can be
committed. Commits that introduce breakage should only be done in cases where
other developers are notified and the breakage raises important issues for
discussion.


Notes
-----
New modules need to be included in the `#module imports` section below, as
well as in the truncated module list where `mods` is first defined.

To deal with relative paths in the doctests a symlink must first be made from
within the `tests` directory as follows::

     ln -s ../examples .

"""

__author__ = "Sergio J. Rey <srey@asu.edu>, David C. Folch <david.folch@asu.edu>"

#### do not modify this ####
import unittest
import doctest
suite = unittest.TestSuite()
############################




# import modules with doc tests here
import ols, spHetErr, twosls
import twosls_sp, diagnostics, robust
import probit
# add modules to this list
mods = 'ols', 'spHetErr', 'twosls', 'twosls_sp',\
       'diagnostics', 'robust', 'probit'


# add unit tests here
import test_ols
suite.addTest(test_ols.suite)








#### do not modify this ####################
for mod in mods:
    suite.addTest(doctest.DocTestSuite(mod))
runner = unittest.TextTestRunner()
runner.run(suite)
############################################
