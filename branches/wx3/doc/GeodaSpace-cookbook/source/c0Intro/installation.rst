.. _installation:

############
Installation
############

This page provides instructions on how to install PySAL. Dependencies
are listed first to get you off on the right foot.

************
Dependencies
************

Before installing PySAL, make sure the following libraries are properly
installed on you machine.

*Optional, bundled installation:* 

With the `Enthought Python Distribution (EPD) <http://www.enthought.com/products/epd.php>`_, 
you will install all of PySAL's required dependencies, as well as iPython and a lot more, rather than installing dependencies one by one.
The package is free for academics, and downloadable `here <http://www.enthought.com/products/edudownload.php>`_. 

*Required for PySAL:*

* `Python <http://www.python.org/>`_ 2.5 or later
* `Numpy <http://numpy.scipy.org/>`_ 1.3 or later
* `Scipy <http://www.scipy.org/>`_ 0.7 or later

*Required to compile the documentation (optional):*

* `Sphinx <http://sphinx.pocoo.org/>`_
* `numpydoc <http://pypi.python.org/pypi/numpydoc/0.2>`_  extension to Sphinx (NOT included in EPD)

*Optional, recommended additions:*

* `iPython <http://ipython.scipy.org/moin/Download>`_
* `rtree <http://pypi.python.org/pypi/Rtree>`_ (NOT included in EPD)



************************
Source code installation
************************

As PySAL is currently in alpha stage, we are not yet preparing binary
releases. Users can grab the source code from our subversion repository using
the following instructions, depending on the operating system:

Linux and Mac OS X (and most \*nix machines)
=============================================

To download the code, open up a terminal (`/Applications/Utilities/Terminal`)
and move to the directory where you wish to download PySAL by typing:

``cd /path_to_desired/folder``

Once there, type the following command:

``svn checkout http://pysal.googlecode.com/svn/trunk/ pysal-read-only``

**Note**: mind there must be a space between 'trunk/' and 'pysal-read-only'.

This will create a folder called 'pysal-read-only' containing all the source
code into the folder you chose and will allow you to easily update any change
that is made to the code by the developer team. Since PySAL is in active and
intense development, a lot of these changes are often introduced. For this
reason it is preferable to 'tell' Python to look for PySAL in that folder
rather than properly install it as a package. You can do this by adding the
PySAL folder to the Python path. Open the bash profile (if it doesn't already
exist, just create a new text file in the home directory and name it
``.bash_profile``) by typing in the terminal:

``open ~/.bash_profile``

**Note**: replace the command ``open`` by that of a text editor if you are in Linux
(``gedit`` for instance, if you are in Ubuntu).
Now add the following line at the end of the text file:

``export PYTHONPATH=${PYTHONPATH}:"/path_to_desired/folder/pysal-read-only/"``

Save and quit the file. Source the bash profile again:

``source ~/.bash_profile``

You are all set!!! Now you can open up a fresh python session and start
enjoying PySAL, you should be able to do (within a python session)::

 import pysal
 pysal.open.check()
 PySAL File I/O understands the following file extensions:
 Ext: '.shp', Modes: ['r', 'wb', 'w', 'rb']
 Ext: '.shx', Modes: ['r', 'wb', 'w', 'rb']
 Ext: '.geoda_txt', Modes: ['r']
 Ext: '.dbf', Modes: ['r', 'w']
 Ext: '.gwt', Modes: ['r']
 Ext: '.gal', Modes: ['r', 'w']
 Ext: '.csv', Modes: ['r']
 Ext: '.wkt', Modes: ['r']


Windows
========

To be able to use PySAL, you will need a SVN client that allows you to access,
download and update the code from our repository. We recommend to use
`TortoiseSVN <http://tortoisesvn.tigris.org/>`_, which is free and very easy to
install. The following instructions assume you are using it.

First, create a folder where you want to store PySAL's code. For the sake of this
example, we will name it ``PySALsvn`` and put it in the root folder, so the
path is:
 
``C:\PySALsvn``

Right-click on the folder with the mouse and then click on 'SVN checkout'.
The 'Checkout directory should be filled with the path to your folder
(``C:\PySALsvn`` in this case). Copy and paste on the 'URL of repository'
space the following link:

``http://pysal.googlecode.com/svn/trunk/ pysal-read-only``

**Note**: mind there must be a space between 'trunk/' and 'pysal-read-only'.

Once you click 'OK', a folder called 'pysal-read-only' will be created under
``C:\PySALsvn`` and  all the code will be downloaded to your computer.

Now you have to tell Python to 'look for' PySAL in that folder whenever you
import it in a Python session. There are several ways to do this, here we
will use a very simple one that only implies creating a simple text file.
Open a text editor and create a file called ``sitecustomize.py`` located in the
Site Packages folder of you Python distribution, so the path looks more or
less like this one:
 
``C:\PythonXX\Lib\site-packages\sitecustomize.py``

where XX corresponds to the version of the Python distribution you are using
(25 for 2.5, for example).

Add to the file the following text:

``import sys
sys.path.append("C:/PySALsvn/pysal-read-only")``
 
Save and close the window.

You are all set!!! Now you should be able to do the following on a Python
interactive session (on IDLE, for instance)::

    import pysal
    pysal.open.check()
    PySAL File I/O understands the following file extensions:
    Ext: '.shp', Modes: ['r', 'wb', 'w', 'rb']
    Ext: '.shx', Modes: ['r', 'wb', 'w', 'rb']
    Ext: '.geoda_txt', Modes: ['r']
    Ext: '.dbf', Modes: ['r', 'w']
    Ext: '.gwt', Modes: ['r']
    Ext: '.gal', Modes: ['r', 'w']
    Ext: '.csv', Modes: ['r']
    Ext: '.wkt', Modes: ['r']


