.. _ihelp:

================
Interactive help
================

As a standard python library, PySAL has enabled direct access to the
docstrings documentation, which means you can easily read the help provided in
the actual code. Although its style is more technical, it may be
of great help in interactive sessions, for instance, when you want to know
what type of object a function is expecting as arguments or what are the
attributes of a class; and it is very handy and convenient as you only have to
fire up a python session and type in the interpreter. The interactive help in
PySAL uses the built-in help system, which operates through the command 
`help([object])`, which means you only have to type that, replacing `[object]`
by the actual object you want to obtain help about. The object can be pretty
much anything: from the name of a module, class or function to an object that
you have created with PySAL. A few examples you can type and check out are::

    import pysal

    help(pysal)

    help(pysal.esda)

    help(pysal.weights.W)

The contents you access by typing `help([object])` are exactly the same that
those hosted in `pysal.org <http://pysal.org>`_ under the `API reference
<http://pysal.org/library/index.html>`_.
