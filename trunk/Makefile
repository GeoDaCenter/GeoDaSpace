# developer Makefile for repeated tasks
# 
.PHONY: clean

test:
	nosetests 

doctest:
	cd doc; make pickle; make doctest

space:
	/Library/Frameworks/Python.framework/Versions/2.7/bin/python  build_geodaspace.py

clean: 
	rm -rf dist/*
	rm -rf build/*
	find . -name "*.pyc" -exec rm '{}' ';'
