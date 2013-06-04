# developer Makefile for repeated tasks
# 
.PHONY: clean

test:
	nosetests 

doctest:
	cd doc; make pickle; make doctest

build:
	/Library/Frameworks/Python.framework/Versions/2.7/bin/python  build_geodaspace.py

nightly:
	/Library/Frameworks/Python.framework/Versions/2.7/bin/python  build_geodaspace.py force

clean: 
	rm -rf dist/
	rm -rf build/
	find . -name "*.pyc" -exec rm '{}' ';'
