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
	rm -rf dist/
	rm -rf build/
	find . -name "*.pyc" -exec rm '{}' ';'
	cd ~/Desktop/pysal; git pull
	cd ~/Desktop/spreg/trunk
	svn update
	/Library/Frameworks/Python.framework/Versions/2.7/bin/python  build_geodaspace.py force
	cd dist/
	tar -czvf GeoDaSpace_OSX_Nightly.tar.gz *
	cp GeoDaSpace_OSX_Nightly.tar.gz /Volumes/GeoDa/Projects/GeoDaSpace/Nightly/

clean: 
	rm -rf dist/
	rm -rf build/
	find . -name "*.pyc" -exec rm '{}' ';'
