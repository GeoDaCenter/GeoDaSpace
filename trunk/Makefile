# developer Makefile for repeated tasks
# 
.PHONY: clean

test:
	nosetests 

doctest:
	cd doc; make pickle; make doctest


clean: 
	rm -rf dist/*
	rm -rf build/*
	find . -name "*.pyc" -exec rm '{}' ';'
