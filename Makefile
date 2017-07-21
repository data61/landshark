.PHONY: help clean clean-pyc clean-build list test test-all coverage

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "coverage - check code coverage quickly with the default Python"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

lint:
	py.test -s --junit-xml=test_output/flake8/results.xml --flake8 ./landshark -p no:regtest --cache-clear

test:
	py.test -s --junit-xml=test_output/pytest/results.xml --cache-clear .

coverage:
	py.test -s --junit-xml=test_output/pytest/results.xml --cov=./landshark --cov-report=html:test_output/coverage --cache-clear --cov-fail-under=80 .

