.PHONY: help clean clean-pyc clean-build list test test-all coverage

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests and check coverage"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

typecheck:
	mypy ./landshark

typecheck-xml:
	mypy --junit-xml=test_output/mypy/results.xml ./landshark

lint:
	py.test --flake8 ./landshark -p no:regtest --cache-clear

lint-xml:
	py.test --junit-xml=test_output/flake8/results.xml --flake8 ./landshark -p no:regtest --cache-clear

test:
	py.test --cov=./landshark --cache-clear --cov-fail-under=80 .

test-xml:
	py.test --junit-xml=test_output/pytest/results.xml --cov=./landshark --cov-report=html:test_output/coverage --cache-clear --cov-fail-under=80 .

