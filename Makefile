.PHONY: help clean clean-pyc clean-build typecheck typecheck-xml lint lint-xml test test-xml integration

help:
	@echo "clean - clean all artefacts"
	@echo "clean-build - remove build artefacts"
	@echo "clean-pyc - remove Python file artefacts"
	@echo "typecheck - check types with mypy"
	@echo "test - run tests and check coverage"
	@echo "integration - run full pipeline test"
	@echo "lint - check style with flake8"
	@echo "typecheck-xml - check types with mypy with xml output"
	@echo "test-xml - run tests and check coverage with xml output"
	@echo "lint-xml - check style with flake8 with xml output"


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
	py.test --cov=./landshark --cache-clear --cov-fail-under=80 tests landshark

integration:
	py.test --cache-clear --boxed integration

integration-xml:
	py.test --junit-xml=test_output/integration/results.xml  --cache-clear --boxed integration

test-xml:
	py.test --junit-xml=test_output/pytest/results.xml --cov=./landshark --cov-report=html:test_output/coverage --cache-clear --cov-fail-under=80 tests landshark

