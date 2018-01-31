"""Configuration for test suite."""

import os

import pytest


@pytest.fixture(scope="module")
def data_loc(request):
    """Return the directory of the currently running test script"""
    test_dir = request.fspath.join("..")
    data_dir = os.path.join(test_dir, "data")
    target_dir = os.path.join(data_dir, "targets")
    cat_dir = os.path.join(data_dir, "categorical")
    ord_dir = os.path.join(data_dir, "ordinal")
    model_dir = os.path.abspath(
        os.path.join(test_dir, "..", "configs"))
    result_dir = os.path.abspath(
        os.path.join(test_dir, "..", "test_output", "pipeline"))
    try:
        os.makedirs(result_dir)
    except FileExistsError:
        pass

    return ord_dir, cat_dir, target_dir, model_dir, result_dir
