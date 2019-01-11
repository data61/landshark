"""Configuration for test suite."""
import os
import warnings

import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)


@pytest.fixture(scope="module")
def data_loc(request):
    """Return the directory of the currently running test script"""
    test_dir = request.fspath.join("..")
    data_dir = os.path.join(test_dir, "data")
    target_dir = os.path.join(data_dir, "targets")
    cat_dir = os.path.join(data_dir, "categorical")
    con_dir = os.path.join(data_dir, "continuous")
    model_dir = os.path.abspath(
        os.path.join(test_dir, "..", "configs"))
    result_dir = os.path.abspath(
        os.path.join(test_dir, "..", "test_output", "pipeline"))
    try:
        os.makedirs(result_dir)
    except FileExistsError:
        pass

    return con_dir, cat_dir, target_dir, model_dir, result_dir
