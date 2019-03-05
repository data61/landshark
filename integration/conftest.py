"""Configuration for test suite."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from typing import Tuple

import pytest
from _pytest.fixtures import FixtureRequest

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)


@pytest.fixture(scope="module")
def data_loc(request: FixtureRequest) -> Tuple[str, str, str, str, str]:
    """Return the directory of the currently running test script."""
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
