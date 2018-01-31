"""Configuration for test suite."""

import os
from collections import namedtuple

import numpy as np
import pytest
import rasterio.transform


TestImageData = namedtuple("TestImageData", ["pixel_width", "pixel_height",
                                             "origin_x", "origin_y",
                                             "width", "height", "affine"])


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


@pytest.fixture(params=list(range(100)))
def random_image_transform(request):
    """Make a bunch of random image transforms."""
    r = np.random.RandomState(request.param)
    pixel_width = r.uniform(-5, 5)
    pixel_height = r.uniform(-5, 5)
    origin_x = r.uniform(-10, 10)
    origin_y = r.uniform(-10, 10)
    width = r.randint(1, 100)
    height = r.randint(1, 200)
    affine = rasterio.transform.from_origin(origin_x,
                                            origin_y,
                                            pixel_width,
                                            pixel_height)

    data = TestImageData(pixel_width=pixel_width, pixel_height=pixel_height,
                         origin_x=origin_x, origin_y=origin_y, width=width,
                         height=height, affine=affine)
    return data
